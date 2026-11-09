"""The relay pipeline.

Five steps, one tool, one loop, one branch:

  1. classify  - decide which papers and which query terms matter (model call)
  2. retrieve  - the tool: BM25 over the chunk index
  3. expand    - LOOP: if the best score is weak, broaden the query and retry
  4. draft     - streamed answer with citations
  5. verify    - BRANCH: citations must resolve to retrieved pages; if not,
                 retry the draft once, then refuse rather than guess

Every step is timed so the report can split a slow request into waiting,
working and retrying, before and after the first token.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from relay.config import (
    PER_CALL_TIMEOUT_S, REQUEST_DEADLINE_S, RETRIEVAL_SCORE_FLOOR,
)
from relay.corpus import BM25, Chunk, est_tokens
from relay.limits import RateLimiter, call_with_retries
from relay.model import ExtractiveDrafter, GenResult, citations_in

REFUSAL = (
    "I can't answer that from the three papers in the library. "
    "Nothing in Attention Is All You Need, Moshi, or Dynamic Chunking covers it, "
    "so I won't guess."
)

CLASSIFY_PROMPT = (
    "You route questions to a 3-paper library: attention (Transformers), "
    "moshi (speech-text dialogue), dynamic_chunking (H-Net chunking). "
    "Reply with JSON only: {\"papers\": [keys], \"keywords\": [up to 6 terms]}. "
    "If unsure, include all three keys."
)

DRAFT_PROMPT = (
    "Answer ONLY from the numbered passages. Cite every claim as [paper p.N] "
    "using the passage's tag. If the passages do not answer the question, say "
    "exactly that you cannot answer from the papers and do not guess. "
    "Be concise: at most four sentences."
)


@dataclass
class Phase:
    classify_ms: float = 0.0
    retrieve_ms: float = 0.0
    expand_ms: float = 0.0
    draft_ms: float = 0.0
    verify_ms: float = 0.0
    retry_ms: float = 0.0
    queue_wait_ms: float = 0.0


@dataclass
class Answer:
    text: str = ""
    refused: bool = False
    citations: set = field(default_factory=set)
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    status: str = "ok"
    retries: int = 0
    steps: int = 0
    passages: list = field(default_factory=list)
    phases: Phase = field(default_factory=Phase)


class Pipeline:
    def __init__(
        self,
        index: BM25,
        limiter: RateLimiter,
        model=None,                 # _OpenAICompatible | None
        retrieval: dict | None = None,
        deadline_s: float = REQUEST_DEADLINE_S,
    ):
        self.index = index
        self.limiter = limiter
        self.model = model
        self.retrieval = retrieval or {"top_k": 5, "rerank": True}
        self.deadline_s = deadline_s
        # give the extractive baseline the corpus idf so rare terms dominate
        self.extractive = ExtractiveDrafter(
            idf={t: index._idf(t) for t in index.df}
        )

    # -- helpers ------------------------------------------------------------
    def _pack(self, passages: list[tuple[Chunk, float]]) -> str:
        return "\n\n".join(
            f"[{c.paper} p.{c.page}] ({c.chunk_id})\n{c.text}" for c, _ in passages
        )

    async def _model_call(self, messages: list[dict], max_tokens: int,
                          deadline: float, stream: bool = True) -> GenResult:
        est = est_tokens(" ".join(m.get("content", "") for m in messages)) + max_tokens
        return await call_with_retries(
            lambda: self.model.stream(messages, max_tokens=max_tokens),
            self.limiter, est, deadline,
        )

    # -- steps --------------------------------------------------------------
    async def _classify(self, question: str, deadline: float, ans: Answer) -> list[str]:
        t = time.perf_counter()
        if self.model is None:
            q = question.lower()
            keys = []
            if any(w in q for w in ("attention", "transformer", "bleu", "head", "encoder")):
                keys.append("attention")
            if any(w in q for w in ("moshi", "speech", "audio", "dialogue", "duplex")):
                keys.append("moshi")
            if any(w in q for w in ("chunk", "h-net", "byte", "tokeniz", "hierarch")):
                keys.append("dynamic_chunking")
            ans.phases.classify_ms = (time.perf_counter() - t) * 1000
            ans.steps += 1
            return keys or ["attention", "moshi", "dynamic_chunking"]

        r = await self._model_call(
            [{"role": "system", "content": CLASSIFY_PROMPT},
             {"role": "user", "content": question}],
            max_tokens=60, deadline=deadline, stream=False,
        )
        ans.retries += max(0, r.attempts - 1)
        ans.tokens_in += r.tokens_in
        ans.tokens_out += r.tokens_out
        ans.phases.classify_ms = (time.perf_counter() - t) * 1000
        ans.steps += 1
        try:
            obj = json.loads(re.search(r"\{.*\}", r.text, re.S).group(0))
            keys = [k for k in obj.get("papers", []) if isinstance(k, str)]
            return keys or ["attention", "moshi", "dynamic_chunking"]
        except Exception:
            return ["attention", "moshi", "dynamic_chunking"]

    def _retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        return self.index.search(query, self.retrieval["top_k"], self.retrieval["rerank"])

    async def answer(self, question: str) -> Answer:
        t_start = time.perf_counter()
        deadline = t_start + self.deadline_s
        ans = Answer()

        keys = await self._classify(question, deadline, ans)

        # step 2: the retrieval tool
        t = time.perf_counter()
        passages = self._retrieve(question)
        ans.phases.retrieve_ms = (time.perf_counter() - t) * 1000
        ans.steps += 1

        # step 3: LOOP - broaden once or twice when the best hit is weak
        expansions = 0
        t = time.perf_counter()
        while (not passages or passages[0][1] < RETRIEVAL_SCORE_FLOOR) and expansions < 2:
            expansions += 1
            broadened = f"{question} {keys and ' '.join(keys) or ''}".strip()
            passages = self._retrieve(broadened)
        ans.phases.expand_ms = (time.perf_counter() - t) * 1000
        ans.steps += 1
        ans.passages = [c for c, _ in passages]

        if not passages:
            ans.text = REFUSAL
            ans.refused = True
            ans.total_ms = (time.perf_counter() - t_start) * 1000
            ans.ttft_ms = ans.total_ms
            ans.status = "ok"
            return ans

        # step 4: draft (streamed)
        t = time.perf_counter()
        if self.model is None:
            r = self.extractive.draft(question, passages)
        else:
            msgs = [
                {"role": "system", "content": DRAFT_PROMPT},
                {"role": "user", "content":
                    f"Question: {question}\n\nPassages:\n{self._pack(passages)}"},
            ]
            r = await self._model_call(msgs, max_tokens=400, deadline=deadline)
        ans.phases.draft_ms = (time.perf_counter() - t) * 1000
        ans.ttft_ms = ans.phases.classify_ms + ans.phases.retrieve_ms + \
            ans.phases.expand_ms + r.ttft_ms
        ans.tokens_in += r.tokens_in
        ans.tokens_out += r.tokens_out
        ans.retries += max(0, r.attempts - 1)
        ans.steps += 1

        if r.status != "ok":
            ans.status = r.status
            ans.text = self._degraded(r.status)
            ans.refused = True
            ans.total_ms = (time.perf_counter() - t_start) * 1000
            return ans

        text = r.text.strip()

        # step 5: BRANCH - verify citations, retry once, else refuse
        t = time.perf_counter()
        ok, cites = self._verify(text, passages)
        if not ok and self.model is not None:
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content":
                "Your answer must cite at least one supplied passage as [paper p.N] "
                "and only make claims those passages support. Rewrite it."})
            r2 = await self._model_call(msgs, max_tokens=400, deadline=deadline)
            ans.phases.retry_ms += (time.perf_counter() - t) * 1000
            ans.retries += max(0, r2.attempts - 1)
            ans.tokens_in += r2.tokens_in
            ans.tokens_out += r2.tokens_out
            ans.steps += 1
            if r2.status == "ok":
                text = r2.text.strip()
            ok, cites = self._verify(text, passages)
        ans.phases.verify_ms = (time.perf_counter() - t) * 1000 - ans.phases.retry_ms

        if not ok:
            text, cites = REFUSAL, set()
            ans.refused = True

        ans.text = text
        ans.citations = cites
        ans.total_ms = (time.perf_counter() - t_start) * 1000
        ans.status = "ok"
        return ans

    @staticmethod
    def _verify(text: str, passages) -> tuple[bool, set]:
        cites = citations_in(text)
        allowed = {(c.paper, c.page) for c, _ in passages}
        if not cites:
            return False, cites
        if not cites <= allowed:
            return False, cites
        return True, cites

    @staticmethod
    def _degraded(status: str) -> str:
        if status == "rate_limited":
            return ("I've hit the model provider's daily limit, so I can't write a "
                    "fresh answer right now. Here is what I can still do: the three "
                    "papers are indexed and searchable, so ask me for the passage "
                    "and I'll quote it directly. Full answers resume when the "
                    "provider's daily budget resets.")
        return ("The model call failed before it produced an answer. I won't guess; "
                "please retry in a moment.")
