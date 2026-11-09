"""Model clients.

Three interchangeable backends behind one streaming call:

* ``GroqClient``   - the real hosted service (needs GROQ_API_KEY).
* ``StandInClient``- our own service that can be slow, fail, rate-limit,
                     slow down under load, and stream piece by piece.
* ``ExtractiveDrafter`` - no network; quotes the best retrieved passage and
                     cites it. Used to isolate retrieval quality from model
                     variance in the Table 3 comparison.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

from relay.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, STANDIN_URL


@dataclass
class GenResult:
    text: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    status: str = "ok"           # ok | rate_limited | timeout | error
    attempts: int = 1
    retry_after_s: float | None = None
    error: str | None = None


class _OpenAICompatible:
    """Shared SSE streaming implementation for OpenAI-shaped endpoints."""

    name = "base"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 12.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def stream(
        self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.2
    ) -> GenResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        res = GenResult()
        t0 = time.perf_counter()
        first: float | None = None
        chunks: list[str] = []
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=getattr(self, "_transport", None)
            ) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions",
                    json=payload, headers=headers,
                ) as resp:
                    if resp.status_code == 429:
                        res.status = "rate_limited"
                        ra = resp.headers.get("retry-after")
                        res.retry_after_s = float(ra) if ra and ra.replace(".", "").isdigit() else None
                        await resp.aread()
                        res.total_ms = (time.perf_counter() - t0) * 1000
                        res.error = "429"
                        return res
                    if resp.status_code >= 400:
                        await resp.aread()
                        res.status = "error"
                        res.error = f"http {resp.status_code}"
                        res.total_ms = (time.perf_counter() - t0) * 1000
                        return res

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta", {}).get("content")
                            if delta:
                                if first is None:
                                    first = time.perf_counter()
                                chunks.append(delta)
                        usage = obj.get("usage")
                        if usage:
                            res.tokens_in = usage.get("prompt_tokens", res.tokens_in)
                            res.tokens_out = usage.get("completion_tokens", res.tokens_out)
        except (httpx.TimeoutException, httpx.ReadTimeout):
            res.status = "timeout"
            res.error = "timeout"
        except httpx.HTTPError as exc:
            res.status = "error"
            res.error = str(exc)[:120]

        now = time.perf_counter()
        res.text = "".join(chunks)
        res.ttft_ms = ((first or now) - t0) * 1000 if chunks else (now - t0) * 1000
        res.total_ms = (now - t0) * 1000
        if not res.tokens_out:
            res.tokens_out = max(1, len(res.text) // 4)
        if not res.tokens_in:
            res.tokens_in = sum(len(m.get("content", "")) for m in messages) // 4
        return res


class GroqClient(_OpenAICompatible):
    name = "groq"

    def __init__(self, model: str = GROQ_MODEL, timeout: float = 12.0):
        super().__init__(GROQ_BASE_URL, model, GROQ_API_KEY, timeout)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class StandInClient(_OpenAICompatible):
    name = "stand-in"

    def __init__(self, model: str = "standin-1", timeout: float = 12.0, transport=None):
        super().__init__(STANDIN_URL, model, "", timeout)
        # transport lets the harness drive the FastAPI stand-in in-process
        # (no background server needed); over HTTP it stays None.
        self._transport = transport


_CITE = re.compile(r"\[([a-z_]+)\s+p\.(\d+)\]")


class ExtractiveDrafter:
    """Deterministic: answers by quoting the single best-matching sentence.

    No model call, so its quality reflects retrieval quality alone. Sentences are
    ranked by idf-weighted term overlap (rare terms matter more), numeric
    questions prefer sentences that contain a number, and it refuses when no
    sentence covers enough of the question.
    """

    name = "extractive"
    _NUMERIC_Q = ("how many", "how much", "what score", "what bleu", "what value",
                  "dimension", "how long")

    def __init__(self, idf: dict[str, float] | None = None, min_coverage: float = 0.45):
        self.idf = idf or {}
        self.min_coverage = min_coverage

    def _w(self, term: str) -> float:
        return self.idf.get(term, 1.0)

    def draft(self, question: str, passages: list) -> GenResult:
        from relay.corpus import tokens

        t0 = time.perf_counter()
        q = set(tokens(question))
        wants_number = any(w in question.lower() for w in self._NUMERIC_Q)
        best: tuple[str, object] | None = None
        best_score = -1.0

        # the most distinctive query term must appear, or the match is noise
        anchor = max(q, key=self._w, default=None)
        for ch, _score in passages:
            for sentence in re.split(r"(?<=[.!?])\s+", ch.text):
                st = set(tokens(sentence))
                matched = q & st
                if not matched:
                    continue
                coverage = len(matched) / (len(q) or 1)
                if coverage < self.min_coverage:
                    continue
                # the most distinctive term must appear, unless the sentence
                # covers most of the question anyway
                if anchor is not None and anchor not in st and coverage < 0.6:
                    continue
                if wants_number and not re.search(r"\d", sentence):
                    continue
                score = sum(self._w(t) for t in matched) * (1 + len(matched))
                if score > best_score:
                    best, best_score = (sentence.strip(), ch), score

        if best is None:
            text = ("The papers provided do not answer this question. "
                    "I can only answer from the three papers in the library.")
        else:
            sentence, ch = best
            text = f"{sentence} {ch.cite}"
        now = time.perf_counter()
        return GenResult(
            text=text, ttft_ms=(now - t0) * 1000, total_ms=(now - t0) * 1000,
            tokens_in=sum(len(c.text) for c, _ in passages) // 4,
            tokens_out=len(text) // 4,
        )


def citations_in(text: str) -> set[tuple[str, int]]:
    return {(m.group(1), int(m.group(2))) for m in _CITE.finditer(text)}
