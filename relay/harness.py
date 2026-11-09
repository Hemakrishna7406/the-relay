"""Questions, grading, load generation, and results.csv / table computation."""

from __future__ import annotations

import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from relay.config import RETRIEVAL_SETTINGS
from relay.limits import RateLimiter
from relay.model import StandInClient
from relay.pipeline import REFUSAL, Pipeline

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
    "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]

# Assumed provider list price, used only for the cost column. Free tier = $0.
PRICE_IN_PER_M = 0.05
PRICE_OUT_PER_M = 0.08


@dataclass
class Question:
    qid: str
    text: str
    paper: str | None        # expected source paper, None => unanswerable
    must: list[str] = field(default_factory=list)   # all must appear (lowercase)

    @property
    def answerable(self) -> bool:
        return self.paper is not None


QUESTIONS: list[Question] = [
    # --- Attention Is All You Need -----------------------------------------
    Question("q01", "What BLEU score did the Transformer reach on WMT 2014 English-to-German?",
             "attention", ["28.4"]),
    Question("q02", "How long did the big Transformer train for, and on how many GPUs?",
             "attention", ["3.5"]),
    Question("q03", "What model dimension (d_model) does the base Transformer use?",
             "attention", ["512"]),
    Question("q04", "How many parallel attention heads does the base model use?",
             "attention", ["8", "heads"]),
    Question("q05", "What does the Transformer dispense with, relative to recurrent and convolutional models?",
             "attention", ["recurrence"]),
    Question("q06", "What BLEU score was reported for WMT 2014 English-to-French?",
             "attention", ["41.8"]),
    # --- Moshi --------------------------------------------------------------
    Question("q07", "What audio codec does Moshi use?",
             "moshi", ["mimi"]),
    Question("q08", "Is Moshi a pipeline of separate ASR and TTS components?",
             "moshi", ["speech-to-speech"]),
    Question("q09", "What does Moshi mean by full-duplex dialogue?",
             "moshi", ["duplex"]),
    # --- Dynamic Chunking ---------------------------------------------------
    Question("q10", "What network architecture does Dynamic Chunking introduce?",
             "dynamic_chunking", ["h-net"]),
    Question("q11", "Does the H-Net operate at the byte level?",
             "dynamic_chunking", ["byte"]),
    Question("q12", "What does dynamic chunking replace in the standard pipeline?",
             "dynamic_chunking", ["tokeniz"]),
    # --- unanswerable: the papers do not say --------------------------------
    Question("q13", "What is the capital of France?", None),
    Question("q14", "How much does Groq charge per million tokens?", None),
    Question("q15", "Which blood-pressure monitor is most accurate for home use?", None),
    Question("q16", "What is Moshi's monthly subscription price?", None),
    # --- cross-paper traps for a naive retriever ----------------------------
    Question("q17", "Which paper applies dynamic chunking to speech dialogue?", None),
    Question("q18", "How many attention heads does the Moshi model use?", None),
]

REFUSAL_MARKERS = ("can't answer", "cannot answer", "do not answer", "don't guess",
                   "won't guess", "not answer", "no answer")


def grade(answer_text: str, q: Question) -> str:
    low = answer_text.lower().strip()
    refused = (answer_text.strip() == REFUSAL) or any(m in low for m in REFUSAL_MARKERS)
    if refused:
        return "refused_correctly" if not q.answerable else "refused_wrongly"
    if not q.answerable:
        return "wrong"
    if all(k in low for k in q.must):
        return "correct"
    return "wrong"


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


# --------------------------------------------------------------------------- #
# load generation
# --------------------------------------------------------------------------- #


@dataclass
class Scenario:
    name: str
    n: int
    concurrency: int
    offered_rate_per_min: float      # 0 => all at once (burst)
    limiter: dict
    source: str = "stand-in"


async def run_scenario(scn: Scenario, index, retrieval: dict, retrieval_name: str,
                       mode: str = "model", transport=None) -> list[dict]:
    limiter = RateLimiter(**scn.limiter)
    model = (StandInClient(transport=transport)
             if (mode == "model" and scn.source == "stand-in") else None)
    pipe = Pipeline(index, limiter, model=model, retrieval=retrieval)

    sem = asyncio.Semaphore(scn.concurrency)
    waiting = 0
    peak_queue = 0
    rows: list[dict] = []
    lock = asyncio.Lock()

    async def one(i: int):
        nonlocal waiting, peak_queue
        q = QUESTIONS[i % len(QUESTIONS)]
        async with lock:
            waiting += 1
            peak_queue = max(peak_queue, waiting)
        t_enq = time.perf_counter()
        async with sem:
            async with lock:
                waiting -= 1
            start = datetime.now(timezone.utc).isoformat()
            ans = await pipe.answer(q.text)
            status = ans.status
            if ans.total_ms >= pipe.deadline_s * 1000:
                status = "timeout"
            rows.append({
                "request_id": f"{scn.name}-{i:04d}",
                "scenario": scn.name,
                "source": scn.source,
                "start_time": start,
                "first_token_ms": round(ans.ttft_ms, 1),
                "total_ms": round(ans.total_ms, 1),
                "status": status,
                "retries": ans.retries,
                "tokens_in": ans.tokens_in,
                "tokens_out": ans.tokens_out,
                "retrieval_setting": retrieval_name,
                "answer_grade": grade(ans.text, q),
                "phases": {
                    "classify": round(ans.phases.classify_ms, 1),
                    "retrieve": round(ans.phases.retrieve_ms, 1),
                    "expand": round(ans.phases.expand_ms, 1),
                    "draft": round(ans.phases.draft_ms, 1),
                    "verify": round(ans.phases.verify_ms, 1),
                    "retry": round(ans.phases.retry_ms, 1),
                },
                "steps": ans.steps,
            })
        return time.perf_counter() - t_enq

    t0 = time.perf_counter()
    tasks = []
    for i in range(scn.n):
        if scn.offered_rate_per_min and i:
            await asyncio.sleep(60.0 / scn.offered_rate_per_min)
        tasks.append(asyncio.create_task(one(i)))
    await asyncio.gather(*tasks)
    duration = time.perf_counter() - t0
    for r in rows:
        r["_peak_queue"] = peak_queue
        r["_duration_s"] = round(duration, 2)
        r["_offered_rate"] = scn.offered_rate_per_min or round(scn.n / duration * 60, 1)
    return rows


# --------------------------------------------------------------------------- #
# retrieval-quality evaluation (Table 3)
# --------------------------------------------------------------------------- #


async def eval_retrieval(index, setting: dict, limiter: RateLimiter) -> dict:
    pipe = Pipeline(index, limiter, model=None, retrieval=setting)
    grades = []
    ttfts, totals, toks = [], [], []
    for q in QUESTIONS:
        ans = await pipe.answer(q.text)
        grades.append(grade(ans.text, q))
        ttfts.append(ans.ttft_ms)
        totals.append(ans.total_ms)
        toks.append(ans.tokens_in + ans.tokens_out)
    n = len(grades)
    return {
        "setting": setting["name"],
        "questions": n,
        "correct_cited_pct": round(100 * grades.count("correct") / n, 1),
        "wrong_pct": round(100 * grades.count("wrong") / n, 1),
        "refused_correctly_pct": round(100 * grades.count("refused_correctly") / n, 1),
        "tokens_per_answer": round(statistics.mean(toks), 1),
        "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95),
        "total_p50": pct(totals, 50), "total_p95": pct(totals, 95),
    }


# --------------------------------------------------------------------------- #
# csv + tables
# --------------------------------------------------------------------------- #


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    n = len(rows) or 1
    ft = [r["first_token_ms"] for r in ok]
    tt = [r["total_ms"] for r in ok]
    tok = [r["tokens_in"] + r["tokens_out"] for r in ok] or [0]
    tin = [r["tokens_in"] for r in ok] or [0]
    tout = [r["tokens_out"] for r in ok] or [0]
    cost = (statistics.mean(tin) / 1e6 * PRICE_IN_PER_M +
            statistics.mean(tout) / 1e6 * PRICE_OUT_PER_M) * 1000
    return {
        "scenario": rows[0]["scenario"],
        "source": rows[0]["source"],
        "requests": len(rows),
        "duration_s": rows[0].get("_duration_s"),
        "offered_rate": rows[0].get("_offered_rate"),
        "ttft_p50": pct(ft, 50), "ttft_p95": pct(ft, 95), "ttft_p99": pct(ft, 99),
        "total_p50": pct(tt, 50), "total_p95": pct(tt, 95), "total_p99": pct(tt, 99),
        "succeeded_pct": round(100 * len(ok) / n, 1),
        "rate_limited_pct": round(100 * sum(r["status"] == "rate_limited" for r in rows) / n, 1),
        "timeout_pct": round(100 * sum(r["status"] == "timeout" for r in rows) / n, 1),
        "error_pct": round(100 * sum(r["status"] == "error" for r in rows) / n, 1),
        "retries_per_request": round(statistics.mean([r["retries"] for r in rows]), 2),
        "peak_queue": rows[0].get("_peak_queue"),
        "tokens_per_answer": round(statistics.mean(tok), 1),
        "cost_per_1000": round(cost, 2),
    }
