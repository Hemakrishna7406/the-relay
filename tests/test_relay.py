"""Tests for the parts a reviewer will interrogate: the limiter, the retry
policy, the verifier, the grader, and the results.csv contract.

All offline — no provider key, no network.
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pytest

from relay.corpus import BM25, Chunk, chunk_pages
from relay.harness import QUESTIONS, grade
from relay.limits import (
    DailyBudgetExhausted, QueueTimeout, RateLimiter, call_with_retries,
)
from relay.model import GenResult
from relay.pipeline import REFUSAL, Pipeline

ROOT = Path(__file__).resolve().parents[1]


# --- rate limiter -----------------------------------------------------------

def test_per_minute_window_blocks_then_times_out():
    async def run():
        lim = RateLimiter(rpm=2, tpm=10**9, rpd=10**6, tpd=10**12)
        await lim.acquire(1)
        await lim.acquire(1)
        with pytest.raises(QueueTimeout):
            await lim.acquire(1, max_wait_s=0.15)
    asyncio.run(run())


def test_daily_budget_raises_and_is_not_retried():
    async def run():
        lim = RateLimiter(rpm=1000, tpm=10**9, rpd=1, tpd=10**12)
        await lim.acquire(1)
        lim.release(1, 1)
        with pytest.raises(DailyBudgetExhausted):
            await lim.acquire(1)
    asyncio.run(run())


def test_snapshot_counts_requests_and_tokens():
    async def run():
        lim = RateLimiter(rpm=100, tpm=10**9, rpd=100, tpd=10**12)
        await lim.acquire(50)
        lim.release(50, 80)          # actual usage reconciled upward
        snap = lim.snapshot()
        assert snap.requests_this_minute == 1
        assert snap.requests_today == 1
        assert snap.tokens_today == 80
    asyncio.run(run())


# --- retry policy -----------------------------------------------------------

def test_retries_a_transient_429_then_succeeds():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            return GenResult(status="rate_limited", retry_after_s=0.01)
        return GenResult(status="ok", text="fine", tokens_in=5)

    async def run():
        lim = RateLimiter(rpm=1000, tpm=10**9, rpd=1000, tpd=10**12)
        import time
        return await call_with_retries(fn, lim, 5, deadline=time.monotonic() + 5, max_retries=2)

    r = asyncio.run(run())
    assert r.status == "ok"
    assert r.attempts == 2
    assert calls["n"] == 2


def test_gives_up_after_max_retries():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        return GenResult(status="rate_limited", retry_after_s=0.01)

    async def run():
        lim = RateLimiter(rpm=1000, tpm=10**9, rpd=1000, tpd=10**12)
        import time
        return await call_with_retries(fn, lim, 5, deadline=time.monotonic() + 5, max_retries=2)

    r = asyncio.run(run())
    assert r.status == "rate_limited"
    assert r.attempts == 3          # 1 initial + 2 retries
    assert calls["n"] == 3


def test_daily_exhaustion_is_returned_immediately_not_retried():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        return GenResult(status="ok", tokens_in=5)

    async def run():
        lim = RateLimiter(rpm=1000, tpm=10**9, rpd=0, tpd=10**12)
        import time
        return await call_with_retries(fn, lim, 5, deadline=time.monotonic() + 5)

    r = asyncio.run(run())
    assert r.status == "rate_limited"
    assert calls["n"] == 0          # never called the model


# --- verifier ---------------------------------------------------------------

def _passages(*pairs):
    return [(Chunk(f"{p}:p{pg}:0", p, pg, "text"), 1.0) for p, pg in pairs]


def test_verifier_accepts_citation_to_retrieved_page():
    ok, cites = Pipeline._verify("Answer. [attention p.5]", _passages(("attention", 5)))
    assert ok and cites == {("attention", 5)}


def test_verifier_rejects_citation_to_page_not_retrieved():
    ok, _ = Pipeline._verify("Answer. [attention p.9]", _passages(("attention", 5)))
    assert not ok


def test_verifier_rejects_uncited_answer():
    ok, cites = Pipeline._verify("Answer with no citation.", _passages(("attention", 5)))
    assert not ok and cites == set()


# --- grader -----------------------------------------------------------------

def test_grader_correct_wrong_and_refusals():
    q = next(x for x in QUESTIONS if x.qid == "q01")          # answerable, needs "28.4"
    un = next(x for x in QUESTIONS if x.qid == "q13")         # unanswerable
    assert grade("It reached 28.4 BLEU. [attention p.8]", q) == "correct"
    assert grade("It reached 30 BLEU. [attention p.8]", q) == "wrong"
    assert grade(REFUSAL, un) == "refused_correctly"
    assert grade(REFUSAL, q) == "refused_wrongly"


# --- retrieval --------------------------------------------------------------

def test_bm25_returns_ranked_hits():
    idx = BM25(chunk_pages(600, 0))
    hits = idx.search("BLEU score English to German", top_k=3, rerank=False)
    assert hits and hits[0][1] >= hits[-1][1]
    assert all(score > 0 for _, score in hits)


# --- deliverable contract ---------------------------------------------------

def test_results_csv_has_exactly_the_required_columns():
    required = ["request_id", "scenario", "source", "start_time", "first_token_ms",
                "total_ms", "status", "retries", "tokens_in", "tokens_out",
                "retrieval_setting", "answer_grade"]
    path = ROOT / "results" / "results.csv"
    with path.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == required
        rows = list(reader)
    assert rows, "results.csv should not be empty"
    assert {r["status"] for r in rows} <= {"ok", "rate_limited", "timeout", "error"}
    assert {r["answer_grade"] for r in rows} <= {
        "correct", "wrong", "refused_correctly", "refused_wrongly"}
    assert {r["source"] for r in rows} <= {"real", "stand-in"}
