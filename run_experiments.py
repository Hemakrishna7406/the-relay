#!/usr/bin/env python3
"""Run every experiment that does not need a Groq key, and write results.csv.

The stand-in is driven in-process through an ASGI transport, so no background
server is required. The same FastAPI app runs over HTTP with:
    uvicorn relay.standin:app --port 8100
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

from relay.config import RETRIEVAL_SETTINGS
from relay.corpus import build_index
from relay.harness import (
    QUESTIONS, Scenario, eval_retrieval, run_scenario, summarise, write_csv,
)
from relay.limits import RateLimiter
from relay.standin import app as standin_app

OUT = Path(__file__).resolve().parent / "results"


def index_for(setting: dict):
    return build_index(setting["chunk_chars"], setting["overlap"])


async def main() -> int:
    random.seed(7)
    OUT.mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    transport = httpx.ASGITransport(app=standin_app)

    # ---- Table 3: retrieval quality vs cost (deterministic baseline) -------
    print("== retrieval settings ==")
    retrieval_rows = []
    for s in RETRIEVAL_SETTINGS:
        idx = index_for(s)
        row = await eval_retrieval(idx, s, RateLimiter())
        retrieval_rows.append(row)
        print(f"  {row['setting']:22} correct={row['correct_cited_pct']:5}% "
              f"wrong={row['wrong_pct']:5}% refused_ok={row['refused_correctly_pct']:5}% "
              f"tok/ans={row['tokens_per_answer']}")

    # ---- Table 2: load shapes against the stand-in -------------------------
    print("\n== load scenarios (stand-in) ==")
    chosen = next(s for s in RETRIEVAL_SETTINGS if s["name"].startswith("S2"))
    idx = index_for(chosen)
    retrieval = chosen

    # Capacity is ~rpm/2.5 requests per minute (2-3 model calls per request).
    scenarios = [
        Scenario("steady", n=18, concurrency=3, offered_rate_per_min=24,
                 limiter={"rpm": 120, "tpm": 200000}, source="stand-in"),
        Scenario("burst", n=24, concurrency=6, offered_rate_per_min=0,
                 limiter={"rpm": 120, "tpm": 200000}, source="stand-in"),
        Scenario("sustained_overload", n=45, concurrency=10, offered_rate_per_min=120,
                 limiter={"rpm": 60, "tpm": 120000}, source="stand-in"),
        Scenario("rpm_halved", n=24, concurrency=6, offered_rate_per_min=60,
                 limiter={"rpm": 30, "tpm": 60000}, source="stand-in"),
        Scenario("daily_exhausted", n=16, concurrency=4, offered_rate_per_min=60,
                 limiter={"rpd": 10}, source="stand-in"),
        Scenario("tpm_binding", n=12, concurrency=4, offered_rate_per_min=24,
                 limiter={"rpm": 120, "tpm": 6000}, source="stand-in"),
    ]

    all_rows: list[dict] = []
    summaries = []
    for scn in scenarios:
        rows = await run_scenario(scn, idx, retrieval, retrieval["name"], transport=transport)
        all_rows.extend(rows)
        s = summarise(rows)
        summaries.append(s)
        print(f"  {scn.name:20} n={s['requests']:3} ok={s['succeeded_pct']:5}% "
              f"rl={s['rate_limited_pct']:5}% to={s['timeout_pct']:4}% "
              f"ttft p50/p95={s['ttft_p50']}/{s['ttft_p95']}ms "
              f"peakq={s['peak_queue']}")
        (OUT / "raw" / f"{scn.name}.json").write_text(json.dumps(rows, indent=1))

    write_csv(all_rows, OUT / "results.csv")
    (OUT / "summary.json").write_text(json.dumps(
        {"retrieval": retrieval_rows, "scenarios": summaries}, indent=2))

    print(f"\nwrote {OUT/'results.csv'} ({len(all_rows)} rows)")
    print(f"wrote {OUT/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
