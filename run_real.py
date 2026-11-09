#!/usr/bin/env python3
"""Real-service runs against Groq. Requires GROQ_API_KEY.

    python run_real.py --probe                 # discover limits
    python run_real.py --start 0 --count 9     # pipeline sample (appendable)

Writes results/limits_real.json and appends to results/results_real.csv.
The key is read from the environment only; it is never written anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relay.config import GROQ_API_KEY, RETRIEVAL_SETTINGS
from relay.corpus import build_index
from relay.harness import CSV_COLUMNS, QUESTIONS, grade
from relay.limits import RateLimiter
from relay.model import GroqClient
from relay.pipeline import Pipeline

OUT = Path(__file__).resolve().parent / "results"
REAL_CSV = OUT / "results_real.csv"


async def probe_limits(client: GroqClient, n: int = 40) -> dict:
    """Fire n tiny calls back to back and watch for 429s and Retry-After."""
    seen = {"ok": 0, "429": 0, "errors": 0, "retry_after": [], "ttft_ms": [],
            "probe_requests": n, "wall_s": 0.0}
    import time
    t0 = time.perf_counter()
    for _ in range(n):
        r = await client.stream(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=4, temperature=0,
        )
        if r.status == "rate_limited":
            seen["429"] += 1
            if r.retry_after_s:
                seen["retry_after"].append(r.retry_after_s)
        elif r.status == "ok":
            seen["ok"] += 1
            seen["ttft_ms"].append(round(r.ttft_ms, 1))
        else:
            seen["errors"] += 1
    seen["wall_s"] = round(time.perf_counter() - t0, 2)
    seen["observed_rate_per_min"] = round(60 * n / max(seen["wall_s"], 0.01), 1)
    return seen


def append_rows(rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    new = not REAL_CSV.exists()
    with REAL_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=len(QUESTIONS))
    ap.add_argument("--probe-n", type=int, default=40)
    ap.add_argument("--polite", action="store_true",
                    help="space requests instead of letting the limiter block")
    ap.add_argument("--sleep", type=float, default=19.0)
    args = ap.parse_args()

    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set.")
        return 2
    client = GroqClient()

    if args.probe:
        print(f"== probing limits with {args.probe_n} rapid calls ==")
        limits = await probe_limits(client, args.probe_n)
        OUT.mkdir(exist_ok=True)
        (OUT / "limits_real.json").write_text(json.dumps(limits, indent=2))
        print(json.dumps(limits, indent=2))
        return 0

    chosen = next(s for s in RETRIEVAL_SETTINGS if s["name"].startswith("S2"))
    idx = build_index(chosen["chunk_chars"], chosen["overlap"])
    limiter = (RateLimiter(rpm=1000, tpm=10_000_000, rpd=100000, tpd=10**9)
               if args.polite else RateLimiter())
    pipe = Pipeline(idx, limiter, model=client, retrieval=chosen)

    subset = QUESTIONS[args.start:args.start + args.count]
    rows = []
    for i, q in enumerate(subset, start=args.start):
        ans = await pipe.answer(q.text)
        g = grade(ans.text, q) if ans.status == "ok" else "wrong"
        rows.append({
            "request_id": f"real-{i:04d}",
            "scenario": "real_sample",
            "source": "real",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "first_token_ms": round(ans.ttft_ms, 1),
            "total_ms": round(ans.total_ms, 1),
            "status": ans.status,
            "retries": ans.retries,
            "tokens_in": ans.tokens_in,
            "tokens_out": ans.tokens_out,
            "retrieval_setting": chosen["name"],
            "answer_grade": g,
        })
        print(f"  {q.qid} {ans.status:12} ttft={ans.ttft_ms:7.1f} total={ans.total_ms:7.1f} {g}")
        if args.polite:
            await asyncio.sleep(args.sleep)
    append_rows(rows)
    print(f"appended {len(rows)} rows -> {REAL_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
