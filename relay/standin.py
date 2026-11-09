"""Stand-in model service.

An OpenAI-shaped endpoint we control. It can be slow, fail, refuse on a limit,
become slower under load, and stream its reply a piece at a time — the five
behaviours the brief asks for. Calibration lives in ``relay.config.STANDIN``.

Run:  uvicorn relay.standin:app --port 8100
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from relay.config import STANDIN

app = FastAPI(title="Relay stand-in model")

STATE = dict(STANDIN)
random.seed(STATE.get("seed", 7))
_state = {
    "inflight": 0,
    "peak_inflight": 0,
    "req_window": collections.deque(),
    "tok_window": collections.deque(),
    "req_today": 0,
    "served": 0,
    "rate_limited": 0,
    "failed": 0,
}

_CITE = re.compile(r"\[([a-z_]+)\s+p\.(\d+)\]")


@app.post("/admin/config")
async def set_config(payload: dict):
    STATE.update({k: v for k, v in payload.items() if k in STATE})
    return {"ok": True, "state": STATE}


@app.post("/admin/reset")
async def reset():
    random.seed(STATE.get("seed", 7))
    _state.update({"inflight": 0, "peak_inflight": 0, "req_today": 0,
                   "served": 0, "rate_limited": 0, "failed": 0})
    _state["req_window"].clear()
    _state["tok_window"].clear()
    return {"ok": True}


@app.get("/stats")
async def stats():
    return {
        "served": _state["served"],
        "rate_limited": _state["rate_limited"],
        "failed": _state["failed"],
        "peak_inflight": _state["peak_inflight"],
        "requests_this_minute": len(_state["req_window"]),
        "requests_today": _state["req_today"],
        "config": STATE,
    }


def _prune(now: float) -> None:
    while _state["req_window"] and now - _state["req_window"][0] >= 60:
        _state["req_window"].popleft()
    while _state["tok_window"] and now - _state["tok_window"][0][0] >= 60:
        _state["tok_window"].popleft()


def _ttft_ms() -> float:
    """Lognormal TTFT with the configured p50 and spread."""
    import math

    p50 = STATE["ttft_p50_ms"]
    sigma = STATE["ttft_sigma"]
    mu = math.log(p50)
    return random.lognormvariate(mu, sigma)


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = " ".join(m.get("content", "") for m in messages)
    est_in = max(1, len(prompt) // 4)
    max_out = int(body.get("max_tokens", 256))

    now = time.monotonic()
    _prune(now)

    # --- our own limits -----------------------------------------------------
    tok_min = sum(t for _, t in _state["tok_window"])
    over = (
        len(_state["req_window"]) >= STATE["rpm"]
        or tok_min + est_in >= STATE["tpm"]
        or _state["req_today"] >= STATE["rpd"]
    )
    if over:
        _state["rate_limited"] += 1
        return JSONResponse(
            status_code=429,
            headers={"retry-after": "2"},
            content={"error": {"message": "rate limit reached", "type": "rate_limit_error"}},
        )

    _state["req_window"].append(now)
    _state["tok_window"].append((now, est_in + max_out))
    _state["req_today"] += 1
    _state["served"] += 1

    # --- injected failure ---------------------------------------------------
    if random.random() < STATE["failure_rate"]:
        _state["failed"] += 1
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "upstream temporarily unavailable", "type": "server_error"}},
        )

    _state["inflight"] += 1
    _state["peak_inflight"] = max(_state["peak_inflight"], _state["inflight"])

    # --- latency model ------------------------------------------------------
    slowdown = 1.0 + min(
        STATE["slowdown_cap"], STATE["slowdown_per_waiter"] * _state["inflight"]
    )
    ttft = _ttft_ms() * slowdown
    if random.random() < STATE["slow_tail_rate"]:
        ttft *= STATE["slow_tail_multiplier"]

    cite = _CITE.search(prompt)
    tag = f"[{cite.group(1)} p.{cite.group(2)}]" if cite else ""
    text = ("Based on the retrieved passage, the answer is supported by the source. "
            f"{tag}").strip()
    words = text.split()
    pieces = [" ".join(words[i:i + 4]) + " " for i in range(0, len(words), 4)]

    async def stream():
        try:
            await asyncio.sleep(ttft / 1000.0)
            for p in pieces:
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"content": p}}]}
                ) + "\n\n"
                await asyncio.sleep(STATE["stream_ms_per_chunk"] / 1000.0)
            yield "data: " + json.dumps(
                {"choices": [{"delta": {}}],
                 "usage": {"prompt_tokens": est_in, "completion_tokens": max(1, len(text) // 4)}}
            ) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            _state["inflight"] = max(0, _state["inflight"] - 1)

    return StreamingResponse(stream(), media_type="text/event-stream")
