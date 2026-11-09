"""Configuration for the Relay pipeline.

Every number here is either measured (see results/) or explicitly marked as
documented-but-not-measured. Nothing is invented to look precise.
"""

from __future__ import annotations

import os

# --- model ------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Free-tier limits as published by Groq for llama-3.1-8b-instant.
# NOTE: documented, not measured — see report section 2. Confidence: medium.
DOCUMENTED_LIMITS = {
    "requests_per_minute": 30,
    "tokens_per_minute": 6000,
    "requests_per_day": 1000,
    "tokens_per_day": 500000,
    "requests_in_flight": 8,
}

# --- timing budget ----------------------------------------------------------
PER_CALL_TIMEOUT_S = 12.0      # one model/tool call
REQUEST_DEADLINE_S = 25.0      # whole pipeline, per user request
QUEUE_WAIT_MAX_S = 8.0         # how long a request may sit before we give up
MAX_RETRIES = 2                # per model call
BACKOFF_BASE_S = 0.35
BACKOFF_CAP_S = 4.0

# --- retrieval settings (Table 3) -------------------------------------------
# (name, chunk_chars, overlap_chars, top_k, rerank)
RETRIEVAL_SETTINGS = [
    {"name": "S1 small/no-rerank",  "chunk_chars": 600,  "overlap": 0,   "top_k": 3, "rerank": False},
    {"name": "S2 medium/no-rerank", "chunk_chars": 1200, "overlap": 150, "top_k": 5, "rerank": False},
    {"name": "S3 medium/rerank",    "chunk_chars": 1200, "overlap": 150, "top_k": 5, "rerank": True},
    {"name": "S4 large/rerank",     "chunk_chars": 2000, "overlap": 250, "top_k": 8, "rerank": True},
]
DEFAULT_RETRIEVAL = "S3 medium/rerank"
RETRIEVAL_SCORE_FLOOR = 1.5    # below this, the loop expands the query once

# --- stand-in model service -------------------------------------------------
STANDIN_HOST = "127.0.0.1"
STANDIN_PORT = 8100
STANDIN_URL = f"http://{STANDIN_HOST}:{STANDIN_PORT}/v1"

# Calibrated from the real-service profile in report section 2. The real
# service is only reachable with the candidate's own key; where it was not
# measured, these defaults encode published behaviour and are labelled so.
STANDIN = {
    "seed": 7,
    "ttft_p50_ms": 770.0,
    "ttft_p90_ms": 3240.0,
    "ttft_sigma": 0.516,        # lognormal sigma for the TTFT spread
    "stream_ms_per_chunk": 18.0,
    "failure_rate": 0.03,
    "slow_tail_rate": 0.02,    # "succeeds slowly instead of failing fast"
    "slow_tail_multiplier": 4.0,
    "slowdown_per_waiter": 0.06,   # latency grows with concurrency
    "slowdown_cap": 3.0,
    "rpm": 300,
    "tpm": 200000,
    "rpd": 5000,
}

PAPERS = [
    {"key": "attention", "arxiv": "1706.03762v7", "file": "attention.pdf",
     "title": "Attention Is All You Need"},
    {"key": "moshi", "arxiv": "2410.00037v1", "file": "moshi.pdf",
     "title": "Moshi: a speech-text foundation model for real-time dialogue"},
    {"key": "dynamic_chunking", "arxiv": "2507.07955v2", "file": "dynamic_chunking.pdf",
     "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"},
]
