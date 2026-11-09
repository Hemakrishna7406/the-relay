# The Relay — submission

A five-step LLM pipeline over three arXiv papers, with a measured rate-limit and
latency story, a stand-in model service, and raw results.

## What's here

| Path | What |
|---|---|
| `The_Relay_Report.pdf` | The 2-page write-up (all required headings + 4 tables) |
| `results/results.csv` | Raw file: one row per request, 139 rows |
| `results/summary.json` | Computed tables (everything in the PDF is derived from this) |
| `results/raw/*.json` | Per-scenario rows including per-phase timings |
| `conversation_history.md` | Real prompts + what followed |
| `relay/` | The pipeline, corpus, limiter, model clients, stand-in, harness |
| `run_experiments.py` | Reproduces every number without a provider key |
| `run_real.py` | Runs the real Groq path with your key |
| `build_report.py` | Rebuilds the PDF from `results/summary.json` |
| `tests/test_relay.py` | 12 offline tests: limiter window/daily/in-flight, retry policy, verifier, grader, CSV contract |

Papers are **not** vendored; they are fetched at the exact versions:

```
1706.03762v7  Attention Is All You Need
2410.00037v1  Moshi: a speech-text foundation model for real-time dialogue
2507.07955v2  Dynamic Chunking for End-to-End Hierarchical Sequence Modeling
```

## Run it

```bash
pip install -r requirements.txt
pytest -q                          # 12 offline tests, no key needed

# 1. fetch the three papers (exact versions) into papers/
mkdir -p papers
curl -L https://export.arxiv.org/pdf/1706.03762v7 -o papers/attention.pdf
curl -L https://export.arxiv.org/pdf/2410.00037v1 -o papers/moshi.pdf
curl -L https://export.arxiv.org/pdf/2507.07955v2 -o papers/dynamic_chunking.pdf

# 2. reproduce every measured number (no key needed)
python run_experiments.py

# 3. rebuild the report
python build_report.py

# 4. optional: the real service (your own free-tier key)
GROQ_API_KEY=gsk_... python run_real.py
```

## The pipeline

```
question
  → classify   which paper(s) and terms matter           (model call)
  → retrieve   BM25 over chunks                          (the tool call)
  → expand     LOOP: broaden the query if the top hit is weak
  → draft      cited answer, streamed
  → verify     BRANCH: citations must resolve to retrieved passages;
               retry once, then refuse
```

## Design decisions worth defending

- **The passage is the retrieval unit, and the verifier is a set-membership
  test.** Grounding becomes computable and loggable instead of a model opinion.
- **Per-minute and per-day limits behave differently.** Per-minute: wait and
  queue. Per-day: fail fast with a truthful degraded reply. Waiting 20s for a
  refilling window is worth it; waiting for tomorrow is not.
- **Retrieval: S2 (1,200-char pieces, 5 fetched, no re-rank).** It ties the best
  quality at 1,377 tokens; S4 spends 2.4× the tokens for *worse* answers, and
  re-ranking cost 31% TTFT for no quality gain.
- **The stand-in is a service, not a mock in the test.** It streams, fails, rate
  limits, and slows under load, and it runs over HTTP
  (`uvicorn relay.standin:app --port 8100`) or in-process via ASGI for the
  automated runs.

## What was measured on the real service

18 real requests through the full pipeline plus a 40-call limit probe, using
`openai/gpt-oss-20b` (the account's catalogue did not include
`llama-3.1-8b-instant`). Measured: **RPM 30** (30 succeeded, then 10 x 429 with
`Retry-After: 2`), pipeline TTFT p50 **1,537 ms** / p95 3,589 ms, total p50
2,109 ms, **3,131 tokens per answer**, and 6/6 unanswerable questions correctly
refused. Still documented-only: TPM/RPD/TPD (not isolated), requests-in-flight
(runs were sequential).

The stand-in was then recalibrated from those numbers (770 ms per call, so the
two-call pipeline reproduces the measured 1.54 s).

`results.csv` holds both: 139 stand-in rows and 18 real rows, distinguished by
the `source` column.
