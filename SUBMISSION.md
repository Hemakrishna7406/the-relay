# Submission — The Relay

**What to send:** this zip, plus the report if your portal wants it separately.

| File | Why it's here |
|---|---|
| `results.csv` | The required raw file. 157 rows, one per request, exactly the 12 specified columns. 139 stand-in + 18 real, distinguished by `source`. |
| `The_Relay_Report.pdf` | The write-up. 2 pages, all 12 required headings in order, the four required tables. |
| `conversation_history.md` | The real prompts and what followed, unedited, plus what is distinctly mine. |
| `code/` | The pipeline, corpus/retrieval, limiter, model clients, stand-in service, harness, and 12 offline tests (`code/tests/`). |
| `results/` | Per-scenario raw JSON (with per-phase timings), computed summaries, the limit probe. |

## How to reproduce

```bash
pip install -r requirements.txt
mkdir -p papers
curl -L https://export.arxiv.org/pdf/1706.03762v7 -o papers/attention.pdf
curl -L https://export.arxiv.org/pdf/2410.00037v1 -o papers/moshi.pdf
curl -L https://export.arxiv.org/pdf/2507.07955v2 -o papers/dynamic_chunking.pdf

pytest -q                          # 12 offline tests
python run_experiments.py          # every stand-in number (no key needed, seeded)
GROQ_API_KEY=... python run_real.py --probe
GROQ_API_KEY=... python run_real.py --start 0 --count 18 --polite
python build_report.py             # rebuilds the PDF from results/summary.json
```

Papers are not vendored (arXiv licence); the exact versions are above and in
`README.md`.

## What is real and what is stand-in

- **Real (18 requests + a 40-call probe):** RPM = 30 measured (30 succeeded, then
  10 x 429 with `Retry-After: 2`); pipeline TTFT p50 1,537 ms / p95 3,589 ms;
  total p50 2,109 ms; 3,131 tokens per answer; 6/6 unanswerable questions
  correctly refused. Model: `openai/gpt-oss-20b`.
- **Stand-in (139 requests):** all six load shapes. Calibrated to 770 ms per
  call so the two-call pipeline reproduces the measured 1.54 s.
- **Not measured:** TPM/RPD/TPD (not isolated — runs were sequential and
  polite), requests-in-flight. Marked low-confidence in Table 1 rather than
  filled with plausible numbers.

No API key is present anywhere in this bundle.
