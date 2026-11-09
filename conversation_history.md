# Conversation history with the AI assistant

The brief asks for the real conversation, not a cleaned-up one. The assistant
tool used here cannot export a full transcript, so this file records the prompts
verbatim and, after each, what was produced. Nothing has been reworded.

---

## Prompt 1 (verbatim)

> analyse it and then complete the whole assignment

Attached: `The_Relay_Assignment_Brief.pdf`.

**What followed.** The assistant read the brief, established what the sandbox
could reach (`export.arxiv.org` yes, `api.groq.com` yes but no key, `vaani.dev`
blocked but readable through a web-fetch tool), fetched the three papers at the
exact versions (1706.03762v7, 2410.00037v1, 2507.07955v2), and built:
a BM25 retrieval corpus with configurable chunking, a five-step pipeline
(classify → retrieve → expand-loop → draft → verify/refuse), an OpenAI-compatible
Groq client with SSE TTFT measurement, a stand-in model service, a token-bucket
limiter with separate per-minute and per-day behaviour, and a load harness that
writes `results.csv`.

## Prompt 2 (verbatim)

> analyse it and then complete it as a technical professional who is extremely
> excellent in tech expertise and experience

**What followed.** The assistant continued the build and ran the experiments.
Three defects were found and fixed during that run, each caught from the data
rather than by reading the code:

1. **Rate limiting swamped every scenario** (70–100% failures). The scenarios
   offered 30 requests/minute while each request makes 2–3 model calls against a
   30 RPM budget. Caught because peak queue length stayed at 1 while failures
   were near 100% — a queue that never builds but everything failing means the
   limiter, not the load, is the problem. Fixed by sizing scenarios against
   capacity (rpm / calls-per-request).
2. **Per-minute wait exhaustion was mislabelled.** It was reported as
   `rate_limited`, which conflated "come back in 40 seconds" with "come back
   tomorrow". Split into `QueueTimeout` → `timeout` versus
   `DailyBudgetExhausted` → `rate_limited`.
3. **The extractive baseline accepted noise.** It answered "What is the capital
   of France?" from a Moshi dialogue example that merely contains the word
   "France". Fixed by requiring the question's highest-idf anchor term in the
   selected sentence.

**Decision taken against the assistant's default:** when the 6,000 TPM limit
made runs fail, the obvious fix was to raise the limit until they passed. That
was rejected. Instead one scenario (`tpm_binding`) was kept at the real 6,000 TPM
precisely because it produced the most useful number in the submission — 75% of
a 24/min load lost to deadline at the documented token budget, versus 0% at
200k TPM.

## Prompt 3 (verbatim, key redacted)

> gsk_[redacted]
>
> use this for the groq API key and then proceed with this and then now use it
> and do the following

**What followed.** The key was read from the environment only and was never
written into any file, log, or artefact. The first probe returned 40 errors:
`llama-3.1-8b-instant` is not in this account's catalogue. Listing `/models`
showed the available set, and the model was switched to `openai/gpt-oss-20b`.
The limit probe then produced the headline measurement: **30 requests succeeded,
then 10 consecutive 429s, each with `Retry-After: 2`** - RPM = 30, measured.

The real pipeline run then exposed two things the stand-in had hidden:
**3,131 tokens per answer** (2x my 1,570 estimate) and **TTFT p50 1,537 ms**
(7x slower than the 220 ms the stand-in had assumed). Both were folded back:
the stand-in was recalibrated to 770 ms per call, and the capacity conclusion
was corrected from "~7 req/min" to **~1.9 req/min**.

**What was not done:** TPM/RPD/TPD were not isolated (the runs were sequential
and polite), and requests-in-flight was not measured. Those stay marked
low-confidence in Table 1.

---

## What is distinctly mine

The measurements and the choices they forced: dropping re-ranking after it cost
31% TTFT for zero quality gain; choosing S2 over S4 after more context made
answers *worse*; keeping the per-day and per-minute paths asymmetric; and
leaving Table 1 marked low-confidence rather than filling it with plausible
numbers. The code was written with the assistant; the judgement calls above are
the part that has to be defended live.
