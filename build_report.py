"""Build the 2-page write-up (PDF) from results/summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
S = json.loads((ROOT / "results" / "summary.json").read_text())

_SS = getSampleStyleSheet()
NAVY = colors.HexColor("#1F3A5F")
BAND = colors.HexColor("#F1F4F8")
GREY = colors.HexColor("#5A6675")

TITLE = ParagraphStyle("t", parent=_SS["Title"], fontName="Helvetica-Bold",
                       fontSize=14, leading=16, alignment=TA_LEFT, spaceAfter=1)
SUB = ParagraphStyle("s", parent=_SS["Normal"], fontName="Helvetica", fontSize=8,
                     leading=10, textColor=GREY, spaceAfter=4)
H = ParagraphStyle("h", parent=_SS["Heading2"], fontName="Helvetica-Bold", fontSize=9.4,
                   leading=11.2, textColor=NAVY, spaceBefore=5.5, spaceAfter=2, keepWithNext=1)
B = ParagraphStyle("b", parent=_SS["Normal"], fontName="Helvetica", fontSize=7.6,
                   leading=9.6, spaceAfter=3)
CELL = ParagraphStyle("c", parent=B, fontSize=6.4, leading=7.8, spaceAfter=0)
HEAD = ParagraphStyle("ch", parent=CELL, fontName="Helvetica-Bold",
                      fontSize=5.9, leading=6.9, textColor=colors.white)


def _hf(canvas, doc):
    canvas.saveState()
    w, h = doc.pagesize
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(doc.leftMargin, h - 0.45 * inch, "The Relay - assignment submission")
    canvas.drawRightString(w - doc.rightMargin, h - 0.45 * inch, f"page {canvas.getPageNumber()}")
    canvas.restoreState()


def tbl(rows, fracs, width, fs=7.0):
    t = Table(rows, colWidths=[f * width for f in fracs], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), BAND),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1A1A1A")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9AA7B8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.9),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def row(cells):
    return [Paragraph(c, HEAD) for c in cells]


def cells(vals):
    return [Paragraph(v, CELL) for v in vals]


def build(path: str) -> str:
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=0.55 * inch,
                            rightMargin=0.55 * inch, topMargin=0.5 * inch,
                            bottomMargin=0.5 * inch,
                            title="The Relay - assignment submission", author="Hema Krishna")
    W = doc.width
    sc = {s["scenario"]: s for s in S["scenarios"]}
    ret = S["retrieval"]
    st = []

    st.append(Paragraph("The Relay - what I built, measured, and chose", TITLE))
    st.append(Paragraph(
        "Hema Krishna | 4-day brief | pipeline in Python | three arXiv papers at the exact "
        "requested versions | all numbers below are recomputable from results.csv "
        "(157 rows: 139 stand-in + 18 real)", SUB))

    # 1 -------------------------------------------------------------------
    st.append(Paragraph("1. What I built and why", H))
    st.append(Paragraph(
        "A five-step pipeline: <b>classify</b> which paper(s) and terms matter, <b>retrieve</b> "
        "BM25 passages (the tool call), <b>expand</b> the query in a loop when the best hit is "
        "weak, <b>draft</b> a cited answer (streamed), and <b>verify</b> that every citation "
        "resolves to a passage actually retrieved - otherwise retry once, then refuse. Refusal "
        "is a first-class outcome, not an error: the brief says a wrong answer is worse than a "
        "slow one, so the verifier may only downgrade, never invent.", B))
    st.append(Paragraph(
        "<b>The decision that shaped everything:</b> I made the retrieval unit the <i>passage</i>, "
        "not the paper, and made the verifier reject any citation outside the retrieved set. That "
        "single choice turns \"is this answer grounded?\" from a model judgement into a set "
        "membership test I can compute, log per request, and grade offline (it checks that a "
        "citation names a page actually retrieved - not that the page entails the claim). It is also what makes "
        "the refusal behaviour measurable rather than aspirational.", B))

    # 2 -------------------------------------------------------------------
    st.append(Paragraph("2. The real service and my stand-in", H))
    st.append(Paragraph(
        "The account's catalogue did not include <font face='Courier'>llama-3.1-8b-instant</font>, so "
        "all real calls used <font face='Courier'>openai/gpt-oss-20b</font> (discovered with a "
        "<font face='Courier'>/models</font> call). I found the limits by firing 40 minimal "
        "requests back to back (4-token replies) and reading the responses: <b>exactly 30 "
        "succeeded, then 10 consecutive HTTP 429s, each with "
        "<font face='Courier'>Retry-After: 2</font></b> - so RPM=30 is measured, not assumed. "
        "The probe ran at an attempted 115 req/min over 20.8 s. Tiny-call TTFT was p50 458 ms "
        "(min 233, max 1,802).", B))
    t1 = [row(["Limit", "Value", "How I found it", "Conf."])]
    for lim, val, how, conf in [
        ("Requests / minute", "30", "30 succeeded, then 10x 429 in one 60 s window", "high"),
        ("Tokens / minute", "6000 (docs)", "documented; not isolated from RPM", "low"),
        ("Requests / day", "1000 (docs)", "documented; not reached", "low"),
        ("Tokens / day", "500000 (docs)", "documented; not reached", "low"),
        ("Requests in flight", "not separable", "RPM 30 binds before any concurrency cap", "low"),
        ("On limit hit", "429 + Retry-After: 2", "observed on all 10 rejections", "high"),
        ("Time to recover", "<= 60 s", "Retry-After said 2 s, yet 10 consecutive 429s over 20.8 s; window is the minute", "medium"),
    ]:
        t1.append(cells([lim, val, how, conf]))
    st.append(tbl(t1, [0.24, 0.16, 0.48, 0.12], W))
    st.append(Spacer(1, 4))
    st.append(Paragraph(
        "<b>The stand-in</b> (<font face='Courier'>relay/standin.py</font>) is an OpenAI-shaped "
        "streaming service with five behaviours: lognormal TTFT, injected 503s (3%), a slow tail "
        "(2% of calls x4), latency that grows with in-flight requests (+6%/waiter, capped 3x), and "
        "chunk-by-chunk SSE. It enforces its own RPM/TPM and returns 429 + Retry-After. "
        "<b>Calibration:</b> I first set p50 to the measured pipeline TTFT (1,537 ms) and the "
        "stand-in ran 2.7x slow - because pipeline TTFT spans two model calls, while the stand-in "
        "models one. Re-set to 770 ms per call, the two-call pipeline reproduces the real 1.54 s "
        "(stand-in steady: 2.03 s p50). <b>Where it is least like the real thing:</b> real TTFT "
        "depends on prompt and output length and real failures cluster; mine is prompt-independent "
        "and memoryless, so my tail is too thin.", B))

    # 3 -------------------------------------------------------------------
    st.append(Paragraph("3. Results", H))
    st.append(Paragraph(
        "Time is ms. Percentiles are <b>p50 / p95 / p99 over the succeeded requests</b> in each "
        "scenario; the request count is the <i>Sent</i> column (18-45 per scenario, so treat p99 "
        "as the observed maximum, not a stable tail estimate). Offered rate is requests/minute.", B))
    t2a = [row(["Scenario", "Real/stand-in", "Requests sent", "Duration (s)",
                "Offered rate (/min)", "First token p50/p95/p99 (ms)",
                "Total time p50/p95/p99 (ms)"])]
    for name in ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted", "tpm_binding"]:
        s = sc[name]
        t2a.append(cells([
            name, s["source"], str(s["requests"]), str(s["duration_s"]), str(s["offered_rate"]),
            f"{s['ttft_p50']} / {s['ttft_p95']} / {s['ttft_p99']}",
            f"{s['total_p50']} / {s['total_p95']} / {s['total_p99']}",
        ]))
    t2a.append(cells([
        "real_sample", "real", "18", "385.0", "2.8",
        "1537.4 / 3588.6 / 3743.9", "2108.7 / 4555.7 / 5725.1",
    ]))
    st.append(tbl(t2a, [0.15, 0.10, 0.09, 0.08, 0.09, 0.245, 0.245], W))

    t2b = [row(["Scenario", "Succeeded %", "Rate-limited %", "Timed out or given up %",
                "Retries per request", "Peak queue length", "Tokens per answer",
                "Cost per 1,000"])]
    for name in ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted", "tpm_binding"]:
        s = sc[name]
        t2b.append(cells([
            name, f"{s['succeeded_pct']}", f"{s['rate_limited_pct']}", f"{s['timeout_pct']}",
            str(s["retries_per_request"]), str(s["peak_queue"]),
            str(s["tokens_per_answer"]), f"${s['cost_per_1000']}",
        ]))
    t2b.append(cells([
        "real_sample", "100.0", "0.0", "0.0", "0.06", "n/a", "3130.8", "$0.17",
    ]))
    st.append(Spacer(1, 3))
    st.append(tbl(t2b, [0.17, 0.10, 0.13, 0.12, 0.13, 0.09, 0.13, 0.13], W))
    st.append(Spacer(1, 4))
    st.append(Paragraph(
        "Cost uses an assumed list price of $0.05/M input and $0.08/M output tokens (the free "
        "tier itself is $0), so treat it as a relative comparison, not a bill.", B))
    st.append(Paragraph(
        "<b>Answer quality (real model, 18 questions I wrote - 12 answerable, 6 the papers do "
        "not answer): 33.3% correct-and-cited, 16.7% wrong, 33.3% correctly refused, 16.7% "
        "wrongly refused. All six unanswerable questions were correctly refused</b> - the "
        "behaviour the brief cares about most. Table 3 below uses a deterministic extractive "
        "drafter instead, so chunk-size differences are not swamped by model variance. Grading is a "
        "keyword key I wrote per question, not an LLM judge.", B))
    st.append(Paragraph(
        "<b>A number I could not have guessed without running it:</b> the real model spends "
        "<b>3,131 tokens per answer</b> (mean; max 4,113) - 2x the 1,570 my stand-in implied. At "
        "6,000 TPM that is a real capacity of <b>~1.9 requests/minute against an RPM of 30</b>, a "
        "16x gap. The token budget, not the request budget, is the limit that bites; the "
        "tpm_binding row (75.0% lost to deadline) shows what that looks like under load.", B))

    # 4 -------------------------------------------------------------------
    st.append(Paragraph("4. Where the time goes", H))
    st.append(Paragraph(
        "Slowest real request: <font face='Courier'>real-0017</font> (q18), total 6,017.5 ms with "
        "the first token at 1,909.9 ms. So <b>31.7% is waiting for the first token</b> (the "
        "classify call plus the draft's first chunk) and <b>68.3% - 4,108 ms - is streaming the "
        "rest after it</b>. That is the opposite shape to the stand-in's worst case "
        "(<font face='Courier'>burst-0010</font>: 5,657 ms total, first token at 5,657 ms - 100% "
        "before; classify 4,510 ms, draft 1,145 ms, retrieval 1.5 ms), and the difference is real: "
        "on the real service a long answer is dominated by "
        "generation, while my stand-in's worst case is dominated by its load-dependent pre-token "
        "sleep. I know the split because TTFT is stamped at the first SSE delta and total at the "
        "final one, and the per-step timers in the pipeline sum to the reported total.", B))

    # 5 -------------------------------------------------------------------
    st.append(Paragraph("5. When a limit is hit", H))
    st.append(Paragraph(
        "<b>Per-minute (RPM/TPM):</b> requests <i>wait</i>, because the window refills. They queue "
        "against the limiter; if the window cannot clear before the 25 s deadline the request is "
        "given up as a timeout (37.8% in sustained_overload, 41.7% in rpm_halved). Nobody is "
        "silently dropped and nobody gets a weaker answer - they get a late one or an honest "
        "failure. <b>Per-day (RPD/TPD):</b> retrying cannot help, so requests are <i>dropped fast</i> "
        "with a degraded but truthful reply (62.5% in daily_exhausted, in 15 s rather than 60). "
        "The asymmetry is deliberate: waiting on a per-minute window is worth 20 s of a user's "
        "patience; waiting on tomorrow is not.", B))

    # --- two extra headings after 5 ---------------------------------------
    st.append(Paragraph("Retrieval: quality versus cost", H))
    t3 = [row(["Setting (piece size / pieces / re-rank)", "Questions asked",
               "Correct and cited %", "Wrong %", "Correctly refused %", "Tokens per answer",
               "First token p50/p95 (ms)", "Total time p50/p95 (ms)"])]
    labels = {r["setting"]: r["setting"] for r in ret}
    from relay.config import RETRIEVAL_SETTINGS as _RS
    for rs in _RS:
        labels[rs["name"]] = (f"{rs['chunk_chars']} / {rs['top_k']} / "
                              f"{'yes' if rs['rerank'] else 'no'}")
    for r in ret:
        t3.append(cells([
            labels.get(r["setting"], r["setting"]), str(r["questions"]),
            f"{r['correct_cited_pct']}", f"{r['wrong_pct']}",
            f"{r['refused_correctly_pct']}", str(r["tokens_per_answer"]),
            f"{r['ttft_p50']} / {r['ttft_p95']}", f"{r['total_p50']} / {r['total_p95']}",
        ]))
    st.append(tbl(t3, [0.22, 0.07, 0.12, 0.08, 0.11, 0.09, 0.16, 0.15], W))
    st.append(Spacer(1, 4))
    st.append(Paragraph(
        "<b>I chose S2 (1,200-char pieces, 5 fetched, no re-rank):</b> it ties the best quality "
        "(38.9%) at 1,377 tokens, against 3,305 tokens for S4 which was <i>worse</i> (33.3%) - more "
        "context bought noise, not accuracy. Re-ranking (S3) gave identical quality and +31% TTFT, "
        "so I dropped it. <b>When the limit is close</b> I drop to S1 (465 tokens, 33.3%) and cut "
        "top_k to 3; quality falls 5.6 points and tokens fall 66%, which is the trade I would make "
        "before queueing users.", B))

    st.append(Paragraph("How this library will mislead a naive retriever", H))
    st.append(ListFlowable([
        ListItem(Paragraph(
            "<b>Lexical overlap across papers.</b> \"Which paper applies dynamic chunking to speech "
            "dialogue?\" retrieves dynamic_chunking because both terms appear there; no paper does "
            "this. Correct behaviour is refusal - my q17 tests exactly this.", B)),
        ListItem(Paragraph(
            "<b>Dense number soup.</b> \"What BLEU score for WMT 2014 English-to-French?\" "
            "returns the English-to-German results page (28.4); the surface terms match that page "
            "better and 41.8 sits in a neighbouring chunk. Verified: top-1 = attention p.8.", B)),
        ListItem(Paragraph(
            "<b>Prose that merely mentions a term.</b> The Moshi paper contains dialogue examples "
            "mentioning France; \"What is the capital of France?\" matches on 'france' and returns "
            "a story snippet. Without an anchor-term rule this scores as an answer.", B)),
    ], bulletType="1", leftIndent=12))

    # 6 -------------------------------------------------------------------
    st.append(Paragraph("6. Retries", H))
    st.append(Paragraph(
        "Policy: retry only transient failures (429 with Retry-After, timeout, 5xx), at most 2 "
        "retries, exponential backoff 0.35 s base with jitter, capped at 4 s, and never past the "
        "request deadline. A daily-budget exhaustion is never retried. <b>Where retrying made it "
        "worse:</b> under sustained_overload a 429 triggers a backoff and a retry, but the retry "
        "re-enters the same per-minute window behind the queue, so it burns a second token "
        "reservation and a second slot while the user waits - retries/request 0.07 and the p99 "
        "total (1,803 ms) is 2.6x the p50. Retrying into a saturated window converts a fast, "
        "honest 429 into a slow failure.", B))

    # 7 -------------------------------------------------------------------
    st.append(Paragraph("7. What the person sees", H))
    st.append(Paragraph(
        "At 3pm the daily budget is gone. The reply, in full: <i>\"I've hit the model provider's "
        "daily limit, so I can't write a fresh answer right now. Here is what I can still do: the "
        "three papers are indexed and searchable, so ask me for the passage and I'll quote it "
        "directly. Full answers resume when the provider's daily budget resets.\"</i> It arrives in "
        "under a second, states the cause, offers the degraded-but-true capability (retrieval "
        "still works without the model), and does not pretend a retry will help.", B))

    # 8 -------------------------------------------------------------------
    st.append(Paragraph("8. What I'm unsure about, what I left out, what I learned", H))
    st.append(Paragraph(
        "<b>Least certain assumption:</b> that the stand-in's latency profile is representative. If "
        "real Groq TTFT is tighter but its 429s are burstier, my queue sizing is wrong in both "
        "directions - I would under-provision in-flight slots and over-estimate how long a "
        "per-minute wait is tolerable. <b>Two things I deliberately did not build:</b> semantic "
        "(embedding) retrieval, because BM25 let me attribute quality changes to chunk size alone; "
        "and a cross-request cache, which would have flattered the numbers without testing the "
        "limit behaviour. <b>New to me:</b> that rate limits are a token budget wearing a request "
        "budget's clothes - the 30 RPM figure is meaningless once an answer costs 3,131 tokens, "
        "and capacity planning has to start from tokens per answer. The measured capacity was 1.9 "
        "requests/minute, 16x below the request limit. <b>What breaks first at scale:</b> the limiter "
        "is in-process, so N workers would enforce N x the budget - it needs a shared store to be "
        "safe beyond one process.", B))

    # Vaani ---------------------------------------------------------------
    st.append(Paragraph("Your feedback on Vaani", H))
    st.append(Paragraph(
        "<b>Stood out:</b> the framing is unusually concrete for a voice-AI landing page - "
        "\"zero dead air\", turn-taking and the ~450 ms p90 latency claim are the three things that "
        "actually decide whether a voice agent feels human, and the Teacher AI flow (describe the "
        "agent in a sentence, review the prompt diff before it goes live) is a genuinely good idea "
        "because prompt drift is the real failure mode. <b>Confused me:</b> several headline stats "
        "render as placeholders (\"0M+\", \"0.0%\", \"&lt;0ms\") and the case-study numbers show "
        "\"0\" - on a page whose whole argument is measurement, numbers that do not render read as "
        "unfinished. <b>One change:</b> publish a real latency distribution (p50/p90/p99 on a named "
        "telephony path) instead of a single p90 figure; a p90 alone hides the tail that decides "
        "whether callers hang up.", B))

    # Assistant ------------------------------------------------------------
    st.append(Paragraph("How I worked with the assistant", H))
    st.append(Paragraph(
        "The full conversation, unedited, is in "
        "<font face='Courier'>conversation_history.md</font> in this bundle. <b>The three prompts "
        "that mattered most:</b> (1) <i>\"build the pipeline with a retrieval tool and a verifier "
        "that can only downgrade an answer\"</i> - this fixed the grounding design, because it made "
        "\"is this grounded?\" a set-membership test instead of a model opinion; (2) <i>\"run at "
        "least five load shapes and put every number in results.csv\"</i> - this forced the "
        "measurement discipline and is why the tables are recomputable; (3) <i>\"don't invent the "
        "Groq limits - mark them not measured\"</i> - this kept Table 1 honest when there was no key "
        "yet.", B))
    st.append(Paragraph(
        "<b>One thing it got wrong that I caught:</b> the first load run failed 70-100% of "
        "everything. It had offered 30 requests/minute while each request makes 2-3 model calls "
        "against a 30 RPM budget. I caught it because peak queue length stayed at 1 while failures "
        "were near 100% - a queue that never builds but everything failing means the limiter, not "
        "the load, is the problem. <b>One decision where I overruled it:</b> when the 6,000 TPM "
        "limit made runs fail, it proposed raising the limit until they passed. I refused and kept "
        "one scenario at the real 6,000 TPM, which is where the 75% figure came from - the most "
        "useful number in this submission.", B))
    st.append(Paragraph(
        "<b>What is distinctly mine:</b> the measurements and the choices they forced - dropping "
        "re-ranking after it cost 31% TTFT for zero quality gain; choosing S2 over S4 after more "
        "context made answers <i>worse</i>; keeping the per-day and per-minute paths asymmetric; "
        "recalibrating the stand-in to 770 ms per call after my first attempt ran 2.7x slow; and "
        "leaving Table 1 low-confidence where I had not measured. The code was written with the "
        "assistant; those judgement calls, and the ability to defend them live, are the part that "
        "is mine.", B))

    doc.build(st, onFirstPage=_hf, onLaterPages=_hf)
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "The_Relay_Report.pdf")
    print("Wrote", build(out))
