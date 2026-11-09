"""Rate limiting, in-flight control, and the retry policy.

Two different limit behaviours, because they need different answers:

* per-minute (RPM/TPM)  -> *wait*: the window refills, so queue and retry.
* per-day (RPD/TPD)     -> *stop*: the budget is gone, so serve a degraded
                           answer instead of pretending a retry will help.
"""

from __future__ import annotations

import asyncio
import collections
import random
import time
from dataclasses import dataclass

from relay.config import (
    BACKOFF_BASE_S, BACKOFF_CAP_S, DOCUMENTED_LIMITS, MAX_RETRIES,
)
from relay.model import GenResult


class DailyBudgetExhausted(Exception):
    """The day's budget is gone: retrying cannot help."""


class QueueTimeout(Exception):
    """The per-minute window did not clear before the request's deadline."""


@dataclass
class LimitSnapshot:
    requests_this_minute: int
    tokens_this_minute: int
    requests_today: int
    tokens_today: int
    in_flight: int


class RateLimiter:
    def __init__(self, rpm=None, tpm=None, rpd=None, tpd=None, max_in_flight=None):
        d = DOCUMENTED_LIMITS
        # explicit None checks: a limit of 0 is a valid budget and must not
        # silently fall back to the default
        self.rpm = rpm if rpm is not None else d["requests_per_minute"]
        self.tpm = tpm if tpm is not None else d["tokens_per_minute"]
        self.rpd = rpd if rpd is not None else d["requests_per_day"]
        self.tpd = tpd if tpd is not None else d["tokens_per_day"]
        self.max_in_flight = (max_in_flight if max_in_flight is not None
                              else d["requests_in_flight"])
        self._req_min: collections.deque[float] = collections.deque()
        self._tok_min: collections.deque[tuple[float, int]] = collections.deque()
        self._req_day = 0
        self._tok_day = 0
        self._day_start = time.monotonic()
        self._sem = asyncio.Semaphore(self.max_in_flight)
        self._lock = asyncio.Lock()
        self._in_flight = 0

    def _roll_day(self, now: float) -> None:
        if now - self._day_start >= 86400:
            self._day_start = now
            self._req_day = 0
            self._tok_day = 0

    def _prune(self, now: float) -> None:
        while self._req_min and now - self._req_min[0] >= 60:
            self._req_min.popleft()
        while self._tok_min and now - self._tok_min[0][0] >= 60:
            self._tok_min.popleft()

    async def acquire(self, est_tokens: int, max_wait_s: float = 20.0) -> None:
        deadline = time.monotonic() + max_wait_s
        while True:
            async with self._lock:
                now = time.monotonic()
                self._roll_day(now)
                self._prune(now)
                if self._req_day >= self.rpd or self._tok_day + est_tokens > self.tpd:
                    raise DailyBudgetExhausted(
                        f"daily budget exhausted (rpd {self._req_day}/{self.rpd}, "
                        f"tpd {self._tok_day}/{self.tpd})"
                    )
                tok_min = sum(t for _, t in self._tok_min)
                if len(self._req_min) < self.rpm and tok_min + est_tokens <= self.tpm:
                    self._req_min.append(now)
                    self._tok_min.append((now, est_tokens))
                    self._req_day += 1
                    self._tok_day += est_tokens
                    self._in_flight += 1
                    break
                wait = 60.0 - (now - self._req_min[0]) if self._req_min else 0.05
            if time.monotonic() + wait > deadline:
                raise QueueTimeout("per-minute window did not clear before deadline")
            await asyncio.sleep(min(max(wait, 0.02), 0.5))
        await self._sem.acquire()

    def release(self, est_tokens: int, actual_tokens: int) -> None:
        delta = actual_tokens - est_tokens
        if delta:
            self._tok_day += delta
            if self._tok_min:
                ts, t = self._tok_min[-1]
                self._tok_min[-1] = (ts, max(0, t + delta))
        self._in_flight = max(0, self._in_flight - 1)
        self._sem.release()

    def snapshot(self) -> LimitSnapshot:
        now = time.monotonic()
        self._prune(now)
        return LimitSnapshot(
            requests_this_minute=len(self._req_min),
            tokens_this_minute=sum(t for _, t in self._tok_min),
            requests_today=self._req_day,
            tokens_today=self._tok_day,
            in_flight=self._in_flight,
        )


def _backoff(attempt: int, retry_after: float | None) -> float:
    if retry_after:
        return min(retry_after, BACKOFF_CAP_S)
    base = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))
    return base * (0.5 + random.random())


async def call_with_retries(
    fn,
    limiter: RateLimiter,
    est_tokens: int,
    deadline: float,
    max_retries: int = MAX_RETRIES,
) -> GenResult:
    """Run one model call under the limiter, retrying per the policy.

    Retries only *transient* failures (429 per-minute, timeout, 5xx). A daily
    budget exhaustion is returned immediately as ``rate_limited``.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            await limiter.acquire(est_tokens)
        except DailyBudgetExhausted as exc:
            return GenResult(status="rate_limited", attempts=attempts, error=str(exc))
        except QueueTimeout as exc:
            # waited out the window but the deadline arrived first -> given up
            return GenResult(status="timeout", attempts=attempts, error=str(exc))
        r = GenResult(status="error", error="call did not run")
        try:
            r = await fn()
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
            r = GenResult(status="error", error=str(exc)[:120])
        limiter.release(est_tokens, r.tokens_in or est_tokens)
        r.attempts = attempts
        if r.status == "ok":
            return r
        if attempts > max_retries:
            return r
        if time.monotonic() >= deadline:
            r.status = "timeout"
            return r
        await asyncio.sleep(_backoff(attempts, r.retry_after_s))
