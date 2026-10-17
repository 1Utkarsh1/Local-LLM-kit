"""
Retry policy for local LLM servers that are slow to become usable.

The most common local-AI failure is not a rejected request — it is a request
sent too early. A desktop app has not finished binding its port, ``ollama
serve`` is still starting, a model is still being loaded (llama.cpp answers
``503 {"error": {"message": "Loading model"}}`` while it loads), or a server is
briefly overloaded.

This module holds the shared policy for those cases. It is deliberately small
and dependency-free, and it mirrors the defaults the official OpenAI Python
client uses (two retries, 0.5s initial delay, 8s ceiling, honours
``Retry-After``), because that is the behaviour people already expect from an
OpenAI-shaped client.

Only transient failures are retried:

* the connection could not be established or was reset,
* the server answered ``429`` or ``5xx``,
* and, for streaming, nothing had been emitted yet — retrying after a partial
  stream would duplicate output.

Client errors such as ``400`` are never retried.
"""

import random
import time
import urllib.error
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["RetryPolicy", "DEFAULT_RETRY", "is_retryable_status", "retry_after_seconds"]

#: Status codes worth another attempt: rate limits and server-side failures.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Upper bound applied to a server-supplied ``Retry-After``. A server asking a
#: local client to sleep for minutes is more likely misconfigured than right.
MAX_RETRY_AFTER = 120.0


@dataclass(frozen=True)
class RetryPolicy:
    """How many times, and how long, to wait before giving up.

    Args:
        max_retries: Additional attempts after the first one. ``0`` disables
            retrying entirely and restores single-attempt behaviour.
        initial_delay: Seconds before the second attempt.
        max_delay: Ceiling for the computed backoff.
        jitter: Randomise the delay within ``[delay/2, delay]`` so several
            clients starting together do not retry in lockstep.
        retry_on_status: Status codes to retry. Defaults to
            :data:`RETRYABLE_STATUS`.
    """

    max_retries: int = 2
    initial_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True
    retry_on_status: Tuple[int, ...] = tuple(sorted(RETRYABLE_STATUS))

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError("max_retries must be an integer.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        for name in ("initial_delay", "max_delay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("%s must be a number." % name)
            if value < 0:
                raise ValueError("%s must be non-negative." % name)
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay.")

    @property
    def enabled(self) -> bool:
        return self.max_retries > 0

    def should_retry_status(self, status: Optional[int]) -> bool:
        return status is not None and status in self.retry_on_status

    def delay_for(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Seconds to wait before retry number *attempt* (1-based).

        A server-supplied ``Retry-After`` wins over the computed backoff,
        clamped to :data:`MAX_RETRY_AFTER`.
        """
        if retry_after is not None and retry_after >= 0:
            return min(float(retry_after), MAX_RETRY_AFTER)
        backoff = min(self.initial_delay * (2 ** max(0, attempt - 1)), self.max_delay)
        if self.jitter and backoff > 0:
            return backoff * (0.5 + random.random() * 0.5)
        return backoff

    def sleep(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Sleep as computed by :meth:`delay_for`; returns the delay used."""
        delay = self.delay_for(attempt, retry_after)
        if delay > 0:
            time.sleep(delay)
        return delay


#: Default policy. Two retries matches the official OpenAI Python client.
DEFAULT_RETRY = RetryPolicy()


def is_retryable_status(status: Optional[int]) -> bool:
    """True when *status* is a transient server-side condition."""
    return status is not None and status in RETRYABLE_STATUS


def retry_after_seconds(headers: object) -> Optional[float]:
    """Parse a ``Retry-After`` header into seconds, or None.

    Only the delta-seconds form is honoured; an HTTP-date is ignored because
    local servers do not use it and clock skew makes it unreliable. Accepts any
    object with a ``get`` method (``email.message.Message`` from ``urllib``).
    """
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Retry-After")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER)


def coerce_policy(value: object) -> RetryPolicy:
    """Accept a :class:`RetryPolicy`, an int retry count, or None.

    Lets backend constructors take the friendly ``max_retries=3`` form without
    giving up the full policy object.
    """
    if value is None:
        return DEFAULT_RETRY
    if isinstance(value, RetryPolicy):
        return value
    if isinstance(value, bool):
        raise TypeError("retry policy must be a RetryPolicy or an int, not a bool.")
    if isinstance(value, int):
        return RetryPolicy(max_retries=value)
    raise TypeError("retry policy must be a RetryPolicy or an int, got %r" % type(value).__name__)


def retryable_transport_error(exc: BaseException) -> bool:
    """True when *exc* is a connection-level failure worth retrying.

    A refused connection is exactly the "server not up yet" case. Timeouts are
    included because a local model's first token can legitimately take longer
    than a short client timeout.
    """
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))
