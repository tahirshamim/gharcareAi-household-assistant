"""Retry decorator for any LLMProvider. Free API tiers are rate-limited, so retrying with
exponential backoff is a core reliability feature, not an extra. When the provider says how long to
wait (Gemini's per-minute limit does: 'Please retry in 36s'), that wait is honoured."""
from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

from gharcare.errors import LLMRateLimitError, LLMUnavailableError
from gharcare.llm.base import LLMMessage, LLMProvider, LLMResponse


_log = logging.getLogger(__name__)


class RetryingLLMProvider(LLMProvider):
    _RETRYABLE = (LLMRateLimitError, LLMUnavailableError)

    def __init__(self, inner: LLMProvider, *, max_attempts: int = 4, base_delay: float = 1.0,
                 max_delay: float = 65.0, sleep: Callable[[float], None] = time.sleep) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._inner, self._max_attempts = inner, max_attempts
        self._base_delay, self._max_delay, self._sleep = base_delay, max_delay, sleep
        self.name = f"retry({inner.name})"

    def generate(self, messages: Sequence[LLMMessage], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> LLMResponse:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._inner.generate(messages, temperature=temperature, max_tokens=max_tokens)
            except self._RETRYABLE as exc:
                if attempt == self._max_attempts:
                    raise
                delay = self._base_delay * 2 ** (attempt - 1)
                asked = getattr(exc, "retry_after", None)
                if asked is not None:            # the provider said how long to wait: a shorter retry just fails again
                    delay = max(delay, asked + 1.0)
                delay = min(delay, self._max_delay)
                _log.warning("LLM call failed (%s); retry %d/%d in %.0fs", type(exc).__name__, attempt, self._max_attempts - 1, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
