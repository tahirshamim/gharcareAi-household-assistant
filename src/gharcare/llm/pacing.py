"""Client-side pacing: keeps calls under a requests-per-minute limit so the provider never has to reject
them. Free tiers allow very few requests per minute (Gemini Flash-Lite: 15), and an evaluation or a
single assistant answer makes several calls in quick succession."""
from __future__ import annotations

import threading
import time
from typing import Callable, Sequence

from gharcare.llm.base import LLMMessage, LLMProvider, LLMResponse


class PacedLLMProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, *, requests_per_minute: float,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._inner, self._interval = inner, 60.0 / requests_per_minute
        self._clock, self._sleep = clock, sleep
        self._next_allowed: float | None = None
        self._lock = threading.Lock()
        self.name = inner.name

    def generate(self, messages: Sequence[LLMMessage], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> LLMResponse:
        with self._lock:      # reserve a time slot; concurrent callers queue behind each other
            now = self._clock()
            start = now if self._next_allowed is None else max(now, self._next_allowed)
            self._next_allowed = start + self._interval
        wait = start - now
        if wait > 0:
            self._sleep(wait)
        return self._inner.generate(messages, temperature=temperature, max_tokens=max_tokens)
