"""Counts calls and time spent in an LLMProvider (used by evaluations to report cost per question)."""
from __future__ import annotations

import time
from typing import Callable, Sequence

from gharcare.llm.base import LLMMessage, LLMProvider, LLMResponse


class MeteredLLMProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, on_call: Callable[[float, bool], None] | None = None) -> None:
        self._inner, self.name, self._on_call = inner, f"metered({inner.name})", on_call
        self.calls, self.seconds = 0, 0.0

    def generate(self, messages: Sequence[LLMMessage], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> LLMResponse:
        started = time.perf_counter()
        self.calls += 1                      # failed calls count too: they cost quota
        ok = False
        try:
            result = self._inner.generate(messages, temperature=temperature, max_tokens=max_tokens)
            ok = True
            return result
        finally:
            elapsed = time.perf_counter() - started
            self.seconds += elapsed
            if self._on_call:
                self._on_call(elapsed, ok)
