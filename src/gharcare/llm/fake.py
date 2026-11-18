"""Deterministic provider for tests and offline development."""
from __future__ import annotations

from collections import deque
from typing import Callable, Sequence

from gharcare.errors import LLMError, LLMResponseError
from gharcare.llm.base import LLMMessage, LLMProvider, LLMResponse


class FakeLLMProvider(LLMProvider):
    """Scripted replies: pass strings (returned in order) or exceptions (raised in order).
    With a `responder` callable, replies are computed from the messages instead.
    Every call is recorded in `.calls` so tests can assert on what the app sent."""

    name = "fake"

    def __init__(self, script: Sequence[str | LLMError] = (),
                 responder: Callable[[Sequence[LLMMessage]], str] | None = None) -> None:
        self._script = deque(script)
        self._responder = responder
        self.calls: list[list[LLMMessage]] = []

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        self.calls.append(list(messages))
        if self._responder is not None:
            return LLMResponse(text=self._responder(messages), model="fake")
        if not self._script:
            raise LLMResponseError("FakeLLMProvider script is exhausted")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, model="fake")
