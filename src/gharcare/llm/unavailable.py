"""Stands in when no real LLM is configured: every call raises LLMUnavailableError, so the assistant degrades to
showing the cited manual passages (and deterministic answers keep working) instead of crashing."""
from __future__ import annotations

from typing import Sequence

from gharcare.errors import LLMUnavailableError
from gharcare.llm.base import LLMMessage, LLMProvider, LLMResponse


class UnavailableLLMProvider(LLMProvider):
    name = "unavailable"

    def __init__(self, reason: str = "no LLM is configured") -> None:
        self._reason = reason

    def generate(self, messages: Sequence[LLMMessage], *, temperature: float = 0.0, max_tokens: int | None = None) -> LLMResponse:
        raise LLMUnavailableError(self._reason)
