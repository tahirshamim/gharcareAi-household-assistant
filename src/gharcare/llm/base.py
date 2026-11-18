"""Provider-agnostic LLM interface. The rest of the app (and LangGraph nodes, later) depend on
LLMProvider only, so Gemini <-> Ollama <-> a test fake is a one-line configuration change."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self, messages: Sequence[LLMMessage], *, temperature: float = 0.0, max_tokens: int | None = None,
    ) -> LLMResponse:
        """Raise LLMRateLimitError / LLMUnavailableError for retryable failures,
        LLMResponseError for unusable answers."""
