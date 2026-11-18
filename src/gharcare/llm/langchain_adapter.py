"""Adapts any LangChain chat model (anything with .invoke(messages)) to our LLMProvider interface.
Messages are passed as (role, text) tuples, which LangChain accepts, so this module needs no
LangChain import and can be tested with a plain fake object. Temperature is fixed when the chat
model is constructed (the factory sets 0), so the per-call value is not forwarded."""
from __future__ import annotations

import re
from typing import Any, Sequence

from gharcare.errors import LLMError, LLMQuotaExhaustedError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from gharcare.llm.base import LLMMessage, LLMProvider, LLMResponse

_ROLE = {"system": "system", "user": "human", "assistant": "ai"}
_RATE = ("429", "rate limit", "ratelimit", "quota", "resource_exhausted", "resourceexhausted", "too many requests")
_QUOTA = ("per day", "perday", "requests per day", "limit: 0")
_RETRY_AFTER = (re.compile(r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.I),
                re.compile(r"retry_?delay\W+([0-9]+(?:\.[0-9]+)?)\s*s", re.I))
_DOWN = ("503", "502", "unavailable", "timeout", "timed out", "connection", "connect", "refused", "overloaded")


def parse_retry_after(text: str) -> float | None:
    """The wait the provider asked for ('Please retry in 36.3s' / retryDelay '36s'), or None."""
    for rx in _RETRY_AFTER:
        m = rx.search(text)
        if m:
            return float(m.group(1))
    return None


def map_exception(exc: Exception) -> LLMError:
    """Provider SDKs raise many different types; classify by message so retry logic can work."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(k in text for k in _QUOTA):
        return LLMQuotaExhaustedError(
            "the daily free-tier quota for this model is used up (it resets at midnight Pacific time; quotas are per "
            "model, so a different model may work now). Original error: " + str(exc)[:200])
    if any(k in text for k in _RATE):
        return LLMRateLimitError(str(exc), retry_after=parse_retry_after(str(exc)))
    if any(k in text for k in _DOWN):
        return LLMUnavailableError(str(exc))
    return LLMError(str(exc))


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):   # newer chat models return a list of content blocks
        return "".join(part if isinstance(part, str) else str(part.get("text", "")) for part in content
                       if isinstance(part, (str, dict)))
    return str(content or "")


class LangChainChatProvider(LLMProvider):
    def __init__(self, chat_model: Any, *, name: str = "langchain", model: str = "unknown") -> None:
        self._chat, self.name, self._model = chat_model, name, model

    def generate(self, messages: Sequence[LLMMessage], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> LLMResponse:
        try:
            result = self._chat.invoke([(_ROLE[m.role], m.content) for m in messages])
        except Exception as exc:  # SDK boundary: translate into our retryable / non-retryable errors
            raise map_exception(exc) from exc
        text = _flatten(getattr(result, "content", result)).strip()
        if not text:
            raise LLMResponseError("the model returned an empty response (possibly blocked)")
        usage = getattr(result, "usage_metadata", None) or {}
        return LLMResponse(text=text, model=self._model, input_tokens=usage.get("input_tokens"),
                           output_tokens=usage.get("output_tokens"))
