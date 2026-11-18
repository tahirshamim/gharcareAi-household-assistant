"""Builds the configured LLM provider. Config problems are reported BEFORE any library import, so
the messages are clear even when optional packages are missing."""
from __future__ import annotations

import os
import warnings
from typing import Mapping

from gharcare.config import Settings
from gharcare.errors import DomainValidationError, LLMError
from gharcare.llm.base import LLMProvider
from gharcare.llm.langchain_adapter import LangChainChatProvider
from gharcare.llm.pacing import PacedLLMProvider
from gharcare.llm.retry import RetryingLLMProvider


def build_llm(settings: Settings, env: Mapping[str, str] | None = None) -> LLMProvider:
    env = os.environ if env is None else env
    provider = settings.llm_provider
    if provider not in ("gemini", "ollama"):
        raise DomainValidationError(f"unknown llm provider {provider!r} (use 'gemini' or 'ollama')")
    if not settings.llm_model:
        raise DomainValidationError(
            "set GHARCARE_LLM_MODEL to a model name (for Gemini, pick a current Flash model in Google AI Studio; "
            "for Ollama, a model you have pulled)")
    if provider == "gemini":
        warnings.filterwarnings("ignore", message=r".*fixed sampling defaults.*", category=UserWarning)
        key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
        if not key:
            raise DomainValidationError("set GEMINI_API_KEY (a free key from Google AI Studio)")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise LLMError("Gemini support needs: pip install langchain-google-genai") from exc
        timeout = float(env.get("GHARCARE_LLM_TIMEOUT_SECONDS") or 60)
        try:   # our RetryingLLMProvider owns backoff, so keep the SDK's own retries minimal and visible
            chat = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0, google_api_key=key,
                                          timeout=timeout, max_retries=1)
        except Exception:   # older/newer SDK versions may not accept these options
            chat = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0, google_api_key=key)
    else:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise LLMError("Ollama support needs: pip install langchain-ollama") from exc
        chat = ChatOllama(model=settings.llm_model, temperature=0)
    inner: LLMProvider = LangChainChatProvider(chat, name=provider, model=settings.llm_model)
    rpm = requests_per_minute(provider, env)
    if rpm:          # pace inside the retry wrapper, so retries are paced too
        inner = PacedLLMProvider(inner, requests_per_minute=rpm)
    return RetryingLLMProvider(inner)


# Gemini's free tier allows ~15 requests/minute per model; stay a little under it by default.
_DEFAULT_RPM = {"gemini": 12.0}


def requests_per_minute(provider: str, env: Mapping[str, str]) -> float | None:
    """GHARCARE_LLM_RPM overrides the default; 0 turns pacing off (e.g. on a paid tier)."""
    raw = (env.get("GHARCARE_LLM_RPM") or "").strip()
    if not raw:
        return _DEFAULT_RPM.get(provider)
    try:
        value = float(raw)
    except ValueError:
        raise DomainValidationError(f"GHARCARE_LLM_RPM must be a number, got {raw!r}") from None
    if value < 0:
        raise DomainValidationError("GHARCARE_LLM_RPM cannot be negative")
    return value or None
