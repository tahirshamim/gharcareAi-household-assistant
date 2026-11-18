"""Logging setup that can never leak the API key: every record passes through a redacting filter."""
from __future__ import annotations

import logging
import os
from typing import Iterable, Mapping

SECRET_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def secrets_from_env(env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    return [env[n] for n in SECRET_ENV_NAMES if env.get(n)]


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***")
    return text


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = list(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact(record.getMessage(), self._secrets), None
        return True


def configure_logging(level: int = logging.WARNING, env: Mapping[str, str] | None = None) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter(secrets_from_env(env)))
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
