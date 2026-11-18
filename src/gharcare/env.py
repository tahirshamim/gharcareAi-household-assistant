"""Minimal .env support (standard library only).

Rules: KEY=VALUE per line; '#' starts a comment; optional 'export ' prefix; values may be quoted;
EMPTY values are ignored (so a blank 'GEMINI_API_KEY=' placeholder counts as 'not set'); variables
already present in the real environment are NOT overwritten unless override=True. Only variable
NAMES are ever returned or logged, never values."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.lstrip("\ufeff").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not _KEY.match(key):
            continue
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            value = value.split(" #", 1)[0].strip()      # inline comment (only when preceded by a space)
        values[key] = value
    return values


def load_env_file(path: str | Path | None = None, *, override: bool = False,
                  environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Loads variables into `environ` (default: os.environ). Returns the NAMES that were set.
    A missing file is not an error (the real environment may already be configured)."""
    environ = os.environ if environ is None else environ
    file = Path(path or environ.get("GHARCARE_ENV_FILE") or ".env")
    try:
        text = file.read_text(encoding="utf-8-sig")        # utf-8-sig: Windows Notepad may add a BOM
    except FileNotFoundError:
        return []
    loaded = []
    for key, value in parse_env_text(text).items():
        if value and (override or key not in environ):
            environ[key] = value
            loaded.append(key)
    return loaded
