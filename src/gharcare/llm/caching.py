"""Disk cache for LLM responses (JSON lines). Keyed by a hash of the model namespace and the full prompt, so
changing a prompt, the retrieved excerpts or the model automatically misses the cache. Only SUCCESSFUL responses
are stored, so errors are never replayed. Used by evaluations so a run stopped by a quota limit can resume,
and repeated runs cost nothing. Note: the cache file contains model answers (which can quote your documents)."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Sequence

from gharcare.llm.base import LLMMessage, LLMProvider, LLMResponse


class CachingLLMProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, path: str | Path, namespace: str = "") -> None:
        self._inner, self._path, self._namespace = inner, Path(path), namespace
        self.name = f"cached({inner.name})"
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        self._store: dict[str, tuple[str, str]] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._store[row["k"]] = (row["t"], row.get("m", "cached"))
                except (ValueError, KeyError, TypeError):
                    continue                                   # skip a corrupt line instead of failing

    def _key(self, messages: Sequence[LLMMessage]) -> str:
        payload = json.dumps([self._namespace, [(m.role, m.content) for m in messages]], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, messages: Sequence[LLMMessage], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> LLMResponse:
        key = self._key(messages)
        with self._lock:
            if key in self._store:
                self.hits += 1
                text, model = self._store[key]
                return LLMResponse(text=text, model=model)
        response = self._inner.generate(messages, temperature=temperature, max_tokens=max_tokens)
        with self._lock:
            self.misses += 1
            self._store[key] = (response.text, response.model)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"k": key, "t": response.text, "m": response.model}, ensure_ascii=False) + "\n")
        return response
