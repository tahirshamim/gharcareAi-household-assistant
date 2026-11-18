"""Central configuration. Secrets (API keys) are deliberately NOT stored here;
they are read from the environment by the provider that needs them."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gharcare.errors import DomainValidationError


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("./data")
    max_upload_bytes: int = 20 * 1024 * 1024
    llm_provider: str = "gemini"      # planned: "gemini" | "ollama"; "fake" is for tests
    llm_model: str | None = None      # chosen in the LangGraph step
    embedder: str = "sentence-transformers"           # or "hashing" (lexical baseline, tests/offline)
    embedding_model: str = "intfloat/multilingual-e5-small"
    vector_store: str = "chroma"                      # or "memory" (tests)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.max_upload_bytes <= 0:
            raise DomainValidationError("max_upload_bytes must be positive")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "gharcare.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        kwargs: dict = {}
        if "GHARCARE_DATA_DIR" in env:
            kwargs["data_dir"] = Path(env["GHARCARE_DATA_DIR"])
        if "GHARCARE_MAX_UPLOAD_MB" in env:
            try:
                kwargs["max_upload_bytes"] = int(float(env["GHARCARE_MAX_UPLOAD_MB"]) * 1024 * 1024)
            except ValueError as exc:
                raise DomainValidationError("GHARCARE_MAX_UPLOAD_MB must be a number") from exc
        if "GHARCARE_LLM_PROVIDER" in env:
            kwargs["llm_provider"] = env["GHARCARE_LLM_PROVIDER"].strip().lower()
        if "GHARCARE_LLM_MODEL" in env:
            kwargs["llm_model"] = env["GHARCARE_LLM_MODEL"].strip() or None
        for env_name, field_name in (("GHARCARE_EMBEDDER", "embedder"),
                                     ("GHARCARE_VECTOR_STORE", "vector_store")):
            if env_name in env:
                kwargs[field_name] = env[env_name].strip().lower()
        if env.get("GHARCARE_EMBEDDING_MODEL", "").strip():
            kwargs["embedding_model"] = env["GHARCARE_EMBEDDING_MODEL"].strip()
        return cls(**kwargs)
