"""ArtifactStore: where original uploads, extraction results and generated reports live.

Layout (content-addressed, per user):
    {user_id}/originals/{sha[:2]}/{sha256}{ext}   immutable, deduplicated by content hash
    {user_id}/extracted/{document_id}/v0001.json  every extraction/correction is a NEW version
    {user_id}/reports/{timestamp}_{name}{ext}     generated reports, never overwritten

Which appliance a document belongs to is recorded in the database, NOT in the file path, so
re-assigning a document never moves files, and one file can serve several appliances.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactRef:
    relative_path: str   # POSIX-style, relative to the store root; safe to persist in the DB
    sha256: str
    size_bytes: int
    created: bool        # False when identical content already existed (deduplicated)


class ArtifactStore(ABC):
    @abstractmethod
    def save_original(self, user_id: str, filename: str, content: bytes) -> ArtifactRef: ...

    @abstractmethod
    def read(self, user_id: str, relative_path: str) -> bytes:
        """Raises NotFoundError if missing, StorageError if the path escapes the user's area."""

    @abstractmethod
    def exists(self, user_id: str, relative_path: str) -> bool: ...

    @abstractmethod
    def save_extracted(self, user_id: str, document_id: str, data: Mapping[str, Any]) -> ArtifactRef:
        """Writes a new version; earlier versions are kept so corrections stay auditable."""

    @abstractmethod
    def load_latest_extracted(self, user_id: str, document_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def save_report(self, user_id: str, name: str, content: bytes, extension: str = ".md") -> ArtifactRef: ...
