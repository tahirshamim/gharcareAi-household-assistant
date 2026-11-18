"""Vector database interface + two implementations (in-memory for tests, Chroma for real use).
Both must pass the SAME contract tests (tests/test_vectorstore.py)."""
from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from gharcare.errors import StorageError


@dataclass(frozen=True)
class VectorRecord:
    chunk_id: str
    document_id: str
    embedding: Sequence[float]


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    similarity: float  # cosine similarity, higher is better (1.0 = identical direction)


class VectorStore(ABC):
    """Only ids + embeddings + (user_id, document_id) metadata live here. Chunk TEXT stays in
    SQLite, so this index can be dropped and rebuilt at any time."""

    @abstractmethod
    def upsert(self, user_id: str, records: Sequence[VectorRecord]) -> None: ...

    @abstractmethod
    def query(self, user_id: str, embedding: Sequence[float], k: int,
              document_ids: Sequence[str] | None = None) -> list[VectorHit]:
        """document_ids=None -> all of the user's documents; [] -> nothing (returns [])."""

    @abstractmethod
    def delete_document(self, user_id: str, document_id: str) -> None: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, str, list[float]]] = {}  # chunk_id -> (user, doc, vector)
        self._lock = threading.RLock()

    def upsert(self, user_id: str, records: Sequence[VectorRecord]) -> None:
        with self._lock:
            for r in records:
                existing = self._rows.get(r.chunk_id)
                if existing and existing[0] != user_id:
                    raise StorageError("chunk id already belongs to another user")
                self._rows[r.chunk_id] = (user_id, r.document_id, [float(x) for x in r.embedding])

    def query(self, user_id, embedding, k, document_ids=None) -> list[VectorHit]:
        if k < 1 or (document_ids is not None and len(document_ids) == 0):
            return []
        allowed = set(document_ids) if document_ids is not None else None
        with self._lock:
            hits = [VectorHit(cid, _cosine(embedding, vec))
                    for cid, (uid, did, vec) in self._rows.items()
                    if uid == user_id and (allowed is None or did in allowed)]
        hits.sort(key=lambda h: (-h.similarity, h.chunk_id))
        return hits[:k]

    def delete_document(self, user_id: str, document_id: str) -> None:
        with self._lock:
            for cid in [c for c, (u, d, _) in self._rows.items() if u == user_id and d == document_id]:
                del self._rows[cid]


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection (optional dependency, imported lazily). Cosine space.
    Telemetry is disabled because household documents are private."""

    def __init__(self, path: str | Path, collection_name: str = "gharcare_chunks") -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise StorageError("Chroma support needs chromadb: pip install chromadb") from exc
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
            self._col = self._client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"})
        except Exception as exc:
            raise StorageError(f"could not open Chroma at {path}: {exc}") from exc

    @staticmethod
    def _where(user_id: str, document_ids: Sequence[str] | None) -> dict:
        if document_ids is None:
            return {"user_id": user_id}
        return {"$and": [{"user_id": user_id}, {"document_id": {"$in": list(document_ids)}}]}

    def upsert(self, user_id: str, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        try:
            self._col.upsert(
                ids=[r.chunk_id for r in records],
                embeddings=[[float(x) for x in r.embedding] for r in records],
                metadatas=[{"user_id": user_id, "document_id": r.document_id} for r in records],
            )
        except Exception as exc:
            raise StorageError(f"vector upsert failed: {exc}") from exc

    def query(self, user_id, embedding, k, document_ids=None) -> list[VectorHit]:
        if k < 1 or (document_ids is not None and len(document_ids) == 0):
            return []
        try:
            total = self._col.count()
            if total == 0:
                return []
            res = self._col.query(query_embeddings=[[float(x) for x in embedding]],
                                  n_results=min(k, total), where=self._where(user_id, document_ids))
            return [VectorHit(cid, 1.0 - float(dist))   # Chroma returns cosine DISTANCE
                    for cid, dist in zip(res["ids"][0], res["distances"][0])]
        except Exception as exc:
            raise StorageError(f"vector query failed: {exc}") from exc

    def delete_document(self, user_id: str, document_id: str) -> None:
        try:
            self._col.delete(where={"$and": [{"user_id": user_id}, {"document_id": document_id}]})
        except Exception as exc:
            raise StorageError(f"vector delete failed: {exc}") from exc
