"""Embedding models behind one interface."""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Sequence

from gharcare.errors import EmbeddingError
from gharcare.retrieval.text import tokenize


class Embedder(ABC):
    name = "base"

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Separate from embed_documents because some models (e5) need different prefixes."""


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder (standard library only).

    HONEST LABEL: this is a LEXICAL baseline, not a semantic model. It matches shared words, not
    meaning ('freezer ice' will not match 'frost'). It exists so the pipeline can be tested and
    demonstrated offline, and as the baseline your real model must beat in the evaluation."""
    name = "hashing"

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for token, count in Counter(tokenize(text)).items():
            h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self._dimension] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedder(Embedder):
    """Real multilingual model via sentence-transformers (optional dependency, loaded lazily).
    Default 'intfloat/multilingual-e5-small' is small enough for a laptop; 'BAAI/bge-m3' is
    stronger but ~2 GB. e5 models require the 'query: ' / 'passage: ' prefixes, handled here."""
    name = "sentence-transformers"

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", batch_size: int = 32) -> None:
        self._model_name, self._batch_size = model_name, batch_size
        self._model = None
        self._uses_e5_prefixes = "e5" in model_name.lower()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError("embeddings need sentence-transformers: pip install sentence-transformers") from exc
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:  # download/load failures vary widely by backend
                raise EmbeddingError(f"could not load embedding model {self._model_name!r}: {exc}") from exc
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            matrix = self._load().encode(list(texts), batch_size=self._batch_size,
                                         normalize_embeddings=True, show_progress_bar=False)
            return [[float(x) for x in row] for row in matrix]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"embedding failed: {exc}") from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefix = "passage: " if self._uses_e5_prefixes else ""
        return self._encode([prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        prefix = "query: " if self._uses_e5_prefixes else ""
        return self._encode([prefix + text])[0]
