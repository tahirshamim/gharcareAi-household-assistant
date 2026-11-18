"""Okapi BM25 keyword search (standard library only).

Why keyword search matters here: model numbers and error codes ('E4', 'HRF-306') are exact tokens
that embeddings handle poorly. Built per query from the user's (already filtered) chunks; at
household scale (thousands of chunks) this takes milliseconds."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from gharcare.retrieval.text import tokenize


class BM25Index:
    def __init__(self, documents: Sequence[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._ids = [doc_id for doc_id, _ in documents]
        self._tfs = [Counter(tokenize(text)) for _, text in documents]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter[str] = Counter()
        for tf in self._tfs:
            self._doc_freq.update(tf.keys())
        self._k1, self._b = k1, b

    def _idf(self, term: str) -> float:
        n, df = len(self._ids), self._doc_freq[term]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (id, score) pairs with score > 0, best first; ties keep insertion order."""
        terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for index, tf in enumerate(self._tfs):
            norm = 1 - self._b + self._b * (self._lengths[index] / self._avg_length if self._avg_length else 0)
            score = sum(
                self._idf(t) * tf[t] * (self._k1 + 1) / (tf[t] + self._k1 * norm)
                for t in terms if tf[t]
            )
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(self._ids[i], s) for s, i in scored[:k]]
