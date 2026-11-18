"""Hybrid retrieval: vector search + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF).

IMPORTANT for Step 4 (grounding): RRF scores are rank-based, so a top result always has a
'decent' fused score even when nothing is truly relevant. They cannot be used to decide "I don't
know". The ABSOLUTE signals (`vector_similarity`, `keyword_score`) are returned on every result for
exactly that purpose; the evaluation harness shows how to calibrate a threshold from them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from gharcare.domain import Chunk, DocumentType, IndexStatus
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError
from gharcare.repositories.base import ChunkRepository, DocumentRepository
from gharcare.retrieval.bm25 import BM25Index
from gharcare.retrieval.embeddings import Embedder
from gharcare.retrieval.vectorstore import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    document_name: str
    score: float                       # fused RRF score (ordering only)
    vector_rank: int | None = None     # 1-based rank in the vector results, None if absent
    keyword_rank: int | None = None
    vector_similarity: float | None = None   # absolute cosine similarity
    keyword_score: float | None = None       # absolute BM25 score

    @property
    def citation(self) -> str:
        text = self.document_name
        if self.chunk.page_number is not None:
            text += f", p.{self.chunk.page_number}"
        if self.chunk.section:
            text += f" ({self.chunk.section})"
        return text


def fuse_rankings(rankings: Sequence[Sequence[str]], weights: Sequence[float] | None = None,
                  rrf_k: int = 60) -> dict[str, float]:
    """score(id) = sum over rankings of weight / (rrf_k + rank). Rank starts at 1."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (rrf_k + rank)
    return scores


def reserve_slots(ordered: Sequence[str], keyword_ids: Sequence[str], vector_ids: Sequence[str], k: int, reserve: int) -> list[str]:
    """Top-k of the fused order, but the best `reserve` results of EACH retriever are guaranteed a place.
    Pure rank fusion can bury a perfect exact-keyword match (an error code, 'pork') under chunks that are merely
    mediocre in both lists; this keeps the strongest signal of each retriever in the answer set.
    At most k // 2 slots are reserved (taken alternately: keyword #1, vector #1, keyword #2, ...), so the best
    fused results always keep the other half. Without that cap, k=4 with reserve=2 let reserved chunks fill
    every slot and evict the overall best match (found on the real Gorenje manual)."""
    top = list(ordered[:k])
    interleaved = [cid for pair in zip(keyword_ids[:reserve], vector_ids[:reserve]) for cid in pair]
    longer = keyword_ids if len(keyword_ids) > len(vector_ids) else vector_ids
    interleaved += list(longer[min(len(keyword_ids), len(vector_ids)):reserve])
    reserved = list(dict.fromkeys(interleaved))[: k // 2]
    for chunk_id in reserved:
        if chunk_id in top:
            continue
        for position in range(len(top) - 1, -1, -1):          # replace the lowest-ranked result that is not itself reserved
            if top[position] not in reserved:
                top[position] = chunk_id
                break
    return top


class HybridRetriever:
    def __init__(self, documents: DocumentRepository, chunks: ChunkRepository,
                 embedder: Embedder, vector_store: VectorStore) -> None:
        self._documents, self._chunks = documents, chunks
        self._embedder, self._vectors = embedder, vector_store

    def retrieve(self, user_id: str, query: str, *, appliance_id: str | None = None,
                 doc_types: Sequence[DocumentType] | None = None, k: int = 5,
                 candidate_pool: int = 20, vector_weight: float = 1.0,
                 keyword_weight: float = 1.0, min_vector_similarity: float = 0.0,
                 reserve_per_source: int = 2) -> list[RetrievedChunk]:
        validate_identifier(user_id, "user_id")
        if k < 1:
            raise DomainValidationError("k must be at least 1")
        query = (query or "").strip()
        if not query:
            return []

        wanted = set(doc_types) if doc_types else None
        docs = [d for d in self._documents.list_for_user(user_id, appliance_id=appliance_id)
                if d.index_status == IndexStatus.INDEXED and (wanted is None or d.doc_type in wanted)]
        if not docs:
            return []                       # no eligible documents: never widen the search silently
        names = {d.id: d.original_filename for d in docs}
        chunks = self._chunks.list_for_user(user_id, list(names))
        by_id = {c.id: c for c in chunks}
        if not by_id:
            return []

        # Dropped: vector hits whose chunk no longer exists (deleted between steps) and hits with
        # similarity <= min_vector_similarity (0.0 = no relevance at all, pure noise). Real embedding
        # models rarely score below ~0.5 even for unrelated text, so their 'not relevant' threshold must
        # be calibrated from evaluation data (see evaluation report), not guessed.
        vector_hits = [h for h in self._vectors.query(
            user_id, self._embedder.embed_query(query), candidate_pool, list(names))
            if h.chunk_id in by_id and h.similarity > min_vector_similarity]
        keyword_hits = BM25Index([(c.id, c.text) for c in chunks]).search(query, candidate_pool)

        vec_rank = {h.chunk_id: i for i, h in enumerate(vector_hits, 1)}
        vec_sim = {h.chunk_id: h.similarity for h in vector_hits}
        kw_rank = {cid: i for i, (cid, _) in enumerate(keyword_hits, 1)}
        kw_score = dict(keyword_hits)

        fused = fuse_rankings([[h.chunk_id for h in vector_hits], [cid for cid, _ in keyword_hits]],
                              [vector_weight, keyword_weight])
        ordered = sorted(fused, key=lambda cid: (-fused[cid], -kw_score.get(cid, 0.0),
                                                  by_id[cid].document_id, by_id[cid].chunk_index))
        chosen = reserve_slots(ordered, [cid for cid, _ in keyword_hits], [h.chunk_id for h in vector_hits], k, reserve_per_source)
        chosen.sort(key=ordered.index)                       # keep the fused order for display
        return [RetrievedChunk(
            chunk=by_id[cid], document_name=names[by_id[cid].document_id], score=fused[cid],
            vector_rank=vec_rank.get(cid), keyword_rank=kw_rank.get(cid),
            vector_similarity=vec_sim.get(cid), keyword_score=kw_score.get(cid),
        ) for cid in chosen]
