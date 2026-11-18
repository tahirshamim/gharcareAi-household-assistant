"""Wires the RAG services (ingestion + retrieval) onto an AppContainer."""
from __future__ import annotations

from dataclasses import dataclass

from gharcare.container import AppContainer
from gharcare.ingestion import IngestionService
from gharcare.retrieval import Embedder, HybridRetriever, VectorStore
from gharcare.retrieval.factory import build_embedder, build_vector_store


@dataclass
class RagServices:
    ingestion: IngestionService
    retriever: HybridRetriever
    embedder: Embedder
    vector_store: VectorStore


def build_rag(container: AppContainer, embedder: Embedder | None = None,
              vector_store: VectorStore | None = None) -> RagServices:
    """Pass explicit embedder/vector_store in tests; otherwise they come from Settings."""
    embedder = embedder or build_embedder(container.settings)
    vector_store = vector_store or build_vector_store(container.settings)
    return RagServices(
        ingestion=IngestionService(container.documents, container.chunks, container.artifacts,
                                   embedder, vector_store),
        retriever=HybridRetriever(container.documents, container.chunks, embedder, vector_store),
        embedder=embedder, vector_store=vector_store)


def rebuild_vector_index(app: AppContainer, rag: RagServices, batch_size: int = 32) -> int:
    """Re-embeds every stored chunk into the vector store. SQLite is the source of truth, so an in-memory vector
    store (which is empty after a restart) is restored this way. Returns the number of chunks indexed."""
    from collections import defaultdict
    from gharcare.retrieval.vectorstore import VectorRecord
    by_user = defaultdict(list)
    for chunk in app.chunks.list_all():
        by_user[chunk.user_id].append(chunk)
    total = 0
    for user_id, chunks in by_user.items():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = rag.embedder.embed_documents([c.text for c in batch])
            rag.vector_store.upsert(user_id, [VectorRecord(c.id, c.document_id, v) for c, v in zip(batch, vectors)])
            total += len(batch)
    return total
