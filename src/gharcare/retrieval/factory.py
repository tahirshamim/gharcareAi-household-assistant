"""Builds embedder / vector store from Settings, with clear errors for unknown or missing options."""
from __future__ import annotations

from gharcare.config import Settings
from gharcare.errors import DomainValidationError
from gharcare.retrieval.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from gharcare.retrieval.vectorstore import ChromaVectorStore, InMemoryVectorStore, VectorStore


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    if settings.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise DomainValidationError(f"unknown embedder {settings.embedder!r} (use 'sentence-transformers' or 'hashing')")


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    if settings.vector_store == "chroma":
        return ChromaVectorStore(settings.chroma_dir)
    raise DomainValidationError(f"unknown vector store {settings.vector_store!r} (use 'chroma' or 'memory')")
