"""IngestionService: upload -> store original -> extract text -> chunk -> embed -> index.

State rules (so the system never lies about what it can search):
* A Document row exists as soon as the file is valid and stored. Its `index_status` always
  reflects reality: PENDING while working, INDEXED only after chunks AND vectors are saved,
  FAILED otherwise.
* On any failure the partial chunks/vectors are removed, so a FAILED document is never half-searchable.
* Expected, user-fixable problems (a scanned PDF with no text layer) are RETURNED as a warning.
  Unexpected infrastructure failures (embedding model down) are RAISED after cleanup; the
  document stays FAILED and `reindex()` retries without re-uploading.
"""
from __future__ import annotations

import mimetypes
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from gharcare.domain import Chunk, Document, DocumentType, IndexStatus
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError, EmbeddingError
from gharcare.ingestion.chunker import Chunker
from gharcare.ingestion.extractors import TextExtractor, default_extractors
from gharcare.repositories.base import ChunkRepository, DocumentRepository
from gharcare.retrieval.embeddings import Embedder
from gharcare.retrieval.vectorstore import VectorRecord, VectorStore
from gharcare.storage.base import ArtifactStore

NO_TEXT_MESSAGE = (
    "No readable text was found. If this is a scanned or photographed document, it cannot be "
    "searched yet (OCR is not supported). The file is saved; please enter its details manually."
)


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    chunk_count: int
    warning: str | None = None
    duplicate: bool = False   # True when this exact file was already uploaded for the same appliance

    @property
    def indexed(self) -> bool:
        return self.document.index_status == IndexStatus.INDEXED


class IngestionService:
    def __init__(self, documents: DocumentRepository, chunks: ChunkRepository, artifacts: ArtifactStore,
                 embedder: Embedder, vector_store: VectorStore, chunker: Chunker | None = None,
                 extractors: Mapping[str, TextExtractor] | None = None, embed_batch_size: int = 32) -> None:
        self._documents, self._chunks, self._artifacts = documents, chunks, artifacts
        self._embedder, self._vectors = embedder, vector_store
        self._chunker = chunker or Chunker()
        self._extractors = dict(extractors) if extractors is not None else default_extractors()
        self._batch = embed_batch_size

    # ---- public API ----------------------------------------------------------------------
    def ingest(self, user_id: str, filename: str, content: bytes, doc_type: DocumentType | str,
               appliance_id: str | None = None, mime_type: str | None = None) -> IngestionResult:
        validate_identifier(user_id, "user_id")
        self._extractor_for(filename)                      # reject unsupported types BEFORE storing
        try:
            doc_type = DocumentType(doc_type)
        except ValueError as exc:
            raise DomainValidationError(f"unknown document type {doc_type!r}") from exc

        ref = self._artifacts.save_original(user_id, filename, content)   # also validates size/empty
        same_link = [d for d in self._documents.find_by_hash(user_id, ref.sha256)
                     if d.appliance_id == appliance_id]
        if same_link:
            existing = same_link[0]
            if existing.index_status == IndexStatus.INDEXED:
                count = self._chunks.count_for_document(user_id, existing.id)
                return IngestionResult(existing, count, duplicate=True)
            return replace(self._index(existing, content), duplicate=True)

        document = self._documents.add(Document(
            user_id=user_id, original_filename=filename, doc_type=doc_type, sha256=ref.sha256,
            storage_path=ref.relative_path, size_bytes=ref.size_bytes, appliance_id=appliance_id,
            mime_type=mime_type or mimetypes.guess_type(filename)[0]))
        return self._index(document, content)

    def reindex(self, user_id: str, document_id: str) -> IngestionResult:
        document = self._documents.get(user_id, document_id)
        self._extractor_for(document.original_filename)
        return self._index(document, self._artifacts.read(user_id, document.storage_path))

    def delete_document(self, user_id: str, document_id: str) -> None:
        """Removes the document row, its chunks and its vectors. The original file is kept."""
        document = self._documents.get(user_id, document_id)   # NotFoundError for other users' ids
        self._vectors.delete_document(user_id, document.id)
        self._chunks.delete_for_document(user_id, document.id)
        self._documents.delete(user_id, document.id)

    # ---- internals -----------------------------------------------------------------------
    def _extractor_for(self, filename: str) -> TextExtractor:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in self._extractors:
            raise DomainValidationError(
                f"unsupported file type {suffix or '(none)'!r}; supported: {', '.join(sorted(self._extractors))}")
        return self._extractors[suffix]

    def _set_status(self, document: Document, status: IndexStatus) -> Document:
        return self._documents.update(replace(document, index_status=status))

    def _cleanup(self, document: Document) -> None:
        self._vectors.delete_document(document.user_id, document.id)
        self._chunks.delete_for_document(document.user_id, document.id)

    def _embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self._batch):
            batch = [c.text for c in chunks[start:start + self._batch]]
            result = self._embedder.embed_documents(batch)
            if len(result) != len(batch):
                raise EmbeddingError(f"embedder returned {len(result)} vectors for {len(batch)} texts")
            vectors.extend(result)
        return vectors

    def _index(self, document: Document, content: bytes) -> IngestionResult:
        user_id = document.user_id
        extractor = self._extractor_for(document.original_filename)
        try:
            pages = [p for p in extractor.extract(content) if p.text.strip()]
            if not pages:
                return self._fail(document, NO_TEXT_MESSAGE)
            self._artifacts.save_extracted(user_id, document.id, {
                "extractor": extractor.name, "page_count": len(pages),
                "pages": [{"page": p.page_number, "text": p.text} for p in pages]})
            chunks = self._chunker.chunk_pages(user_id, document.id, pages)
            if not chunks:
                return self._fail(document, NO_TEXT_MESSAGE)
            vectors = self._embed(chunks)                  # embed first: a failure leaves nothing half-written
            self._cleanup(document)                        # replace any earlier index of this document
            self._chunks.add_many(chunks)
            self._vectors.upsert(user_id, [VectorRecord(c.id, c.document_id, v) for c, v in zip(chunks, vectors)])
        except Exception:
            self._cleanup(document)
            self._set_status(document, IndexStatus.FAILED)
            raise
        return IngestionResult(self._set_status(document, IndexStatus.INDEXED), len(chunks))

    def _fail(self, document: Document, message: str) -> IngestionResult:
        self._cleanup(document)
        return IngestionResult(self._set_status(document, IndexStatus.FAILED), 0, warning=message)
