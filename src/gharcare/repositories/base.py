"""Abstract repository interfaces.

Services and the LangGraph workflow depend on THESE classes, never on SQLite. Swapping the
database (e.g. PostgreSQL) or mocking storage in tests means writing a new implementation,
not touching business logic.

Isolation rule: every method takes user_id and must never return or modify another user's data.
A record that exists but belongs to someone else is reported as NotFoundError (not 'forbidden'),
so IDs of other users cannot be probed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from datetime import date
from typing import Sequence

from gharcare.domain import (
    Appliance, Chunk, Document, DocumentType, DraftStatus, ExtractionDraft, MaintenanceLog, MaintenanceTask, RepairRecord, Warranty,
)


class ApplianceRepository(ABC):
    @abstractmethod
    def add(self, appliance: Appliance) -> Appliance: ...
    @abstractmethod
    def get(self, user_id: str, appliance_id: str) -> Appliance: ...
    @abstractmethod
    def list_for_user(self, user_id: str) -> Sequence[Appliance]: ...
    @abstractmethod
    def update(self, appliance: Appliance) -> Appliance: ...
    @abstractmethod
    def delete(self, user_id: str, appliance_id: str) -> None:
        """Cascades to warranties, maintenance tasks and repairs. Documents are KEPT
        (appliance link cleared) because they are the user's original records."""


class WarrantyRepository(ABC):
    @abstractmethod
    def add(self, warranty: Warranty) -> Warranty: ...
    @abstractmethod
    def get(self, user_id: str, warranty_id: str) -> Warranty: ...
    @abstractmethod
    def list_for_user(self, user_id: str, appliance_id: str | None = None) -> Sequence[Warranty]: ...
    @abstractmethod
    def update(self, warranty: Warranty) -> Warranty: ...
    @abstractmethod
    def delete(self, user_id: str, warranty_id: str) -> None: ...


class MaintenanceRepository(ABC):
    @abstractmethod
    def add(self, task: MaintenanceTask) -> MaintenanceTask: ...
    @abstractmethod
    def get(self, user_id: str, task_id: str) -> MaintenanceTask: ...
    @abstractmethod
    def list_for_user(self, user_id: str, appliance_id: str | None = None) -> Sequence[MaintenanceTask]: ...
    @abstractmethod
    def update(self, task: MaintenanceTask) -> MaintenanceTask: ...
    @abstractmethod
    def delete(self, user_id: str, task_id: str) -> None: ...


class RepairRepository(ABC):
    @abstractmethod
    def add(self, repair: RepairRecord) -> RepairRecord: ...
    @abstractmethod
    def get(self, user_id: str, repair_id: str) -> RepairRecord: ...
    @abstractmethod
    def list_for_user(
        self, user_id: str, *, appliance_id: str | None = None,
        since: date | None = None, until: date | None = None,
    ) -> Sequence[RepairRecord]:
        """Newest first. since/until are inclusive bounds on reported_date."""
    @abstractmethod
    def update(self, repair: RepairRecord) -> RepairRecord: ...
    @abstractmethod
    def delete(self, user_id: str, repair_id: str) -> None: ...


class DocumentRepository(ABC):
    @abstractmethod
    def add(self, document: Document) -> Document: ...
    @abstractmethod
    def get(self, user_id: str, document_id: str) -> Document: ...
    @abstractmethod
    def list_for_user(
        self, user_id: str, *, appliance_id: str | None = None, doc_type: DocumentType | None = None,
    ) -> Sequence[Document]: ...
    @abstractmethod
    def find_by_hash(self, user_id: str, sha256: str) -> Sequence[Document]:
        """Lets the upload service warn 'you already uploaded this file'."""
    @abstractmethod
    def update(self, document: Document) -> Document: ...
    @abstractmethod
    def delete(self, user_id: str, document_id: str) -> None:
        """Removes the metadata row only. Original artifacts are never deleted in v1."""


class ChunkRepository(ABC):
    @abstractmethod
    def add_many(self, chunks: Sequence[Chunk]) -> None:
        """All chunks must belong to documents owned by their user_id (else NotFoundError)."""
    @abstractmethod
    def list_for_user(self, user_id: str, document_ids: Sequence[str] | None = None) -> list[Chunk]:
        """document_ids=None means ALL of the user's chunks; an EMPTY list means none
        (an appliance with no documents must not fall back to searching everything)."""
    @abstractmethod
    def list_all(self) -> list[Chunk]:
        """Every chunk of every user (maintenance only: used to rebuild the vector index)."""
    @abstractmethod
    def get_many(self, user_id: str, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...
    @abstractmethod
    def count_for_document(self, user_id: str, document_id: str) -> int: ...
    @abstractmethod
    def delete_for_document(self, user_id: str, document_id: str) -> int:
        """Returns the number of chunks removed."""


class MaintenanceLogRepository(ABC):
    @abstractmethod
    def add(self, log: MaintenanceLog) -> MaintenanceLog: ...
    @abstractmethod
    def get(self, user_id: str, log_id: str) -> MaintenanceLog: ...
    @abstractmethod
    def list_for_user(self, user_id: str, *, appliance_id: str | None = None,
                      since: date | None = None, until: date | None = None) -> Sequence[MaintenanceLog]:
        """Newest first. since/until are inclusive bounds on done_date."""
    @abstractmethod
    def update(self, log: MaintenanceLog) -> MaintenanceLog: ...
    @abstractmethod
    def delete(self, user_id: str, log_id: str) -> None: ...


class TransactionManager(ABC):
    """Groups several repository calls into ONE atomic unit: everything commits, or nothing does.
    Re-entrant: repository calls made inside `transaction()` join the outer transaction.
    An exception raised inside the block must propagate (not be swallowed) so the rollback happens."""
    @abstractmethod
    def transaction(self) -> AbstractContextManager: ...


class ExtractionDraftRepository(ABC):
    @abstractmethod
    def add(self, draft: ExtractionDraft) -> ExtractionDraft: ...
    @abstractmethod
    def get(self, user_id: str, draft_id: str) -> ExtractionDraft: ...
    @abstractmethod
    def list_for_user(self, user_id: str, *, status: DraftStatus | None = None,
                      document_id: str | None = None) -> Sequence[ExtractionDraft]:
        """Oldest first."""
    @abstractmethod
    def update(self, draft: ExtractionDraft) -> ExtractionDraft: ...
