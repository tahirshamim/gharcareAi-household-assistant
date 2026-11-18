"""Composition root: the one place where concrete classes are chosen and wired together.
Everything else receives its dependencies through constructors (dependency injection)."""
from __future__ import annotations

from dataclasses import dataclass

from gharcare.config import Settings
from gharcare.repositories import (
    ApplianceRepository, ChunkRepository, DocumentRepository, ExtractionDraftRepository, MaintenanceLogRepository,
    MaintenanceRepository, RepairRepository, SQLiteApplianceRepository, SQLiteChunkRepository, SQLiteDatabase,
    SQLiteDocumentRepository, SQLiteExtractionDraftRepository, SQLiteMaintenanceLogRepository,
    SQLiteMaintenanceRepository, SQLiteRepairRepository, SQLiteWarrantyRepository, WarrantyRepository,
)
from gharcare.storage import ArtifactStore, LocalArtifactStore


@dataclass
class AppContainer:
    settings: Settings
    db: SQLiteDatabase            # also the TransactionManager for multi-step operations
    appliances: ApplianceRepository
    warranties: WarrantyRepository
    maintenance: MaintenanceRepository
    maintenance_logs: MaintenanceLogRepository
    repairs: RepairRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    extraction_drafts: ExtractionDraftRepository
    artifacts: ArtifactStore

    @classmethod
    def build(cls, settings: Settings) -> "AppContainer":
        db = SQLiteDatabase(settings.db_path)
        return cls(
            settings=settings, db=db,
            appliances=SQLiteApplianceRepository(db),
            warranties=SQLiteWarrantyRepository(db),
            maintenance=SQLiteMaintenanceRepository(db),
            maintenance_logs=SQLiteMaintenanceLogRepository(db),
            repairs=SQLiteRepairRepository(db),
            documents=SQLiteDocumentRepository(db),
            chunks=SQLiteChunkRepository(db),
            extraction_drafts=SQLiteExtractionDraftRepository(db),
            artifacts=LocalArtifactStore(settings.artifacts_dir, settings.max_upload_bytes),
        )

    def close(self) -> None:
        self.db.close()
