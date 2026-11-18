"""SQLite implementations of the repository interfaces (standard library only)."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Sequence

from gharcare.domain import (
    Appliance, Chunk, Document, DocumentType, DraftStatus, ExtractionDraft, IndexStatus, MaintenanceLog, MaintenanceTask, RepairRecord, Warranty,
)
from gharcare.domain import fields_from_json, fields_to_json
from gharcare.errors import NotFoundError, StorageError
from gharcare.repositories.base import (
    ApplianceRepository, ChunkRepository, DocumentRepository, ExtractionDraftRepository, MaintenanceLogRepository,
    MaintenanceRepository, RepairRepository, TransactionManager, WarrantyRepository,
)

SCHEMA_VERSION = 4

_SCHEMA_V1 = """
CREATE TABLE appliances (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
    brand TEXT, model TEXT, serial_number TEXT, purchase_date TEXT,
    purchase_price TEXT, location TEXT, created_at TEXT NOT NULL
);
CREATE INDEX idx_appliances_user ON appliances(user_id);

CREATE TABLE warranties (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    appliance_id TEXT NOT NULL REFERENCES appliances(id) ON DELETE CASCADE,
    start_date TEXT NOT NULL, duration_months INTEGER NOT NULL,
    provider TEXT, coverage_notes TEXT, source_document_id TEXT,
    verified INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE INDEX idx_warranties_user ON warranties(user_id);

CREATE TABLE maintenance_tasks (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    appliance_id TEXT NOT NULL REFERENCES appliances(id) ON DELETE CASCADE,
    title TEXT NOT NULL, next_due_date TEXT, interval_months INTEGER,
    last_done_date TEXT, notes TEXT, created_at TEXT NOT NULL
);
CREATE INDEX idx_maintenance_user ON maintenance_tasks(user_id);

CREATE TABLE repairs (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    appliance_id TEXT NOT NULL REFERENCES appliances(id) ON DELETE CASCADE,
    reported_date TEXT NOT NULL, problem_description TEXT NOT NULL,
    resolution TEXT, technician TEXT, parts_replaced TEXT NOT NULL DEFAULT '',
    cost TEXT NOT NULL DEFAULT '0.00', covered_by_warranty INTEGER,
    source_document_id TEXT, created_at TEXT NOT NULL
);
CREATE INDEX idx_repairs_user_date ON repairs(user_id, reported_date);

CREATE TABLE documents (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    appliance_id TEXT REFERENCES appliances(id) ON DELETE SET NULL,
    original_filename TEXT NOT NULL, doc_type TEXT NOT NULL, sha256 TEXT NOT NULL,
    storage_path TEXT NOT NULL, size_bytes INTEGER NOT NULL, mime_type TEXT,
    index_status TEXT NOT NULL, uploaded_at TEXT NOT NULL
);
CREATE INDEX idx_documents_user ON documents(user_id);
CREATE INDEX idx_documents_hash ON documents(user_id, sha256);
"""

_SCHEMA_V2 = """
CREATE TABLE chunks (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL, page_number INTEGER, section TEXT, text TEXT NOT NULL
);
CREATE INDEX idx_chunks_user ON chunks(user_id);
CREATE INDEX idx_chunks_document ON chunks(document_id);
"""

_SCHEMA_V3 = """
CREATE TABLE maintenance_logs (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    appliance_id TEXT NOT NULL REFERENCES appliances(id) ON DELETE CASCADE,
    task_id TEXT REFERENCES maintenance_tasks(id) ON DELETE SET NULL,
    done_date TEXT NOT NULL, description TEXT NOT NULL, cost TEXT NOT NULL DEFAULT '0.00',
    technician TEXT, notes TEXT, created_at TEXT NOT NULL
);
CREATE INDEX idx_maintenance_logs_user_date ON maintenance_logs(user_id, done_date);
"""

_SCHEMA_V4 = """
CREATE TABLE extraction_drafts (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    doc_type TEXT NOT NULL, status TEXT NOT NULL, fields_json TEXT NOT NULL, final_json TEXT,
    created_at TEXT NOT NULL, resolved_at TEXT
);
CREATE INDEX idx_drafts_user_status ON extraction_drafts(user_id, status);
"""

# version -> SQL that upgrades a database FROM version-1 TO version
_MIGRATIONS = {1: _SCHEMA_V1, 2: _SCHEMA_V2, 3: _SCHEMA_V3, 4: _SCHEMA_V4}

_PARTS_SEPARATOR = "\x1f"  # ASCII unit separator: cannot appear in typed text


class SQLiteDatabase(TransactionManager):
    """Owns the connection and schema. One shared connection guarded by a lock keeps
    behaviour identical for ':memory:' (tests) and file databases (the app)."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        try:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"cannot open database at {self._path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        self._depth = 0
        try:
            self._migrate()
        except BaseException:
            self._conn.close()      # a refused or failed migration must not leave the file open (locked on Windows)
            raise

    def _migrate(self) -> None:
        """Applies each missing migration in order, atomically. A database written by a NEWER
        version of the app is refused rather than silently misread."""
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StorageError(
                    f"database schema version {version} is newer than this app supports ({SCHEMA_VERSION})"
                )
            for target in range(version + 1, SCHEMA_VERSION + 1):
                try:
                    self._conn.executescript(
                        f"BEGIN;\n{_MIGRATIONS[target]}\nPRAGMA user_version = {target};\nCOMMIT;"
                    )
                except sqlite3.Error as exc:
                    self._conn.rollback()
                    raise StorageError(f"database migration to version {target} failed: {exc}") from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Re-entrant: only the OUTERMOST block commits or rolls back, so a service can wrap several
        repository calls (each of which opens its own block) into one atomic operation."""
        with self._lock:
            outermost = self._depth == 0
            self._depth += 1
            try:
                yield self._conn
                if outermost:
                    self._conn.commit()
            except BaseException:
                if outermost:
                    self._conn.rollback()
                raise
            finally:
                self._depth -= 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---- value conversion helpers -------------------------------------------------------------

def _d(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _pd(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _money(value: str | None) -> Decimal | None:
    return Decimal(value) if value is not None else None


def _bool_or_none(value: int | None) -> bool | None:
    return None if value is None else bool(value)


def _require_appliance(conn: sqlite3.Connection, user_id: str, appliance_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM appliances WHERE id = ? AND user_id = ?", (appliance_id, user_id)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"appliance {appliance_id} not found")


class _SQLiteRepository:
    _entity = "record"
    _table = ""

    def __init__(self, db: SQLiteDatabase) -> None:
        self._db = db

    def _fetch_one(self, sql: str, params: tuple, what: str):
        with self._db.transaction() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            raise NotFoundError(f"{what} not found")
        return row

    def _fetch_all(self, sql: str, params: tuple):
        with self._db.transaction() as conn:
            return conn.execute(sql, params).fetchall()

    def _delete(self, user_id: str, entity_id: str) -> None:
        with self._db.transaction() as conn:
            cur = conn.execute(
                f"DELETE FROM {self._table} WHERE id = ? AND user_id = ?", (entity_id, user_id)
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"{self._entity} {entity_id} not found")


# ---- Appliances ---------------------------------------------------------------------------

def _appliance(r: sqlite3.Row) -> Appliance:
    return Appliance(
        id=r["id"], user_id=r["user_id"], name=r["name"], brand=r["brand"], model=r["model"],
        serial_number=r["serial_number"], purchase_date=_pd(r["purchase_date"]),
        purchase_price=_money(r["purchase_price"]), location=r["location"],
        created_at=datetime.fromisoformat(r["created_at"]),
    )


class SQLiteApplianceRepository(_SQLiteRepository, ApplianceRepository):
    _entity, _table = "appliance", "appliances"

    def add(self, a: Appliance) -> Appliance:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO appliances VALUES (?,?,?,?,?,?,?,?,?,?)",
                (a.id, a.user_id, a.name, a.brand, a.model, a.serial_number, _d(a.purchase_date),
                 str(a.purchase_price) if a.purchase_price is not None else None,
                 a.location, a.created_at.isoformat()),
            )
        return a

    def get(self, user_id: str, appliance_id: str) -> Appliance:
        return _appliance(self._fetch_one(
            "SELECT * FROM appliances WHERE id = ? AND user_id = ?", (appliance_id, user_id),
            f"appliance {appliance_id}"))

    def list_for_user(self, user_id: str) -> Sequence[Appliance]:
        return [_appliance(r) for r in self._fetch_all(
            "SELECT * FROM appliances WHERE user_id = ? ORDER BY created_at, name", (user_id,))]

    def update(self, a: Appliance) -> Appliance:
        with self._db.transaction() as conn:
            cur = conn.execute(
                "UPDATE appliances SET name=?, brand=?, model=?, serial_number=?, purchase_date=?,"
                " purchase_price=?, location=? WHERE id=? AND user_id=?",
                (a.name, a.brand, a.model, a.serial_number, _d(a.purchase_date),
                 str(a.purchase_price) if a.purchase_price is not None else None,
                 a.location, a.id, a.user_id),
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"appliance {a.id} not found")
        return a

    def delete(self, user_id: str, appliance_id: str) -> None:
        self._delete(user_id, appliance_id)


# ---- Warranties ---------------------------------------------------------------------------

def _warranty(r: sqlite3.Row) -> Warranty:
    return Warranty(
        id=r["id"], user_id=r["user_id"], appliance_id=r["appliance_id"],
        start_date=date.fromisoformat(r["start_date"]), duration_months=r["duration_months"],
        provider=r["provider"], coverage_notes=r["coverage_notes"],
        source_document_id=r["source_document_id"], verified=bool(r["verified"]),
        created_at=datetime.fromisoformat(r["created_at"]),
    )


class SQLiteWarrantyRepository(_SQLiteRepository, WarrantyRepository):
    _entity, _table = "warranty", "warranties"

    def add(self, w: Warranty) -> Warranty:
        with self._db.transaction() as conn:
            _require_appliance(conn, w.user_id, w.appliance_id)
            conn.execute(
                "INSERT INTO warranties VALUES (?,?,?,?,?,?,?,?,?,?)",
                (w.id, w.user_id, w.appliance_id, _d(w.start_date), w.duration_months, w.provider,
                 w.coverage_notes, w.source_document_id, int(w.verified), w.created_at.isoformat()),
            )
        return w

    def get(self, user_id: str, warranty_id: str) -> Warranty:
        return _warranty(self._fetch_one(
            "SELECT * FROM warranties WHERE id = ? AND user_id = ?", (warranty_id, user_id),
            f"warranty {warranty_id}"))

    def list_for_user(self, user_id: str, appliance_id: str | None = None) -> Sequence[Warranty]:
        sql, params = "SELECT * FROM warranties WHERE user_id = ?", [user_id]
        if appliance_id:
            sql, params = sql + " AND appliance_id = ?", params + [appliance_id]
        return [_warranty(r) for r in self._fetch_all(sql + " ORDER BY start_date", tuple(params))]

    def update(self, w: Warranty) -> Warranty:
        with self._db.transaction() as conn:
            _require_appliance(conn, w.user_id, w.appliance_id)
            cur = conn.execute(
                "UPDATE warranties SET appliance_id=?, start_date=?, duration_months=?, provider=?,"
                " coverage_notes=?, source_document_id=?, verified=? WHERE id=? AND user_id=?",
                (w.appliance_id, _d(w.start_date), w.duration_months, w.provider, w.coverage_notes,
                 w.source_document_id, int(w.verified), w.id, w.user_id),
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"warranty {w.id} not found")
        return w

    def delete(self, user_id: str, warranty_id: str) -> None:
        self._delete(user_id, warranty_id)


# ---- Maintenance --------------------------------------------------------------------------

def _task(r: sqlite3.Row) -> MaintenanceTask:
    return MaintenanceTask(
        id=r["id"], user_id=r["user_id"], appliance_id=r["appliance_id"], title=r["title"],
        next_due_date=_pd(r["next_due_date"]), interval_months=r["interval_months"],
        last_done_date=_pd(r["last_done_date"]), notes=r["notes"],
        created_at=datetime.fromisoformat(r["created_at"]),
    )


class SQLiteMaintenanceRepository(_SQLiteRepository, MaintenanceRepository):
    _entity, _table = "maintenance task", "maintenance_tasks"

    def add(self, t: MaintenanceTask) -> MaintenanceTask:
        with self._db.transaction() as conn:
            _require_appliance(conn, t.user_id, t.appliance_id)
            conn.execute(
                "INSERT INTO maintenance_tasks VALUES (?,?,?,?,?,?,?,?,?)",
                (t.id, t.user_id, t.appliance_id, t.title, _d(t.next_due_date), t.interval_months,
                 _d(t.last_done_date), t.notes, t.created_at.isoformat()),
            )
        return t

    def get(self, user_id: str, task_id: str) -> MaintenanceTask:
        return _task(self._fetch_one(
            "SELECT * FROM maintenance_tasks WHERE id = ? AND user_id = ?", (task_id, user_id),
            f"maintenance task {task_id}"))

    def list_for_user(self, user_id: str, appliance_id: str | None = None) -> Sequence[MaintenanceTask]:
        sql, params = "SELECT * FROM maintenance_tasks WHERE user_id = ?", [user_id]
        if appliance_id:
            sql, params = sql + " AND appliance_id = ?", params + [appliance_id]
        # completed one-off tasks (NULL due date) sort last
        sql += " ORDER BY next_due_date IS NULL, next_due_date, title"
        return [_task(r) for r in self._fetch_all(sql, tuple(params))]

    def update(self, t: MaintenanceTask) -> MaintenanceTask:
        with self._db.transaction() as conn:
            _require_appliance(conn, t.user_id, t.appliance_id)
            cur = conn.execute(
                "UPDATE maintenance_tasks SET appliance_id=?, title=?, next_due_date=?,"
                " interval_months=?, last_done_date=?, notes=? WHERE id=? AND user_id=?",
                (t.appliance_id, t.title, _d(t.next_due_date), t.interval_months,
                 _d(t.last_done_date), t.notes, t.id, t.user_id),
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"maintenance task {t.id} not found")
        return t

    def delete(self, user_id: str, task_id: str) -> None:
        self._delete(user_id, task_id)


# ---- Repairs ------------------------------------------------------------------------------

def _repair(r: sqlite3.Row) -> RepairRecord:
    return RepairRecord(
        id=r["id"], user_id=r["user_id"], appliance_id=r["appliance_id"],
        reported_date=date.fromisoformat(r["reported_date"]),
        problem_description=r["problem_description"], resolution=r["resolution"],
        technician=r["technician"],
        parts_replaced=tuple(p for p in r["parts_replaced"].split(_PARTS_SEPARATOR) if p),
        cost=Decimal(r["cost"]), covered_by_warranty=_bool_or_none(r["covered_by_warranty"]),
        source_document_id=r["source_document_id"],
        created_at=datetime.fromisoformat(r["created_at"]),
    )


def _repair_values(x: RepairRecord) -> tuple:
    return (x.reported_date.isoformat(), x.problem_description, x.resolution, x.technician,
            _PARTS_SEPARATOR.join(x.parts_replaced), str(x.cost),
            None if x.covered_by_warranty is None else int(x.covered_by_warranty),
            x.source_document_id)


class SQLiteRepairRepository(_SQLiteRepository, RepairRepository):
    _entity, _table = "repair", "repairs"

    def add(self, x: RepairRecord) -> RepairRecord:
        with self._db.transaction() as conn:
            _require_appliance(conn, x.user_id, x.appliance_id)
            conn.execute(
                "INSERT INTO repairs (id, user_id, appliance_id, reported_date, problem_description,"
                " resolution, technician, parts_replaced, cost, covered_by_warranty,"
                " source_document_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (x.id, x.user_id, x.appliance_id, *_repair_values(x), x.created_at.isoformat()),
            )
        return x

    def get(self, user_id: str, repair_id: str) -> RepairRecord:
        return _repair(self._fetch_one(
            "SELECT * FROM repairs WHERE id = ? AND user_id = ?", (repair_id, user_id),
            f"repair {repair_id}"))

    def list_for_user(self, user_id: str, *, appliance_id: str | None = None,
                      since: date | None = None, until: date | None = None) -> Sequence[RepairRecord]:
        sql, params = "SELECT * FROM repairs WHERE user_id = ?", [user_id]
        if appliance_id:
            sql, params = sql + " AND appliance_id = ?", params + [appliance_id]
        if since:
            sql, params = sql + " AND reported_date >= ?", params + [since.isoformat()]
        if until:
            sql, params = sql + " AND reported_date <= ?", params + [until.isoformat()]
        return [_repair(r) for r in self._fetch_all(
            sql + " ORDER BY reported_date DESC, created_at DESC", tuple(params))]

    def update(self, x: RepairRecord) -> RepairRecord:
        with self._db.transaction() as conn:
            _require_appliance(conn, x.user_id, x.appliance_id)
            cur = conn.execute(
                "UPDATE repairs SET reported_date=?, problem_description=?, resolution=?,"
                " technician=?, parts_replaced=?, cost=?, covered_by_warranty=?,"
                " source_document_id=?, appliance_id=? WHERE id=? AND user_id=?",
                (*_repair_values(x), x.appliance_id, x.id, x.user_id),
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"repair {x.id} not found")
        return x

    def delete(self, user_id: str, repair_id: str) -> None:
        self._delete(user_id, repair_id)


# ---- Documents ----------------------------------------------------------------------------

def _document(r: sqlite3.Row) -> Document:
    return Document(
        id=r["id"], user_id=r["user_id"], appliance_id=r["appliance_id"],
        original_filename=r["original_filename"], doc_type=DocumentType(r["doc_type"]),
        sha256=r["sha256"], storage_path=r["storage_path"], size_bytes=r["size_bytes"],
        mime_type=r["mime_type"], index_status=IndexStatus(r["index_status"]),
        uploaded_at=datetime.fromisoformat(r["uploaded_at"]),
    )


class SQLiteDocumentRepository(_SQLiteRepository, DocumentRepository):
    _entity, _table = "document", "documents"

    def add(self, d: Document) -> Document:
        with self._db.transaction() as conn:
            if d.appliance_id:
                _require_appliance(conn, d.user_id, d.appliance_id)
            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (d.id, d.user_id, d.appliance_id, d.original_filename, d.doc_type.value, d.sha256,
                 d.storage_path, d.size_bytes, d.mime_type, d.index_status.value,
                 d.uploaded_at.isoformat()),
            )
        return d

    def get(self, user_id: str, document_id: str) -> Document:
        return _document(self._fetch_one(
            "SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id),
            f"document {document_id}"))

    def list_for_user(self, user_id: str, *, appliance_id: str | None = None,
                      doc_type: DocumentType | None = None) -> Sequence[Document]:
        sql, params = "SELECT * FROM documents WHERE user_id = ?", [user_id]
        if appliance_id:
            sql, params = sql + " AND appliance_id = ?", params + [appliance_id]
        if doc_type:
            sql, params = sql + " AND doc_type = ?", params + [DocumentType(doc_type).value]
        return [_document(r) for r in self._fetch_all(sql + " ORDER BY uploaded_at", tuple(params))]

    def find_by_hash(self, user_id: str, sha256: str) -> Sequence[Document]:
        return [_document(r) for r in self._fetch_all(
            "SELECT * FROM documents WHERE user_id = ? AND sha256 = ? ORDER BY uploaded_at",
            (user_id, sha256))]

    def update(self, d: Document) -> Document:
        with self._db.transaction() as conn:
            if d.appliance_id:
                _require_appliance(conn, d.user_id, d.appliance_id)
            cur = conn.execute(
                "UPDATE documents SET appliance_id=?, original_filename=?, doc_type=?, mime_type=?,"
                " index_status=? WHERE id=? AND user_id=?",
                (d.appliance_id, d.original_filename, d.doc_type.value, d.mime_type,
                 d.index_status.value, d.id, d.user_id),
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"document {d.id} not found")
        return d

    def delete(self, user_id: str, document_id: str) -> None:
        self._delete(user_id, document_id)


# ---- Chunks -------------------------------------------------------------------------------

def _chunk(r: sqlite3.Row) -> Chunk:
    return Chunk(
        id=r["id"], user_id=r["user_id"], document_id=r["document_id"],
        chunk_index=r["chunk_index"], page_number=r["page_number"], section=r["section"],
        text=r["text"],
    )


class SQLiteChunkRepository(_SQLiteRepository, ChunkRepository):
    """SQLite is the source of truth for chunk text; the vector database only holds embeddings,
    so the vector index can always be rebuilt from here."""
    _entity, _table = "chunk", "chunks"

    def add_many(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        with self._db.transaction() as conn:
            for user_id, document_id in {(c.user_id, c.document_id) for c in chunks}:
                owned = conn.execute(
                    "SELECT 1 FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id)
                ).fetchone()
                if owned is None:
                    raise NotFoundError(f"document {document_id} not found")
            try:
                conn.executemany(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                    [(c.id, c.user_id, c.document_id, c.chunk_index, c.page_number, c.section, c.text)
                     for c in chunks],
                )
            except sqlite3.IntegrityError as exc:
                raise StorageError(f"could not store chunks (duplicate id?): {exc}") from exc

    def list_for_user(self, user_id: str, document_ids: Sequence[str] | None = None) -> list[Chunk]:
        if document_ids is not None and len(document_ids) == 0:
            return []
        sql, params = "SELECT * FROM chunks WHERE user_id = ?", [user_id]
        if document_ids is not None:
            sql += f" AND document_id IN ({','.join('?' * len(document_ids))})"
            params += list(document_ids)
        return [_chunk(r) for r in self._fetch_all(sql + " ORDER BY document_id, chunk_index", tuple(params))]

    def list_all(self) -> list[Chunk]:
        return [_chunk(r) for r in self._fetch_all("SELECT * FROM chunks ORDER BY user_id, document_id, chunk_index", ())]

    def get_many(self, user_id: str, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self._fetch_all(
            f"SELECT * FROM chunks WHERE user_id = ? AND id IN ({marks})", (user_id, *chunk_ids))
        return {r["id"]: _chunk(r) for r in rows}

    def count_for_document(self, user_id: str, document_id: str) -> int:
        row = self._fetch_one(
            "SELECT COUNT(*) AS n FROM chunks WHERE user_id = ? AND document_id = ?",
            (user_id, document_id), "count")
        return int(row["n"])

    def delete_for_document(self, user_id: str, document_id: str) -> int:
        with self._db.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM chunks WHERE user_id = ? AND document_id = ?", (user_id, document_id))
        return cur.rowcount


# ---- Maintenance logs ---------------------------------------------------------------------

def _log(r: sqlite3.Row) -> MaintenanceLog:
    return MaintenanceLog(
        id=r["id"], user_id=r["user_id"], appliance_id=r["appliance_id"], task_id=r["task_id"],
        done_date=date.fromisoformat(r["done_date"]), description=r["description"],
        cost=Decimal(r["cost"]), technician=r["technician"], notes=r["notes"],
        created_at=datetime.fromisoformat(r["created_at"]),
    )


def _require_task(conn: sqlite3.Connection, user_id: str, task_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM maintenance_tasks WHERE id = ? AND user_id = ?", (task_id, user_id)).fetchone()
    if row is None:
        raise NotFoundError(f"maintenance task {task_id} not found")


class SQLiteMaintenanceLogRepository(_SQLiteRepository, MaintenanceLogRepository):
    _entity, _table = "maintenance log", "maintenance_logs"

    def add(self, x: MaintenanceLog) -> MaintenanceLog:
        with self._db.transaction() as conn:
            _require_appliance(conn, x.user_id, x.appliance_id)
            if x.task_id:
                _require_task(conn, x.user_id, x.task_id)
            conn.execute(
                "INSERT INTO maintenance_logs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (x.id, x.user_id, x.appliance_id, x.task_id, x.done_date.isoformat(), x.description,
                 str(x.cost), x.technician, x.notes, x.created_at.isoformat()))
        return x

    def get(self, user_id: str, log_id: str) -> MaintenanceLog:
        return _log(self._fetch_one(
            "SELECT * FROM maintenance_logs WHERE id = ? AND user_id = ?", (log_id, user_id),
            f"maintenance log {log_id}"))

    def list_for_user(self, user_id: str, *, appliance_id: str | None = None,
                      since: date | None = None, until: date | None = None) -> Sequence[MaintenanceLog]:
        sql, params = "SELECT * FROM maintenance_logs WHERE user_id = ?", [user_id]
        if appliance_id:
            sql, params = sql + " AND appliance_id = ?", params + [appliance_id]
        if since:
            sql, params = sql + " AND done_date >= ?", params + [since.isoformat()]
        if until:
            sql, params = sql + " AND done_date <= ?", params + [until.isoformat()]
        return [_log(r) for r in self._fetch_all(sql + " ORDER BY done_date DESC, created_at DESC", tuple(params))]

    def update(self, x: MaintenanceLog) -> MaintenanceLog:
        with self._db.transaction() as conn:
            _require_appliance(conn, x.user_id, x.appliance_id)
            if x.task_id:
                _require_task(conn, x.user_id, x.task_id)
            cur = conn.execute(
                "UPDATE maintenance_logs SET appliance_id=?, task_id=?, done_date=?, description=?, cost=?,"
                " technician=?, notes=? WHERE id=? AND user_id=?",
                (x.appliance_id, x.task_id, x.done_date.isoformat(), x.description, str(x.cost),
                 x.technician, x.notes, x.id, x.user_id))
        if cur.rowcount == 0:
            raise NotFoundError(f"maintenance log {x.id} not found")
        return x

    def delete(self, user_id: str, log_id: str) -> None:
        self._delete(user_id, log_id)


# ---- Extraction drafts --------------------------------------------------------------------
import json as _json


def _draft(r: sqlite3.Row) -> ExtractionDraft:
    return ExtractionDraft(
        id=r["id"], user_id=r["user_id"], document_id=r["document_id"], doc_type=DocumentType(r["doc_type"]),
        status=DraftStatus(r["status"]), fields=fields_from_json(r["fields_json"]),
        final_values=_json.loads(r["final_json"]) if r["final_json"] else None,
        created_at=datetime.fromisoformat(r["created_at"]),
        resolved_at=datetime.fromisoformat(r["resolved_at"]) if r["resolved_at"] else None)


class SQLiteExtractionDraftRepository(_SQLiteRepository, ExtractionDraftRepository):
    _entity, _table = "extraction draft", "extraction_drafts"

    def add(self, d: ExtractionDraft) -> ExtractionDraft:
        with self._db.transaction() as conn:
            owned = conn.execute("SELECT 1 FROM documents WHERE id = ? AND user_id = ?", (d.document_id, d.user_id)).fetchone()
            if owned is None:
                raise NotFoundError(f"document {d.document_id} not found")
            conn.execute("INSERT INTO extraction_drafts VALUES (?,?,?,?,?,?,?,?,?)",
                         (d.id, d.user_id, d.document_id, d.doc_type.value, d.status.value, fields_to_json(d.fields),
                          _json.dumps(d.final_values, ensure_ascii=False) if d.final_values is not None else None,
                          d.created_at.isoformat(), d.resolved_at.isoformat() if d.resolved_at else None))
        return d

    def get(self, user_id: str, draft_id: str) -> ExtractionDraft:
        return _draft(self._fetch_one("SELECT * FROM extraction_drafts WHERE id = ? AND user_id = ?",
                                      (draft_id, user_id), f"extraction draft {draft_id}"))

    def list_for_user(self, user_id: str, *, status: DraftStatus | None = None,
                      document_id: str | None = None) -> Sequence[ExtractionDraft]:
        sql, params = "SELECT * FROM extraction_drafts WHERE user_id = ?", [user_id]
        if status:
            sql, params = sql + " AND status = ?", params + [DraftStatus(status).value]
        if document_id:
            sql, params = sql + " AND document_id = ?", params + [document_id]
        return [_draft(r) for r in self._fetch_all(sql + " ORDER BY created_at, id", tuple(params))]

    def update(self, d: ExtractionDraft) -> ExtractionDraft:
        with self._db.transaction() as conn:
            cur = conn.execute(
                "UPDATE extraction_drafts SET status=?, fields_json=?, final_json=?, resolved_at=? WHERE id=? AND user_id=?",
                (d.status.value, fields_to_json(d.fields), _json.dumps(d.final_values, ensure_ascii=False) if d.final_values is not None else None,
                 d.resolved_at.isoformat() if d.resolved_at else None, d.id, d.user_id))
        if cur.rowcount == 0:
            raise NotFoundError(f"extraction draft {d.id} not found")
        return d
