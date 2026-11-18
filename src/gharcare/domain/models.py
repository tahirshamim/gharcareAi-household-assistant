"""Domain entities.

Design choices:
* Frozen dataclasses: an entity is never mutated in place; updates produce a new object
  (dataclasses.replace). This makes bugs and race conditions much easier to reason about.
* Validation happens in __post_init__, so an invalid entity cannot exist.
* Business rules that are pure arithmetic (warranty expiry, overdue status) live HERE in plain
  Python, never in prompts, so they are deterministic and unit-testable.
* All dates are calendar dates (no timezone ambiguity). Money is Decimal in PKR.
"""
from __future__ import annotations

import calendar
import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from gharcare.domain.enums import DocumentType, IndexStatus, TaskStatus, WarrantyStatus
from gharcare.domain.validation import clean_text, to_money, validate_identifier
from gharcare.errors import DomainValidationError


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def add_months(d: date, months: int) -> date:
    """Calendar-aware month arithmetic. Day is clamped to the target month's length
    (31 Jan + 1 month = 28/29 Feb)."""
    total = d.year * 12 + (d.month - 1) + months
    year, month0 = divmod(total, 12)
    month = month0 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _set(obj: object, **values: object) -> None:
    """Assign to a frozen dataclass during __post_init__ (normalisation only)."""
    for name, value in values.items():
        object.__setattr__(obj, name, value)


def _whole_months(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 120:
        raise DomainValidationError(f"{field_name} must be a whole number between 1 and 120")
    return value


@dataclass(frozen=True)
class Appliance:
    user_id: str
    name: str
    brand: str | None = None
    model: str | None = None
    serial_number: str | None = None
    purchase_date: date | None = None
    purchase_price: Decimal | None = None
    location: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        _set(
            self,
            user_id=validate_identifier(self.user_id, "user_id"),
            id=validate_identifier(self.id, "id"),
            name=clean_text(self.name, "name", required=True, max_length=120),
            brand=clean_text(self.brand, "brand", max_length=80),
            model=clean_text(self.model, "model", max_length=80),
            serial_number=clean_text(self.serial_number, "serial_number", max_length=80),
            location=clean_text(self.location, "location", max_length=80),
            purchase_price=to_money(self.purchase_price, "purchase_price"),
        )
        if self.purchase_date and self.purchase_date > date.today() + timedelta(days=1):
            raise DomainValidationError("purchase_date cannot be in the future")

    @property
    def display_name(self) -> str:
        return " ".join(p for p in (self.brand, self.name) if p)


@dataclass(frozen=True)
class Warranty:
    """A warranty is valid THROUGH its expiry date (inclusive)."""
    user_id: str
    appliance_id: str
    start_date: date
    duration_months: int
    provider: str | None = None
    coverage_notes: str | None = None
    source_document_id: str | None = None
    verified: bool = False  # True once the user has confirmed the (possibly AI-extracted) values
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        _set(
            self,
            duration_months=_whole_months(self.duration_months, "duration_months"),
            user_id=validate_identifier(self.user_id, "user_id"),
            id=validate_identifier(self.id, "id"),
            appliance_id=validate_identifier(self.appliance_id, "appliance_id"),
            provider=clean_text(self.provider, "provider", max_length=120),
            coverage_notes=clean_text(self.coverage_notes, "coverage_notes", max_length=2000),
        )

    @property
    def expiry_date(self) -> date:
        return add_months(self.start_date, self.duration_months)

    def days_remaining(self, today: date) -> int:
        return (self.expiry_date - today).days

    def status(self, today: date, soon_days: int = 30) -> WarrantyStatus:
        remaining = self.days_remaining(today)
        if remaining < 0:
            return WarrantyStatus.EXPIRED
        if remaining <= soon_days:
            return WarrantyStatus.EXPIRING_SOON
        return WarrantyStatus.ACTIVE


@dataclass(frozen=True)
class MaintenanceTask:
    """Recurring (interval_months set) or one-off (interval_months None).
    next_due_date is None only for a completed one-off task."""
    user_id: str
    appliance_id: str
    title: str
    next_due_date: date | None
    interval_months: int | None = None
    last_done_date: date | None = None
    notes: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        _set(
            self,
            user_id=validate_identifier(self.user_id, "user_id"),
            id=validate_identifier(self.id, "id"),
            appliance_id=validate_identifier(self.appliance_id, "appliance_id"),
            title=clean_text(self.title, "title", required=True, max_length=160),
            notes=clean_text(self.notes, "notes", max_length=2000),
        )
        if self.interval_months is not None:
            _whole_months(self.interval_months, "interval_months")
        if self.next_due_date is None and (self.interval_months is not None or self.last_done_date is None):
            raise DomainValidationError(
                "next_due_date may be empty only for a completed one-off task"
            )

    def status(self, today: date, soon_days: int = 14) -> TaskStatus:
        if self.next_due_date is None:
            return TaskStatus.COMPLETED
        days_until = (self.next_due_date - today).days
        if days_until < 0:
            return TaskStatus.OVERDUE
        if days_until <= soon_days:
            return TaskStatus.DUE_SOON
        return TaskStatus.UPCOMING

    def mark_done(self, done_on: date) -> "MaintenanceTask":
        """Recurring tasks reschedule from the date the work was actually done."""
        if self.interval_months is None:
            return replace(self, last_done_date=done_on, next_due_date=None)
        return replace(
            self, last_done_date=done_on, next_due_date=add_months(done_on, self.interval_months)
        )


@dataclass(frozen=True)
class RepairRecord:
    user_id: str
    appliance_id: str
    reported_date: date
    problem_description: str
    resolution: str | None = None
    technician: str | None = None
    parts_replaced: tuple[str, ...] = ()
    cost: Decimal = Decimal("0.00")
    covered_by_warranty: bool | None = None  # None = unknown; never guessed
    source_document_id: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        parts = tuple(p for p in (clean_text(p, "parts_replaced item", max_length=120)
                                  for p in self.parts_replaced) if p)
        _set(
            self,
            user_id=validate_identifier(self.user_id, "user_id"),
            id=validate_identifier(self.id, "id"),
            appliance_id=validate_identifier(self.appliance_id, "appliance_id"),
            problem_description=clean_text(self.problem_description, "problem_description",
                                           required=True, max_length=2000),
            resolution=clean_text(self.resolution, "resolution", max_length=2000),
            technician=clean_text(self.technician, "technician", max_length=120),
            parts_replaced=parts,
            cost=to_money(0 if self.cost is None else self.cost, "cost"),
        )


@dataclass(frozen=True)
class Document:
    """Metadata row for an uploaded file. The file itself lives in the ArtifactStore;
    this row links it to a user/appliance and tracks indexing state.
    The same file may be linked to several appliances (e.g. two identical Gree ACs):
    rows are links, the stored artifact is shared."""
    user_id: str
    original_filename: str
    doc_type: DocumentType
    sha256: str
    storage_path: str
    size_bytes: int
    mime_type: str | None = None
    appliance_id: str | None = None
    index_status: IndexStatus = IndexStatus.PENDING
    id: str = field(default_factory=new_id)
    uploaded_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise DomainValidationError("size_bytes cannot be negative")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise DomainValidationError("sha256 must be 64 lowercase hex characters")
        try:
            doc_type, index_status = DocumentType(self.doc_type), IndexStatus(self.index_status)
        except ValueError as exc:
            raise DomainValidationError(str(exc)) from exc
        _set(
            self,
            user_id=validate_identifier(self.user_id, "user_id"),
            id=validate_identifier(self.id, "id"),
            original_filename=clean_text(self.original_filename, "original_filename",
                                         required=True, max_length=255),
            storage_path=clean_text(self.storage_path, "storage_path", required=True, max_length=500),
            mime_type=clean_text(self.mime_type, "mime_type", max_length=100),
            doc_type=doc_type,
            index_status=index_status,
            appliance_id=(validate_identifier(self.appliance_id, "appliance_id")
                          if self.appliance_id is not None else None),
        )


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of a document. Never crosses a page boundary, so `page_number`
    is exact and citations are trustworthy. The default id is deterministic
    (`{document_id}-{chunk_index}`), which makes re-indexing idempotent."""
    user_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int | None = None
    section: str | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_index, int) or isinstance(self.chunk_index, bool) or self.chunk_index < 0:
            raise DomainValidationError("chunk_index must be a whole number >= 0")
        if self.page_number is not None and (
            not isinstance(self.page_number, int) or isinstance(self.page_number, bool) or self.page_number < 1
        ):
            raise DomainValidationError("page_number must be a whole number >= 1")
        document_id = validate_identifier(self.document_id, "document_id")
        _set(
            self,
            user_id=validate_identifier(self.user_id, "user_id"),
            document_id=document_id,
            id=validate_identifier(self.id or f"{document_id}-{self.chunk_index}", "id"),
            text=clean_text(self.text, "text", required=True, max_length=20000),
            section=clean_text(self.section, "section", max_length=200),
        )


@dataclass(frozen=True)
class MaintenanceLog:
    """A completed service/maintenance event (what was done, when, and what it cost).
    Separate from MaintenanceTask (the schedule): tasks say what is DUE, logs record what HAPPENED."""
    user_id: str
    appliance_id: str
    done_date: date
    description: str
    cost: Decimal = Decimal("0.00")
    technician: str | None = None
    task_id: str | None = None
    notes: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        _set(
            self,
            user_id=validate_identifier(self.user_id, "user_id"),
            id=validate_identifier(self.id, "id"),
            appliance_id=validate_identifier(self.appliance_id, "appliance_id"),
            task_id=validate_identifier(self.task_id, "task_id") if self.task_id is not None else None,
            description=clean_text(self.description, "description", required=True, max_length=200),
            technician=clean_text(self.technician, "technician", max_length=120),
            notes=clean_text(self.notes, "notes", max_length=2000),
            cost=to_money(0 if self.cost is None else self.cost, "cost"),
        )
