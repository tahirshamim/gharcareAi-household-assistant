from enum import Enum


class DocumentType(str, Enum):
    MANUAL = "manual"
    RECEIPT = "receipt"
    WARRANTY_CARD = "warranty_card"
    SERVICE_INVOICE = "service_invoice"
    OTHER = "other"


class IndexStatus(str, Enum):
    """Whether a document's text has been chunked and embedded (used from Step 2)."""
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class WarrantyStatus(str, Enum):
    ACTIVE = "active"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"


class TaskStatus(str, Enum):
    OVERDUE = "overdue"
    DUE_SOON = "due_soon"
    UPCOMING = "upcoming"
    COMPLETED = "completed"
