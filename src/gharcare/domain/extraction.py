"""Extraction drafts: what the system READ from a document, awaiting the user's review.

Nothing here touches the user's real records. A draft becomes warranties / repairs / appliance updates only
when the user confirms it (see gharcare.extraction.service)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from gharcare.domain.enums import DocumentType
from gharcare.domain.models import new_id, utcnow
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError


class FieldConfidence(str, Enum):
    HIGH = "high"      # unambiguous, labeled in the document (or two independent readers agree)
    MEDIUM = "medium"  # plausible but needs a look (unlabeled, ambiguous date order, only one reader)
    LOW = "low"        # missing, conflicting or contradicting readings


class DraftStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


EXTRACTABLE_TYPES = (DocumentType.RECEIPT, DocumentType.WARRANTY_CARD, DocumentType.SERVICE_INVOICE)


@dataclass(frozen=True)
class ExtractedField:
    value: str | None = None                 # normalised: dates ISO, money '95000.00', months '24'
    confidence: FieldConfidence = FieldConfidence.LOW
    source: str = "none"                     # rules | llm | both | none
    evidence: str | None = None              # the document line the value came from
    note: str | None = None                  # why it is uncertain, in plain words
    alternatives: tuple[str, ...] = ()       # other readings found (shown to the user to choose from)

    def to_dict(self) -> dict:
        return {"value": self.value, "confidence": self.confidence.value, "source": self.source,
                "evidence": self.evidence, "note": self.note, "alternatives": list(self.alternatives)}

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedField":
        return cls(value=data.get("value"), confidence=FieldConfidence(data.get("confidence", "low")),
                   source=data.get("source", "none"), evidence=data.get("evidence"), note=data.get("note"),
                   alternatives=tuple(data.get("alternatives", ())))


def fields_to_json(fields: dict[str, ExtractedField]) -> str:
    return json.dumps({k: v.to_dict() for k, v in fields.items()}, ensure_ascii=False)


def fields_from_json(text: str) -> dict[str, ExtractedField]:
    return {k: ExtractedField.from_dict(v) for k, v in json.loads(text).items()}


@dataclass(frozen=True)
class ExtractionDraft:
    user_id: str
    document_id: str
    doc_type: DocumentType
    fields: dict[str, ExtractedField]
    status: DraftStatus = DraftStatus.PENDING
    final_values: dict[str, str | None] | None = None     # what the user confirmed (set on confirmation)
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", validate_identifier(self.user_id, "user_id"))
        object.__setattr__(self, "document_id", validate_identifier(self.document_id, "document_id"))
        object.__setattr__(self, "id", validate_identifier(self.id, "id"))
        try:
            doc_type = DocumentType(self.doc_type)
        except ValueError as exc:
            raise DomainValidationError(str(exc)) from exc
        if doc_type not in EXTRACTABLE_TYPES:
            raise DomainValidationError(f"details cannot be extracted from a document of type {doc_type.value!r}")
        object.__setattr__(self, "doc_type", doc_type)
        object.__setattr__(self, "status", DraftStatus(self.status))
