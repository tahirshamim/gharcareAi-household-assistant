"""Which fields each extractable document type has, and how a value is normalised/validated."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from gharcare.domain import DocumentType
from gharcare.domain.validation import clean_text, to_money
from gharcare.errors import DomainValidationError


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str   # text | date | money | months


SCHEMAS: dict[DocumentType, tuple[FieldSpec, ...]] = {
    DocumentType.RECEIPT: (
        FieldSpec("store_name", "Store", "text"), FieldSpec("purchase_date", "Purchase date", "date"),
        FieldSpec("total_amount", "Total amount (PKR)", "money"), FieldSpec("product_name", "Product", "text"),
        FieldSpec("brand", "Brand", "text"), FieldSpec("model_number", "Model number", "text"),
        FieldSpec("serial_number", "Serial number", "text")),
    DocumentType.WARRANTY_CARD: (
        FieldSpec("product_name", "Product", "text"), FieldSpec("brand", "Brand", "text"),
        FieldSpec("model_number", "Model number", "text"), FieldSpec("serial_number", "Serial number", "text"),
        FieldSpec("warranty_start_date", "Warranty start date", "date"),
        FieldSpec("warranty_duration_months", "Warranty duration (months)", "months"),
        FieldSpec("provider", "Warranty provider", "text")),
    DocumentType.SERVICE_INVOICE: (
        FieldSpec("service_date", "Service date", "date"), FieldSpec("total_amount", "Total amount (PKR)", "money"),
        FieldSpec("problem_description", "Problem", "text"), FieldSpec("work_done", "Work done", "text"),
        FieldSpec("technician", "Technician", "text"), FieldSpec("parts_replaced", "Parts replaced", "text")),
}

# The minimum a user must end up with for the saved record to be useful (checked on confirmation).
REQUIRED_ALL = {DocumentType.WARRANTY_CARD: ("warranty_start_date", "warranty_duration_months"),
                DocumentType.SERVICE_INVOICE: ("service_date",)}
REQUIRED_ANY = {DocumentType.RECEIPT: ("purchase_date", "total_amount"),
                DocumentType.SERVICE_INVOICE: ("problem_description", "work_done")}


def spec_map(doc_type: DocumentType) -> dict[str, FieldSpec]:
    return {s.name: s for s in SCHEMAS[doc_type]}


def normalize_value(spec: FieldSpec, raw: object) -> str | None:
    """Canonical string form; raises DomainValidationError with a message a user can act on."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    text = str(raw).strip()
    try:
        if spec.kind == "date":
            return date.fromisoformat(text).isoformat()
        if spec.kind == "money":
            return f"{to_money(text.replace(',', ''), spec.label):.2f}"
        if spec.kind == "months":
            if not text.isdigit() or not 1 <= int(text) <= 120:
                raise ValueError
            return str(int(text))
        return clean_text(text, spec.label, max_length=300)
    except (ValueError, DomainValidationError) as exc:
        hint = {"date": "a date like 2024-03-14", "money": "an amount like 95000.00",
                "months": "a whole number of months (1-120)"}.get(spec.kind, "valid text")
        raise DomainValidationError(f"{spec.label} must be {hint}") from exc
