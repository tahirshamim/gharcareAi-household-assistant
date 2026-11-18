"""Extraction review workflow: propose -> (user reviews/corrects) -> confirm | reject.

Nothing reaches the user's real records until confirm(). Confirmation:
  * requires the user to review every uncertain (non-HIGH) field that has a value,
  * validates every value and the minimum fields for the document type,
  * creates the warranty / repair / appliance changes in ONE transaction (all or nothing),
  * stores both the AI draft and the user's final values, so accuracy can be MEASURED (see accuracy()).
final_values is the COMPLETE set of values the user approves; a missing key means 'blank'."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Collection, Mapping

from gharcare.domain import (
    EXTRACTABLE_TYPES, Appliance, DocumentType, DraftStatus, ExtractionDraft, FieldConfidence, RepairRecord, Warranty,
)
from gharcare.domain.models import utcnow
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError, NoExtractableTextError
from gharcare.extraction.pipeline import ExtractionPipeline
from gharcare.extraction.schemas import REQUIRED_ALL, REQUIRED_ANY, SCHEMAS, normalize_value, spec_map
from gharcare.ingestion.extractors import TextExtractor, default_extractors
from gharcare.repositories.base import (
    ApplianceRepository, DocumentRepository, ExtractionDraftRepository, RepairRepository, TransactionManager,
    WarrantyRepository,
)
from gharcare.storage.base import ArtifactStore


@dataclass(frozen=True)
class ConfirmResult:
    draft: ExtractionDraft
    created: dict[str, str]          # e.g. {"warranty": "<id>"} or {"repair": "<id>"}
    changes: tuple[str, ...]         # human-readable appliance changes (e.g. "purchase date: 2023-01-01 -> 2024-03-14")


@dataclass(frozen=True)
class ExtractionAccuracy:
    drafts: int
    fields_read: int                 # the AI draft had a value
    fields_correct: int              # the user kept it unchanged
    fields_missed: int               # the draft was empty but the user supplied a value
    by_confidence: dict[str, tuple[int, int]]   # confidence -> (correct, read): HIGH should be ~100%
    by_field: dict[str, tuple[int, int]]

    @property
    def accuracy(self) -> float | None:
        return self.fields_correct / self.fields_read if self.fields_read else None


class ExtractionService:
    def __init__(self, documents: DocumentRepository, appliances: ApplianceRepository, warranties: WarrantyRepository,
                 repairs: RepairRepository, drafts: ExtractionDraftRepository, artifacts: ArtifactStore,
                 tx: TransactionManager, pipeline: ExtractionPipeline,
                 extractors: Mapping[str, TextExtractor] | None = None) -> None:
        self._documents, self._appliances, self._warranties, self._repairs = documents, appliances, warranties, repairs
        self._drafts, self._artifacts, self._tx, self._pipeline = drafts, artifacts, tx, pipeline
        self._extractors = dict(extractors) if extractors is not None else default_extractors()

    # ---- propose ---------------------------------------------------------------------------
    def propose(self, user_id: str, document_id: str) -> ExtractionDraft:
        """Reads the document and stores a PENDING draft. Idempotent: an existing pending draft is returned."""
        validate_identifier(user_id, "user_id")
        document = self._documents.get(user_id, document_id)
        if document.doc_type not in EXTRACTABLE_TYPES:
            raise DomainValidationError(f"details can only be extracted from receipts, warranty cards and service invoices, not {document.doc_type.value}")
        pending = self._drafts.list_for_user(user_id, status=DraftStatus.PENDING, document_id=document.id)
        if pending:
            return pending[0]
        suffix = Path(document.original_filename).suffix.lower()
        if suffix not in self._extractors:
            raise DomainValidationError(f"cannot read files of type {suffix!r}")
        pages = self._extractors[suffix].extract(self._artifacts.read(user_id, document.storage_path))
        text = "\n\n".join(p.text for p in pages if p.text.strip())
        if not text.strip():
            raise NoExtractableTextError("No readable text was found (scanned images are not supported yet). Please enter the details manually.")
        draft = ExtractionDraft(user_id=user_id, document_id=document.id, doc_type=document.doc_type,   # validates the type
                                fields=self._pipeline.run(document.doc_type, text))
        self._drafts.add(draft)
        self._artifacts.save_extracted(user_id, f"{document.id}-fields", {
            "kind": "ai_draft", "draft_id": draft.id, "fields": {k: v.to_dict() for k, v in draft.fields.items()}})
        return draft

    def get(self, user_id: str, draft_id: str) -> ExtractionDraft:
        return self._drafts.get(user_id, draft_id)

    def list_pending(self, user_id: str) -> list[ExtractionDraft]:
        return list(self._drafts.list_for_user(user_id, status=DraftStatus.PENDING))

    # ---- confirm / reject ------------------------------------------------------------------
    def confirm(self, user_id: str, draft_id: str, final_values: Mapping[str, object], *,
                reviewed_fields: Collection[str] = (), appliance_id: str | None = None) -> ConfirmResult:
        validate_identifier(user_id, "user_id")
        draft = self._drafts.get(user_id, draft_id)
        if draft.status != DraftStatus.PENDING:
            raise DomainValidationError(f"this draft was already {draft.status.value}")
        specs = spec_map(draft.doc_type)
        unknown = sorted(set(final_values) - set(specs))
        if unknown:
            raise DomainValidationError(f"unknown field(s): {', '.join(unknown)}")

        values: dict[str, str | None] = {}
        errors: list[str] = []
        for name, spec in specs.items():
            try:
                values[name] = normalize_value(spec, final_values.get(name))
            except DomainValidationError as exc:
                errors.append(str(exc))
        if errors:
            raise DomainValidationError("; ".join(errors))

        reviewed = set(reviewed_fields)
        unreviewed = [specs[n].label for n, f in draft.fields.items()
                      if f.value is not None and f.confidence != FieldConfidence.HIGH and values.get(n) == f.value and n not in reviewed]
        if unreviewed:
            raise DomainValidationError("please review these uncertain fields before confirming: " + ", ".join(unreviewed))
        missing = [specs[n].label for n in REQUIRED_ALL.get(draft.doc_type, ()) if not values.get(n)]
        any_group = REQUIRED_ANY.get(draft.doc_type)
        if any_group and not any(values.get(n) for n in any_group):
            missing.append(" or ".join(specs[n].label for n in any_group))
        if missing:
            raise DomainValidationError("missing required information: " + "; ".join(missing))

        document = self._documents.get(user_id, draft.document_id)
        created: dict[str, str] = {}
        with self._tx.transaction():
            appliance = self._resolve_appliance(user_id, document.appliance_id, appliance_id, values)
            appliance, changes = self._enrich(appliance, values, draft.doc_type)
            if document.appliance_id != appliance.id:
                self._documents.update(replace(document, appliance_id=appliance.id))
            if draft.doc_type == DocumentType.WARRANTY_CARD:
                created["warranty"] = self._warranties.add(Warranty(
                    user_id=user_id, appliance_id=appliance.id, start_date=date.fromisoformat(values["warranty_start_date"]),
                    duration_months=int(values["warranty_duration_months"]), provider=values.get("provider"),
                    source_document_id=document.id, verified=True)).id
            elif draft.doc_type == DocumentType.SERVICE_INVOICE:
                parts = tuple(p.strip() for p in (values.get("parts_replaced") or "").replace(";", ",").split(",") if p.strip())
                created["repair"] = self._repairs.add(RepairRecord(
                    user_id=user_id, appliance_id=appliance.id, reported_date=date.fromisoformat(values["service_date"]),
                    problem_description=values.get("problem_description") or values["work_done"], resolution=values.get("work_done"),
                    technician=values.get("technician"), parts_replaced=parts, cost=Decimal(values["total_amount"]) if values.get("total_amount") else Decimal("0"),
                    source_document_id=document.id)).id
            resolved = self._drafts.update(replace(draft, status=DraftStatus.CONFIRMED, final_values=values, resolved_at=utcnow()))
            self._artifacts.save_extracted(user_id, f"{document.id}-fields", {
                "kind": "user_confirmed", "draft_id": draft.id, "user_corrected": True,
                "ai_draft": {k: v.value for k, v in draft.fields.items()}, "final": values, "created": created})
        return ConfirmResult(resolved, created, tuple(changes))

    def reject(self, user_id: str, draft_id: str) -> ExtractionDraft:
        validate_identifier(user_id, "user_id")
        draft = self._drafts.get(user_id, draft_id)
        if draft.status != DraftStatus.PENDING:
            raise DomainValidationError(f"this draft was already {draft.status.value}")
        return self._drafts.update(replace(draft, status=DraftStatus.REJECTED, resolved_at=utcnow()))

    # ---- measurement -----------------------------------------------------------------------
    def accuracy(self, user_id: str) -> ExtractionAccuracy:
        """Field-level accuracy of the AI drafts, measured against what users finally confirmed."""
        validate_identifier(user_id, "user_id")
        confirmed = self._drafts.list_for_user(user_id, status=DraftStatus.CONFIRMED)
        read = correct = missed = 0
        by_conf: dict[str, list[int]] = {}
        by_field: dict[str, list[int]] = {}
        for d in confirmed:
            for name, f in d.fields.items():
                final = (d.final_values or {}).get(name)
                if f.value is None:
                    missed += final is not None
                    continue
                ok = final == f.value
                read, correct = read + 1, correct + ok
                for bucket, key in ((by_conf, f.confidence.value), (by_field, name)):
                    cell = bucket.setdefault(key, [0, 0])
                    cell[0] += ok
                    cell[1] += 1
        return ExtractionAccuracy(len(confirmed), read, correct, missed,
                                  {k: (v[0], v[1]) for k, v in by_conf.items()}, {k: (v[0], v[1]) for k, v in by_field.items()})

    # ---- helpers ---------------------------------------------------------------------------
    def _resolve_appliance(self, user_id: str, linked_id: str | None, chosen_id: str | None, values: dict) -> Appliance:
        target = chosen_id or linked_id
        if target:
            return self._appliances.get(user_id, target)
        existing = self._find_existing(user_id, values)
        if existing:
            return existing
        brand, product = values.get("brand"), values.get("product_name")
        if product and brand and product.casefold().startswith(brand.casefold()):
            product = product[len(brand):].strip(" -") or product
        name = product or " ".join(p for p in (brand, values.get("model_number")) if p)
        if not name:
            raise DomainValidationError("please choose which appliance this document belongs to")
        return self._appliances.add(Appliance(user_id=user_id, name=name, brand=brand, model=values.get("model_number"),
                                              serial_number=values.get("serial_number")))

    def _find_existing(self, user_id: str, values: dict) -> Appliance | None:
        """Match by serial number (exact). Only if the document has NO serial: by brand+model when exactly one appliance
        matches. A document WITH a serial never matches by model, so two identical ACs with different serials stay separate."""
        appliances = self._appliances.list_for_user(user_id)
        serial = (values.get("serial_number") or "").casefold()
        if serial:
            return next((a for a in appliances if (a.serial_number or "").casefold() == serial), None)
        brand, model = (values.get("brand") or "").casefold(), (values.get("model_number") or "").casefold()
        if brand and model:
            matches = [a for a in appliances if (a.brand or "").casefold() == brand and (a.model or "").casefold() == model]
            if len(matches) > 1:
                raise DomainValidationError("more than one of your appliances matches this document; please choose which one")
            return matches[0] if matches else None
        return None

    def _enrich(self, appliance: Appliance, values: dict, doc_type: DocumentType) -> tuple[Appliance, list[str]]:
        """Fills blanks from the document; for receipts also sets purchase date/price (reporting any overwrite)."""
        changes: list[str] = []
        update = {"brand": appliance.brand or values.get("brand"), "model": appliance.model or values.get("model_number"),
                  "serial_number": appliance.serial_number or values.get("serial_number")}
        if doc_type == DocumentType.RECEIPT:
            new_date = date.fromisoformat(values["purchase_date"]) if values.get("purchase_date") else None
            new_price = Decimal(values["total_amount"]) if values.get("total_amount") else None
            for label, old, new, key in (("purchase date", appliance.purchase_date, new_date, "purchase_date"),
                                         ("purchase price", appliance.purchase_price, new_price, "purchase_price")):
                if new is not None and new != old:
                    changes.append(f"{label}: {old if old is not None else 'not set'} -> {new}")
                    update[key] = new
        changed = replace(appliance, **update)
        if changed != appliance:
            self._appliances.update(changed)
        return changed, changes
