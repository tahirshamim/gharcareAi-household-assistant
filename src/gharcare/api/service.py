"""AppService: everything the web API needs, as plain methods taking and returning plain data (dicts / lists).

Why a separate layer: the HTTP framework stays a thin shell (routing, validation, auth), while all behaviour is
testable without a web server. Every method takes `user_id` explicitly, so multi-user support is an auth change,
not a rewrite. Internal details (storage paths, hashes) are never returned."""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from datetime import date
from decimal import Decimal
from typing import Callable, Mapping

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance, Document, DraftStatus, MaintenanceTask, RepairRecord, Warranty
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError, GharCareError, LLMError, NotFoundError
from gharcare.extraction.factory import build_extraction
from gharcare.extraction.schemas import SCHEMAS
from gharcare.llm import LLMProvider, UnavailableLLMProvider, build_llm
from gharcare.rag import build_rag, rebuild_vector_index
from gharcare.services import ExpenseCategory, build_services
from gharcare.workflow import HouseholdAssistant

_log = logging.getLogger(__name__)
MAX_ASK_CHARS = 2000


def _date(value: object) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise DomainValidationError(f"'{value}' is not a valid date (use YYYY-MM-DD)") from exc


def _money(value: Decimal | None) -> str | None:
    return None if value is None else f"{value:.2f}"


# ---- JSON mappers (explicit, so the API surface is stable and leaks nothing internal) ----------
def appliance_json(a: Appliance) -> dict:
    return {"id": a.id, "name": a.name, "brand": a.brand, "model": a.model, "serial_number": a.serial_number,
            "purchase_date": a.purchase_date.isoformat() if a.purchase_date else None, "purchase_price": _money(a.purchase_price),
            "location": a.location, "display_name": a.display_name}


def document_json(d: Document) -> dict:
    return {"id": d.id, "filename": d.original_filename, "doc_type": d.doc_type.value, "appliance_id": d.appliance_id,
            "index_status": d.index_status.value, "size_bytes": d.size_bytes, "uploaded_at": d.uploaded_at.isoformat()}


def repair_json(r: RepairRecord) -> dict:
    return {"id": r.id, "appliance_id": r.appliance_id, "reported_date": r.reported_date.isoformat(), "problem_description": r.problem_description,
            "resolution": r.resolution, "technician": r.technician, "parts_replaced": list(r.parts_replaced), "cost": _money(r.cost),
            "covered_by_warranty": r.covered_by_warranty, "source_document_id": r.source_document_id}


def draft_json(d) -> dict:
    fields = []
    for spec in SCHEMAS[d.doc_type]:
        f = d.fields[spec.name]
        fields.append({"name": spec.name, "label": spec.label, "kind": spec.kind, "value": f.value, "confidence": f.confidence.value,
                       "source": f.source, "evidence": f.evidence, "note": f.note, "alternatives": list(f.alternatives)})
    return {"id": d.id, "document_id": d.document_id, "doc_type": d.doc_type.value, "status": d.status.value, "fields": fields,
            "final_values": d.final_values}


_RECORDS_HEAD, _SOURCES_HEAD = "\n\nFrom your records:\n", "\n\nSources:\n"


def split_answer(answer: str) -> dict:
    """Splits the assistant's answer text into its parts for display: the answer itself, the lines taken from the
    user's records, and the numbered sources. The assistant appends these sections in this order, always with
    these headings, so the split is exact; anything else stays in `body`."""
    body, sources, records = answer, [], []
    if _SOURCES_HEAD in body:
        body, tail = body.rsplit(_SOURCES_HEAD, 1)
        for line in tail.splitlines():
            m = re.match(r"\[(\d+)\]\s+(.*)", line.strip())
            if m:
                sources.append({"n": int(m.group(1)), "citation": m.group(2)})
    if _RECORDS_HEAD in body:
        body, tail = body.split(_RECORDS_HEAD, 1)
        records = [line.strip() for line in tail.splitlines() if line.strip()]
    return {"body": body.strip(), "records": records, "sources": sources}


class AppService:
    def __init__(self, app: AppContainer, rag, extraction, assistant: HouseholdAssistant, clock: Callable[[], date] = date.today) -> None:
        self.app, self.rag, self.extraction, self.assistant, self.clock = app, rag, extraction, assistant, clock
        self.svc = build_services(app, clock)

    @classmethod
    def create(cls, settings: Settings, *, llm: LLMProvider | None = None, clock: Callable[[], date] = date.today,
               rebuild_index: bool = True, workflow_backend: str = "auto") -> "AppService":
        """Wires the whole application. With no `llm`, the configured provider is used; if that is not configured the
        assistant still works for records and shows manual passages, and extraction uses the rules only."""
        app = AppContainer.build(settings)
        rag = build_rag(app)
        if rebuild_index and settings.vector_store == "memory":
            count = rebuild_vector_index(app, rag)
            _log.info("rebuilt the in-memory vector index from %d stored chunks", count)
        real_llm = llm
        if real_llm is None:
            try:
                real_llm = build_llm(settings)
            except (GharCareError, ValueError) as exc:
                _log.warning("no LLM available (%s); AI features are limited", exc)
        extraction = build_extraction(app, real_llm)
        assistant_llm = real_llm or UnavailableLLMProvider("no LLM is configured: set GEMINI_API_KEY and GHARCARE_LLM_MODEL in .env")
        service = cls.__new__(cls)
        AppService.__init__(service, app, rag, extraction, None, clock)
        service.assistant = HouseholdAssistant(app.appliances, service.svc, rag.retriever, assistant_llm, backend=workflow_backend)
        service.llm_configured = real_llm is not None
        return service

    def close(self) -> None:
        self.app.close()

    def status(self) -> dict:
        return {"ok": True, "llm_configured": getattr(self, "llm_configured", True), "workflow_backend": self.assistant.backend,
                "embedder": self.rag.embedder.name, "vector_store": type(self.rag.vector_store).__name__}

    # ---- appliances ---------------------------------------------------------------------
    def appliances(self, user_id: str) -> list[dict]:
        return [appliance_json(a) for a in self.app.appliances.list_for_user(user_id)]

    def appliance_create(self, user_id: str, data: Mapping) -> dict:
        a = Appliance(user_id=user_id, name=data.get("name"), brand=data.get("brand"), model=data.get("model"),
                      serial_number=data.get("serial_number"), purchase_date=_date(data.get("purchase_date")),
                      purchase_price=data.get("purchase_price") or None, location=data.get("location"))
        return appliance_json(self.app.appliances.add(a))

    def appliance_update(self, user_id: str, appliance_id: str, data: Mapping) -> dict:
        current = self.app.appliances.get(user_id, appliance_id)
        changes = {k: data[k] for k in ("name", "brand", "model", "serial_number", "location") if k in data}
        if "purchase_date" in data:
            changes["purchase_date"] = _date(data["purchase_date"])
        if "purchase_price" in data:
            changes["purchase_price"] = data["purchase_price"] or None
        return appliance_json(self.app.appliances.update(replace(current, **changes)))

    def appliance_delete(self, user_id: str, appliance_id: str) -> None:
        self.app.appliances.delete(user_id, appliance_id)

    # ---- documents ----------------------------------------------------------------------
    def documents(self, user_id: str, appliance_id: str | None = None) -> list[dict]:
        return [document_json(d) for d in self.app.documents.list_for_user(user_id, appliance_id=appliance_id)]

    def document_upload(self, user_id: str, filename: str, content: bytes, doc_type: str, appliance_id: str | None = None) -> dict:
        result = self.rag.ingestion.ingest(user_id, filename, content, doc_type, appliance_id=appliance_id or None)
        return {"document": document_json(result.document), "chunk_count": result.chunk_count, "indexed": result.indexed,
                "warning": result.warning, "duplicate": result.duplicate}

    def document_delete(self, user_id: str, document_id: str) -> None:
        self.rag.ingestion.delete_document(user_id, document_id)

    def document_file(self, user_id: str, document_id: str) -> tuple[str, bytes]:
        doc = self.app.documents.get(user_id, document_id)
        return doc.original_filename, self.app.artifacts.read(user_id, doc.storage_path)

    # ---- extraction review --------------------------------------------------------------
    def extract(self, user_id: str, document_id: str) -> dict:
        return draft_json(self.extraction.propose(user_id, document_id))

    def drafts(self, user_id: str, status: str = "pending") -> list[dict]:
        try:
            wanted = DraftStatus(status) if status else None
        except ValueError as exc:
            raise DomainValidationError(f"unknown status {status!r} (use pending, confirmed or rejected)") from exc
        return [draft_json(d) for d in self.app.extraction_drafts.list_for_user(user_id, status=wanted)]

    def draft(self, user_id: str, draft_id: str) -> dict:
        return draft_json(self.extraction.get(user_id, draft_id))

    def draft_confirm(self, user_id: str, draft_id: str, values: Mapping, reviewed_fields=(), appliance_id: str | None = None) -> dict:
        result = self.extraction.confirm(user_id, draft_id, values, reviewed_fields=reviewed_fields, appliance_id=appliance_id or None)
        return {"draft": draft_json(result.draft), "created": result.created, "changes": list(result.changes)}

    def draft_reject(self, user_id: str, draft_id: str) -> dict:
        return draft_json(self.extraction.reject(user_id, draft_id))

    def extraction_accuracy(self, user_id: str) -> dict:
        a = self.extraction.accuracy(user_id)
        return {"drafts": a.drafts, "fields_read": a.fields_read, "fields_correct": a.fields_correct, "fields_missed": a.fields_missed,
                "accuracy": a.accuracy, "by_confidence": {k: list(v) for k, v in a.by_confidence.items()}}

    # ---- warranties / repairs / maintenance ---------------------------------------------
    def warranties(self, user_id: str) -> list[dict]:
        return [{"id": v.warranty.id, "appliance_id": v.appliance.id, "appliance": v.appliance.display_name,
                 "start_date": v.warranty.start_date.isoformat(), "expiry_date": v.expiry_date.isoformat(), "days_remaining": v.days_remaining,
                 "status": v.status.value, "provider": v.warranty.provider, "coverage_notes": v.warranty.coverage_notes,
                 "verified": v.warranty.verified} for v in self.svc.warranties.list_all(user_id)]

    def warranty_create(self, user_id: str, data: Mapping) -> dict:
        w = Warranty(user_id=user_id, appliance_id=data.get("appliance_id"), start_date=_date(data.get("start_date")) or self._bad("start_date"),
                     duration_months=data.get("duration_months"), provider=data.get("provider"),
                     coverage_notes=data.get("coverage_notes"), verified=True)       # typed in by the user: already their own confirmation
        self.app.warranties.add(w)
        return next(v for v in self.warranties(user_id) if v["id"] == w.id)

    def warranty_confirm(self, user_id: str, warranty_id: str) -> dict:
        self.svc.warranties.confirm(user_id, warranty_id)
        return next(v for v in self.warranties(user_id) if v["id"] == warranty_id)

    def repairs(self, user_id: str, appliance_id: str | None = None) -> list[dict]:
        return [repair_json(r) for r in self.app.repairs.list_for_user(user_id, appliance_id=appliance_id)]

    def repair_create(self, user_id: str, data: Mapping) -> dict:
        r = RepairRecord(user_id=user_id, appliance_id=data.get("appliance_id"), reported_date=_date(data.get("reported_date")) or self._bad("reported_date"),
                         problem_description=data.get("problem_description"), resolution=data.get("resolution"), technician=data.get("technician"),
                         parts_replaced=tuple(data.get("parts_replaced") or ()), cost=data.get("cost") or 0,
                         covered_by_warranty=data.get("covered_by_warranty"))
        return repair_json(self.app.repairs.add(r))

    def maintenance(self, user_id: str, appliance_id: str | None = None) -> dict:
        o = self.svc.maintenance.overview(user_id, appliance_id)
        row = lambda v: {"id": v.task.id, "appliance_id": v.appliance.id, "appliance": v.appliance.display_name, "title": v.task.title,
                         "next_due_date": v.task.next_due_date.isoformat() if v.task.next_due_date else None, "days_until": v.days_until,
                         "interval_months": v.task.interval_months, "status": v.status.value}
        return {"overdue": [row(v) for v in o.overdue], "due_soon": [row(v) for v in o.due_soon],
                "upcoming": [row(v) for v in o.upcoming], "completed": [row(v) for v in o.completed]}

    def task_create(self, user_id: str, data: Mapping) -> dict:
        t = MaintenanceTask(user_id=user_id, appliance_id=data.get("appliance_id"), title=data.get("title"),
                            next_due_date=_date(data.get("next_due_date")) or self._bad("next_due_date"),
                            interval_months=data.get("interval_months"), notes=data.get("notes"))
        self.app.maintenance.add(t)
        return {"id": t.id}

    def task_complete(self, user_id: str, task_id: str, data: Mapping) -> dict:
        task, log = self.svc.maintenance.complete_task(user_id, task_id, done_on=_date(data.get("done_on")), cost=data.get("cost") or None,
                                                       technician=data.get("technician"), notes=data.get("notes"))
        return {"task_id": task.id, "next_due_date": task.next_due_date.isoformat() if task.next_due_date else None, "log_id": log.id}

    # ---- expenses / report / chat ---------------------------------------------------------
    def expenses(self, user_id: str, months: int = 6, category: str | None = None, appliance_id: str | None = None) -> dict:
        try:
            categories = [ExpenseCategory(category)] if category else None
        except ValueError as exc:
            raise DomainValidationError(f"unknown category {category!r} (use purchase, repair or maintenance)") from exc
        filters = {"appliance_id": appliance_id or None, "categories": categories}
        s = self.svc.expenses.last_months(user_id, months, **filters)
        c = self.svc.expenses.compare_recent(user_id, months, **filters)
        return {"since": s.since.isoformat(), "until": s.until.isoformat(), "currency": s.currency, "total": _money(s.total),
                "by_category": {k.value: _money(v) for k, v in s.by_category.items()},
                "by_appliance": [{"appliance": t.appliance_name, "total": _money(t.total)} for t in s.by_appliance],
                "by_month": {k: _money(v) for k, v in s.by_month.items()},
                "lines": [{"date": l.date.isoformat(), "category": l.category.value, "appliance": l.appliance_name, "amount": _money(l.amount), "description": l.description} for l in s.lines],
                "gaps": {"records_without_cost": s.gaps.records_without_cost, "purchases_without_price": s.gaps.purchases_without_price,
                         "purchases_without_date": s.gaps.purchases_without_date},
                "comparison": {"previous_total": _money(c.previous.total), "delta": _money(c.delta), "pct_change": str(c.pct_change) if c.pct_change is not None else None}}

    def report(self, user_id: str, months: int = 6) -> dict:
        ref = self.svc.reports.generate(user_id, months)
        return {"path": ref.relative_path, "markdown": self.app.artifacts.read(user_id, ref.relative_path).decode("utf-8")}

    def ask(self, user_id: str, question: str, appliance_id: str | None = None) -> dict:
        if not (question or "").strip():
            raise DomainValidationError("question is empty")
        r = self.assistant.ask(user_id, question[:MAX_ASK_CHARS * 2], appliance_id or None)
        return {"status": r.status, "answer": r.answer, "citations": list(r.citations), "intent": r.intent, "trace": list(r.trace),
                "parts": split_answer(r.answer)}

    @staticmethod
    def _bad(field: str):
        raise DomainValidationError(f"{field} is required")
