"""FastAPI shell around AppService: routing, request validation, auth, error mapping and security headers.
All behaviour lives in AppService (tested without a web server)."""
from __future__ import annotations

import hmac
import re
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gharcare.api.service import AppService
from gharcare.errors import DomainValidationError, GharCareError, LLMError, NotFoundError, StorageError

STATIC = Path(__file__).parent / "static"


class ApplianceIn(BaseModel):
    name: str
    brand: str | None = None
    model: str | None = None
    serial_number: str | None = None
    purchase_date: str | None = None
    purchase_price: str | None = None
    location: str | None = None


class ApplianceUpdate(BaseModel):
    name: str | None = None
    brand: str | None = None
    model: str | None = None
    serial_number: str | None = None
    purchase_date: str | None = None
    purchase_price: str | None = None
    location: str | None = None


class AskIn(BaseModel):
    question: str = Field(max_length=4000)
    appliance_id: str | None = None


class ConfirmIn(BaseModel):
    values: dict[str, str | None]
    reviewed_fields: list[str] = []
    appliance_id: str | None = None


class WarrantyIn(BaseModel):
    appliance_id: str
    start_date: str
    duration_months: int
    provider: str | None = None
    coverage_notes: str | None = None


class RepairIn(BaseModel):
    appliance_id: str
    reported_date: str
    problem_description: str
    resolution: str | None = None
    technician: str | None = None
    parts_replaced: list[str] = []
    cost: str | None = None
    covered_by_warranty: bool | None = None


class TaskIn(BaseModel):
    appliance_id: str
    title: str
    next_due_date: str
    interval_months: int | None = None
    notes: str | None = None


class CompleteIn(BaseModel):
    done_on: str | None = None
    cost: str | None = None
    technician: str | None = None
    notes: str | None = None


def _status_for(exc: Exception) -> int:
    if isinstance(exc, NotFoundError):
        return 404
    if isinstance(exc, (DomainValidationError, ValueError)):
        return 422
    if isinstance(exc, LLMError):
        return 503
    if isinstance(exc, StorageError):
        return 500
    return 400


def create_app(service: AppService, *, user_id: str, token: str | None = None, max_upload_bytes: int = 20 * 1024 * 1024) -> FastAPI:
    app = FastAPI(title="GharCare AI", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.exception_handler(GharCareError)
    async def gharcare_error(request: Request, exc: GharCareError):
        status = _status_for(exc)
        return JSONResponse({"error": "internal storage error" if status == 500 else str(exc)}, status_code=status)

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith("/api/docs"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    def require_auth(authorization: str | None = Header(default=None)) -> str:
        if token and not (authorization and hmac.compare_digest(authorization, f"Bearer {token}")):
            raise HTTPException(status_code=401, detail="missing or wrong access token")
        return user_id

    @app.get("/api/health")
    def health():
        return service.status()

    api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

    # ---- chat
    @api.post("/ask")
    def ask(body: AskIn):
        return service.ask(user_id, body.question, body.appliance_id)

    # ---- appliances
    @api.get("/appliances")
    def appliances():
        return service.appliances(user_id)

    @api.post("/appliances", status_code=201)
    def appliance_create(body: ApplianceIn):
        return service.appliance_create(user_id, body.model_dump())

    @api.put("/appliances/{appliance_id}")
    def appliance_update(appliance_id: str, body: ApplianceUpdate):
        return service.appliance_update(user_id, appliance_id, body.model_dump(exclude_unset=True))

    @api.delete("/appliances/{appliance_id}", status_code=204)
    def appliance_delete(appliance_id: str):
        service.appliance_delete(user_id, appliance_id)
        return Response(status_code=204)

    # ---- documents
    @api.get("/documents")
    def documents(appliance_id: str | None = None):
        return service.documents(user_id, appliance_id)

    @api.post("/documents", status_code=201)
    def upload(file: UploadFile = File(...), doc_type: str = Form(...), appliance_id: str | None = Form(None)):
        content = file.file.read(max_upload_bytes + 1)          # sync route: runs in FastAPI's threadpool, never blocks the event loop
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"file is larger than {max_upload_bytes // (1024 * 1024)} MB")
        return service.document_upload(user_id, file.filename or "upload", content, doc_type, appliance_id)

    @api.delete("/documents/{document_id}", status_code=204)
    def document_delete(document_id: str):
        service.document_delete(user_id, document_id)
        return Response(status_code=204)

    @api.get("/documents/{document_id}/file")
    def document_file(document_id: str):
        name, content = service.document_file(user_id, document_id)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        return Response(content, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{safe}"'})

    @api.post("/documents/{document_id}/extract")
    def extract(document_id: str):
        return service.extract(user_id, document_id)

    # ---- extraction review
    @api.get("/drafts")
    def drafts(status: str = "pending"):
        return service.drafts(user_id, status)

    @api.get("/drafts/{draft_id}")
    def draft(draft_id: str):
        return service.draft(user_id, draft_id)

    @api.post("/drafts/{draft_id}/confirm")
    def draft_confirm(draft_id: str, body: ConfirmIn):
        return service.draft_confirm(user_id, draft_id, body.values, body.reviewed_fields, body.appliance_id)

    @api.post("/drafts/{draft_id}/reject")
    def draft_reject(draft_id: str):
        return service.draft_reject(user_id, draft_id)

    @api.get("/extraction/accuracy")
    def accuracy():
        return service.extraction_accuracy(user_id)

    # ---- records
    @api.get("/warranties")
    def warranties():
        return service.warranties(user_id)

    @api.post("/warranties", status_code=201)
    def warranty_create(body: WarrantyIn):
        return service.warranty_create(user_id, body.model_dump())

    @api.post("/warranties/{warranty_id}/confirm")
    def warranty_confirm(warranty_id: str):
        return service.warranty_confirm(user_id, warranty_id)

    @api.get("/repairs")
    def repairs(appliance_id: str | None = None):
        return service.repairs(user_id, appliance_id)

    @api.post("/repairs", status_code=201)
    def repair_create(body: RepairIn):
        return service.repair_create(user_id, body.model_dump())

    @api.get("/maintenance")
    def maintenance(appliance_id: str | None = None):
        return service.maintenance(user_id, appliance_id)

    @api.post("/maintenance", status_code=201)
    def task_create(body: TaskIn):
        return service.task_create(user_id, body.model_dump())

    @api.post("/maintenance/{task_id}/complete")
    def task_complete(task_id: str, body: CompleteIn):
        return service.task_complete(user_id, task_id, body.model_dump())

    # ---- expenses and report
    @api.get("/expenses")
    def expenses(months: int = 6, category: str | None = None, appliance_id: str | None = None):
        if not 1 <= months <= 120:
            raise HTTPException(status_code=422, detail="months must be between 1 and 120")
        return service.expenses(user_id, months, category or None, appliance_id or None)

    @api.post("/report")
    def report(months: int = 6):
        if not 1 <= months <= 120:
            raise HTTPException(status_code=422, detail="months must be between 1 and 120")
        return service.report(user_id, months)

    app.include_router(api)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app
