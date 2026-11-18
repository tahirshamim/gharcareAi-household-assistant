from __future__ import annotations

from gharcare.container import AppContainer
from gharcare.extraction.pipeline import ExtractionPipeline
from gharcare.extraction.service import ExtractionService
from gharcare.llm.base import LLMProvider


def build_extraction(app: AppContainer, llm: LLMProvider | None = None) -> ExtractionService:
    """Pass an LLM to add the AI reader; without one, extraction uses the deterministic rules only."""
    return ExtractionService(app.documents, app.appliances, app.warranties, app.repairs, app.extraction_drafts,
                             app.artifacts, app.db, ExtractionPipeline(llm))
