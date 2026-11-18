"""Step-4 demo: the full assistant (LangGraph workflow) on the Ayesha household + SYNTHETIC sample manual.

Reads settings from a .env file (copy .env.example to .env and paste your key). With no LLM configured it uses an
OFFLINE STAND-IN model (plain extracts), so it still runs.

  PYTHONPATH=src python examples/demo_assistant.py                 # uses .env if present
  PYTHONPATH=src python examples/demo_assistant.py --llm fake      # force the offline stand-in
"""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import argparse
import dataclasses
import os
import re
import tempfile
from datetime import date
from pathlib import Path

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance, MaintenanceTask, RepairRecord, Warranty
from gharcare.env import load_env_file
from gharcare.llm import FakeLLMProvider, build_llm
from gharcare.logging_utils import configure_logging
from gharcare.rag import build_rag
from gharcare.services import build_services
from gharcare.workflow import HouseholdAssistant

load_env_file()
configure_logging()          # shows the real reason if an LLM call fails (your key is redacted)
parser = argparse.ArgumentParser()
parser.add_argument("--llm", default="auto", choices=["auto", "fake"])
parser.add_argument("--embedder", default=None, choices=["hashing", "sentence-transformers"])
parser.add_argument("--vector-store", default=None, choices=["memory", "chroma"])
parser.add_argument("--backend", default="auto", choices=["auto", "local", "langgraph"])
parser.add_argument("--ask", action="append", default=[])
args = parser.parse_args()
TODAY, U = date(2026, 2, 20), "ayesha"


def offline_model(messages):
    """OFFLINE STAND-IN, not a real model: judges 'supported', classifies as troubleshooting, translates one demo phrase,
    and answers by quoting excerpt 1."""
    system, user = messages[0].content, messages[-1].content
    if system.startswith("You are a strict fact checker"):
        return '{"supported": true, "unsupported_claims": []}'
    if system.startswith("Classify"):
        return "troubleshooting"
    if system.startswith("Translate"):          # a tiny phrase table, enough for the demo's Roman Urdu question
        return "Ice is building up in the fridge freezer, what should I do?" if "barf" in user else user
    m = re.search(r'<excerpt id="1"[^>]*>\n(.*?)\n</excerpt>', user, re.S)
    from gharcare.services.context import _keywords
    question = re.search(r"QUESTION: (.*)", user).group(1)
    if len(_keywords(question) & _keywords(m.group(1) if m else "")) < 2:
        return "NOT_FOUND"          # the stand-in declines when the excerpt shares fewer than 2 of the question's keywords
    body = " ".join((m.group(1) if m else "").split("\n")[1:4])
    return f"From your manual: {body[:260]} [1]"


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    base = Settings.from_env()
    settings = dataclasses.replace(
        base, data_dir=Path(tmp),
        embedder=args.embedder or os.environ.get("GHARCARE_EMBEDDER") or "hashing",
        vector_store=args.vector_store or os.environ.get("GHARCARE_VECTOR_STORE") or "memory")
    app = AppContainer.build(settings)
    svc = build_services(app, clock=lambda: TODAY)
    rag = build_rag(app)
    configured = bool(settings.llm_model) and (settings.llm_provider == "ollama" or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if args.llm == "fake" or not configured:
        llm, llm_name = FakeLLMProvider(responder=offline_model), "offline stand-in (set GEMINI_API_KEY and GHARCARE_LLM_MODEL in .env for a real model)"
    else:
        llm, llm_name = build_llm(settings), f"{settings.llm_provider}/{settings.llm_model}"
    fridge = app.appliances.add(Appliance(user_id=U, name="Fridge", brand="Haier", model="HRF-306", purchase_date=date(2024, 3, 14), purchase_price="95000"))
    ac = app.appliances.add(Appliance(user_id=U, name="Split AC", brand="Gree", purchase_date=date(2025, 6, 1), purchase_price="150000"))
    app.warranties.add(Warranty(user_id=U, appliance_id=fridge.id, start_date=date(2024, 3, 14), duration_months=24, verified=True))
    app.repairs.add(RepairRecord(user_id=U, appliance_id=fridge.id, reported_date=date(2025, 11, 2), problem_description="Ice build-up in freezer",
                                 resolution="Replaced defrost sensor", cost="3500", parts_replaced=("Defrost sensor",)))
    app.maintenance.add(MaintenanceTask(user_id=U, appliance_id=ac.id, title="AC service", next_due_date=date(2026, 2, 1), interval_months=6))
    manual = Path(__file__).parent / "sample_docs" / "xr200_sample_manual_SYNTHETIC.txt"
    rag.ingestion.ingest(U, manual.name, manual.read_bytes(), "manual", fridge.id)

    assistant = HouseholdAssistant(app.appliances, svc, rag.retriever, llm, backend=args.backend)
    print(f"Workflow backend: {assistant.backend} | llm: {llm_name} | embedder: {settings.embedder} | vector store: {settings.vector_store}\n")
    questions = ["fridge mein barf jam rahi hai kya karun", "What does error code E4 mean?", "AC se jalne ki smell aa rahi hai aur spark nikla",
                 "How much did I spend on repairs in the last 6 months?", "Which warranties will expire soon?", "Which maintenance is overdue?",
                 "What is the freezer capacity in litres?"] + args.ask
    for q in questions:
        r = assistant.ask(U, q)
        print(f"Q: {q}\n[{r.status} | intent={r.intent} | {' > '.join(r.trace)}]\n{r.answer}\n")
    app.close()
