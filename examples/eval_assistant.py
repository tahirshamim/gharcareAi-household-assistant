"""End-to-end evaluation of the assistant with a REAL model. Uses the SYNTHETIC manual only.

  python examples/eval_assistant.py --out results.json
  python examples/eval_assistant.py --limit 5            # quick smoke test (saves free-tier quota)
Settings come from .env (GEMINI_API_KEY, GHARCARE_LLM_MODEL, ...). Add --embedder sentence-transformers
--vector-store chroma to evaluate the real semantic stack."""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import argparse
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance
from gharcare.env import load_env_file
from gharcare.evaluation import evaluate_assistant, load_eval_cases
from gharcare.llm import CachingLLMProvider, MeteredLLMProvider, build_llm
from gharcare.logging_utils import configure_logging
from gharcare.rag import build_rag
from gharcare.services import build_services
from gharcare.workflow import HouseholdAssistant

load_env_file()
configure_logging()
parser = argparse.ArgumentParser()
parser.add_argument("--out", default=None, help="write the full results as JSON")
parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
parser.add_argument("--pause", type=float, default=0.0, help="seconds between questions (use e.g. 6 if you see rate-limit warnings)")
parser.add_argument("--max-calls", type=int, default=None, help="stop before using more than N real LLM calls (protects a small daily quota)")
parser.add_argument("--cache", default="eval_llm_cache.jsonl", help="cache of LLM answers; lets a stopped run resume")
parser.add_argument("--no-cache", action="store_true")
parser.add_argument("--manual", default=None, help="manual to evaluate on (default: the synthetic sample)")
parser.add_argument("--questions", default=None, help="labeled question file (default: the synthetic sample set)")
parser.add_argument("--quiet", action="store_true", help="hide per-call timings")
parser.add_argument("--embedder", default=None, choices=["hashing", "sentence-transformers"])
parser.add_argument("--vector-store", default=None, choices=["memory", "chroma"])
parser.add_argument("--backend", default="auto", choices=["auto", "local", "langgraph"])
args = parser.parse_args()

HERE, U = Path(__file__).parent / "sample_docs", "eval"
base = Settings.from_env()
if not base.llm_model or not (base.llm_provider == "ollama" or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    sys.exit("No real LLM configured: set GEMINI_API_KEY and GHARCARE_LLM_MODEL in .env (check with: python -m gharcare.doctor --live)")

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    settings = dataclasses.replace(base, data_dir=Path(tmp),
                                   embedder=args.embedder or os.environ.get("GHARCARE_EMBEDDER") or "hashing",
                                   vector_store=args.vector_store or os.environ.get("GHARCARE_VECTOR_STORE") or "memory")
    app = AppContainer.build(settings)
    rag = build_rag(app)
    import time
    t0 = time.perf_counter()
    meter = MeteredLLMProvider(build_llm(settings),
                               on_call=None if args.quiet else (lambda secs, ok: print(f"      LLM call {secs:5.1f}s {'ok' if ok else 'FAILED'}", flush=True)))
    fridge = app.appliances.add(Appliance(user_id=U, name="Fridge", brand="Sample"))
    manual = Path(args.manual) if args.manual else HERE / "xr200_sample_manual_SYNTHETIC.txt"
    rag.ingestion.ingest(U, manual.name, manual.read_bytes(), "manual", fridge.id)
    llm = meter if args.no_cache else CachingLLMProvider(meter, args.cache, namespace=f"{settings.llm_provider}/{settings.llm_model}")
    assistant = HouseholdAssistant(app.appliances, build_services(app), rag.retriever, llm, backend=args.backend)
    cases = load_eval_cases(args.questions or HERE / "eval_questions.json")[: args.limit]
    print(f"Running {len(cases)} questions. Progress is shown per question; each LLM call is timed.\n", flush=True)

    def progress(i, n, r):
        elapsed = time.perf_counter() - t0
        print(f"[{i}/{n}] {r.seconds:5.1f}s, {r.llm_calls} LLM call(s) -> {r.outcome:<18} {r.question[:60]}   (elapsed {elapsed:.0f}s, ~{elapsed / i * (n - i):.0f}s left)", flush=True)
    report = evaluate_assistant(assistant, U, cases, appliance_id=fridge.id, meter=meter, pause_seconds=args.pause, on_result=progress, max_llm_calls=args.max_calls,
                                meta={"llm": f"{settings.llm_provider}/{settings.llm_model}", "embedder": settings.embedder,
                                      "vector_store": settings.vector_store, "workflow_backend": assistant.backend,
                                      "document": manual.name + ("" if args.manual else " (synthetic)")})
    print(report.format())
    if not args.no_cache:
        print(f"\nLLM answers served from cache: {llm.hits}; new real calls: {meter.calls}")
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull results written to {args.out}")
    app.close()
