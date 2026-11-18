"""Explain individual evaluation cases: what was retrieved, what the model drafted, what the judge said.
Replays eval_assistant.py's response cache and makes NO real LLM calls (zero quota), so use the same
--manual / --embedder / --vector-store flags as the evaluation run.

  python examples/inspect_cases.py --manual examples/real_docs/590939.pdf ^
      --questions examples/sample_docs/eval_questions_gorenje.json ^
      --embedder sentence-transformers --vector-store chroma --results gorenje_results.json --out inspect.txt

Without --results, pass --question "power strip" (substring, repeatable) to choose cases.
Note: the output quotes your document, so treat it like the document itself."""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import argparse
import dataclasses
import json
import os
import tempfile
from pathlib import Path

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance
from gharcare.env import load_env_file
from gharcare.errors import LLMResponseError
from gharcare.evaluation import load_eval_cases
from gharcare.llm import CachingLLMProvider, LLMProvider
from gharcare.rag import build_rag
from gharcare.services import build_services
from gharcare.workflow import HouseholdAssistant

FAILED = {"abstained", "unverified", "wrong_citation", "answered", "safe_fallback", "other", "unavailable"}


class CacheOnly(LLMProvider):
    """Stands in for the real model: anything not already cached fails instead of using quota."""
    name = "cache-only"

    def generate(self, messages, **kw):
        raise LLMResponseError("not in the cache (this script never calls the real model)")


class Recorder(LLMProvider):
    def __init__(self, inner):
        self._inner, self.name, self.log = inner, inner.name, []

    def generate(self, messages, **kw):
        try:
            r = self._inner.generate(messages, **kw)
            self.log.append(r.text)
            return r
        except Exception as exc:
            self.log.append(f"<{type(exc).__name__}: {exc}>")
            raise


load_env_file()
p = argparse.ArgumentParser()
p.add_argument("--manual", required=True)
p.add_argument("--questions", required=True)
p.add_argument("--results", help="results JSON from eval_assistant.py: inspect every case that did not succeed")
p.add_argument("--question", action="append", default=[], help="substring of a question to inspect (repeatable)")
p.add_argument("--cache", default="eval_llm_cache.jsonl")
p.add_argument("--embedder", default=None, choices=["hashing", "sentence-transformers"])
p.add_argument("--vector-store", default=None, choices=["memory", "chroma"])
p.add_argument("--out", default=None, help="also write the report to this file")
args = p.parse_args()

cases = load_eval_cases(args.questions)
wanted = {s.lower() for s in args.question}
if args.results:
    rows = json.loads(Path(args.results).read_text(encoding="utf-8")).get("cases", [])
    wanted |= {r["question"].lower() for r in rows if r.get("outcome") in FAILED}
chosen = [c for c in cases if any(w in c.question.lower() for w in wanted)]
if not chosen:
    raise SystemExit("no cases selected: pass --results or --question")

lines = []
out = lines.append
base = Settings.from_env()
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    settings = dataclasses.replace(base, data_dir=Path(tmp),
                                   embedder=args.embedder or os.environ.get("GHARCARE_EMBEDDER") or "hashing",
                                   vector_store=args.vector_store or os.environ.get("GHARCARE_VECTOR_STORE") or "memory")
    app = AppContainer.build(settings)
    rag = build_rag(app)
    fridge = app.appliances.add(Appliance(user_id="eval", name="Fridge", brand="Sample"))
    manual = Path(args.manual)
    rag.ingestion.ingest("eval", manual.name, manual.read_bytes(), "manual", fridge.id)
    llm = Recorder(CachingLLMProvider(CacheOnly(), args.cache, namespace=f"{settings.llm_provider}/{settings.llm_model}"))
    assistant = HouseholdAssistant(app.appliances, build_services(app), rag.retriever, llm)
    for case in chosen:
        llm.log.clear()
        hits = rag.retriever.retrieve("eval", case.question, appliance_id=fridge.id, k=4)
        r = assistant.ask("eval", case.question, appliance_id=fridge.id)
        out("=" * 100)
        out(f"Q: {case.question}")
        out(f"   expected pages {sorted(pg for _, pg in case.relevant)}; evidence phrase: {case.must_contain!r}")
        out(f"   final status: {r.status}")
        for i, h in enumerate(hits, 1):
            text = " ".join(h.chunk.text.split())
            has = case.must_contain and case.must_contain.lower() in text.lower()
            sim = f"{h.vector_similarity:.3f}" if h.vector_similarity is not None else "-"
            bm = f"{h.keyword_score:.1f}" if h.keyword_score is not None else "-"
            out(f"  [{i}] p.{h.chunk.page_number} {h.chunk.section or ''} | sim {sim} bm25 {bm}{'  <-- has evidence' if has else ''}")
            out(f"      {text[:400]}")
        for i, t in enumerate(llm.log, 1):
            out(f"  LLM response {i}: {' '.join(t.split())[:600]}")
    app.close()

report = "\n".join(lines)
print(report)
if args.out:
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nwritten to {args.out}")
