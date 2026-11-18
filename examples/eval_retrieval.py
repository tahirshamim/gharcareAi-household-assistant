"""Retrieval-only evaluation on any manual + labeled question file (no LLM, no API quota).

  python examples/eval_retrieval.py --manual examples/real_docs/590939.pdf --questions examples/sample_docs/eval_questions_gorenje.json
  add: --embedder sentence-transformers --vector-store chroma     to compare the real semantic model with the baseline
"""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import argparse
import dataclasses
import os
import tempfile
from pathlib import Path

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance
from gharcare.env import load_env_file
from gharcare.evaluation import evaluate_retrieval, first_relevant_rank, load_eval_cases
from gharcare.rag import build_rag

load_env_file()
HERE = Path(__file__).parent / "sample_docs"
parser = argparse.ArgumentParser()
parser.add_argument("--manual", default=str(HERE / "xr200_sample_manual_SYNTHETIC.txt"))
parser.add_argument("--questions", default=str(HERE / "eval_questions.json"))
parser.add_argument("--embedder", default=None, choices=["hashing", "sentence-transformers"])
parser.add_argument("--vector-store", default=None, choices=["memory", "chroma"])
args = parser.parse_args()
U = "eval"

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    base = Settings.from_env()
    settings = dataclasses.replace(base, data_dir=Path(tmp), embedder=args.embedder or os.environ.get("GHARCARE_EMBEDDER") or "hashing",
                                   vector_store=args.vector_store or os.environ.get("GHARCARE_VECTOR_STORE") or "memory")
    app = AppContainer.build(settings)
    rag = build_rag(app)
    appliance = app.appliances.add(Appliance(user_id=U, name="Appliance"))
    manual = Path(args.manual)
    result = rag.ingestion.ingest(U, manual.name, manual.read_bytes(), "manual", appliance.id)
    print(f"{manual.name}: indexed={result.indexed}, {result.chunk_count} chunks | embedder={settings.embedder}, vector store={settings.vector_store}")
    cases = load_eval_cases(args.questions)
    report = evaluate_retrieval(rag.retriever, U, cases, k_values=(1, 3, 4, 5), appliance_id=appliance.id)
    print(report.format())
    print("\nAnswerable questions whose first correct chunk is NOT ranked first:")
    for case in cases:
        if not case.answerable:
            continue
        hits = rag.retriever.retrieve(U, case.question, appliance_id=appliance.id, k=5)
        rank = first_relevant_rank(hits, case.relevant, case.must_contain)
        if rank != 1:
            print(f"  rank {rank or '>5'}: {case.question}\n      wanted pages {sorted(p for _, p in case.relevant)}; got pages {[h.chunk.page_number for h in hits[:3]]}")
    app.close()
