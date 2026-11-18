"""Step-2 demo: ingest the SYNTHETIC sample manual, run the labeled evaluation, show cited answers.

  Offline baseline (standard library only):
      PYTHONPATH=src python examples/demo_rag.py
  Real stack (after: pip install -r requirements-rag.txt):
      PYTHONPATH=src python examples/demo_rag.py --embedder sentence-transformers --vector-store chroma
      (first run downloads the embedding model)
"""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import argparse
import tempfile
from pathlib import Path

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance
from gharcare.evaluation import evaluate_retrieval, load_eval_cases
from gharcare.rag import build_rag

HERE = Path(__file__).parent / "sample_docs"
USER = "demo"

parser = argparse.ArgumentParser()
parser.add_argument("--embedder", default="hashing", choices=["hashing", "sentence-transformers"])
parser.add_argument("--vector-store", default="memory", choices=["memory", "chroma"])
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
parser.add_argument("--ask", action="append", default=[], help="extra question(s) to try")
args = parser.parse_args()

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    settings = Settings(data_dir=tmp, embedder=args.embedder, vector_store=args.vector_store,
                        embedding_model=args.model)
    app = AppContainer.build(settings)
    rag = build_rag(app)
    fridge = app.appliances.add(Appliance(user_id=USER, name="Fridge", brand="Sample"))

    manual = HERE / "xr200_sample_manual_SYNTHETIC.txt"
    result = rag.ingestion.ingest(USER, manual.name, manual.read_bytes(), "manual", fridge.id)
    print(f"Embedder: {args.embedder} | vector store: {args.vector_store}")
    print(f"Ingested {manual.name}: indexed={result.indexed}, chunks={result.chunk_count}\n")

    questions = ["How do I clear ice build-up in the freezer?", "What does error code E4 mean?",
                 "fridge mein barf jam rahi hai kya karun", "What is the freezer capacity in litres?"] + args.ask
    for q in questions:
        print(f"Q: {q}")
        hits = rag.retriever.retrieve(USER, q, appliance_id=fridge.id, k=2)
        if not hits:
            print("   (nothing relevant found in your documents)")
        for h in hits:
            preview = h.chunk.text.replace("\n", " ")[:90]
            sim = "-" if h.vector_similarity is None else f"{h.vector_similarity:.2f}"
            bm = "-" if h.keyword_score is None else f"{h.keyword_score:.1f}"
            print(f"   [{h.citation}] sim={sim} bm25={bm}  {preview}...")
        print()

    cases = load_eval_cases(HERE / "eval_questions.json")
    report = evaluate_retrieval(rag.retriever, USER, cases, appliance_id=fridge.id)
    print(f"=== Retrieval evaluation ({len(cases)} labeled questions on a 5-page synthetic manual) ===")
    print(report.format())
    app.close()
