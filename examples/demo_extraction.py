"""Step-5 demo: upload synthetic documents, read them, show confidence, review, confirm, and measure accuracy.

  python examples/demo_extraction.py            # rules only (no API key needed)
  python examples/demo_extraction.py --llm      # also use the AI reader configured in .env
Uses SYNTHETIC documents only."""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import argparse
import dataclasses
import os
import tempfile
from pathlib import Path

from gharcare.config import Settings
from gharcare.domain import Appliance
from gharcare.container import AppContainer
from gharcare.env import load_env_file
from gharcare.extraction.factory import build_extraction
from gharcare.llm import build_llm
from gharcare.logging_utils import configure_logging
from gharcare.rag import build_rag

load_env_file()
configure_logging()
parser = argparse.ArgumentParser()
parser.add_argument("--llm", action="store_true", help="add the AI reader (needs .env)")
args = parser.parse_args()
HERE, U = Path(__file__).parent / "sample_docs", "ayesha"

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    base = Settings.from_env()
    settings = dataclasses.replace(base, data_dir=Path(tmp), embedder="hashing", vector_store="memory")
    app = AppContainer.build(settings)
    rag = build_rag(app)
    svc = build_extraction(app, build_llm(settings) if args.llm else None)
    print(f"Readers: rules{' + AI (' + str(settings.llm_model) + ')' if args.llm else ' only'}\n")

    for filename, doc_type in (("receipt_SYNTHETIC.txt", "receipt"), ("warranty_card_SYNTHETIC.txt", "warranty_card"),
                               ("service_invoice_SYNTHETIC.txt", "service_invoice")):
        doc = rag.ingestion.ingest(U, filename, (HERE / filename).read_bytes(), doc_type).document
        draft = svc.propose(U, doc.id)
        print(f"== {filename}")
        for name, f in draft.fields.items():
            flag = "" if f.confidence.value == "high" else f"   <-- REVIEW ({f.note})"
            alt = f" [other readings: {', '.join(f.alternatives)}]" if f.alternatives else ""
            print(f"   {name:<26} {str(f.value):<32} {f.confidence.value:<6} {f.source:<5}{alt}{flag}")
        values = {n: f.value for n, f in draft.fields.items()}
        uncertain = [n for n, f in draft.fields.items() if f.value is not None and f.confidence.value != "high"]
        if filename.startswith("warranty"):
            values["warranty_duration_months"] = "36"        # simulate the user correcting one field
            print("   (user corrects warranty duration 24 -> 36)")
        ac = app.appliances.add(Appliance(user_id=U, name="Split AC", brand="Gree")) if doc_type == "service_invoice" else None
        result = svc.confirm(U, draft.id, values, reviewed_fields=uncertain, appliance_id=ac.id if ac else None)
        print(f"   saved: {result.created or 'appliance details only'}   changes: {list(result.changes)}\n")

    acc = svc.accuracy(U)
    print(f"Extraction accuracy vs what the user confirmed: {acc.fields_correct}/{acc.fields_read} = {acc.accuracy:.0%}; missed by the reader: {acc.fields_missed}")
    print("By confidence (correct, read):", acc.by_confidence)
    print("Appliances now:", [a.display_name for a in app.appliances.list_for_user(U)], "| warranties:", len(app.warranties.list_for_user(U)),
          "| repairs:", len(app.repairs.list_for_user(U)))
    app.close()
