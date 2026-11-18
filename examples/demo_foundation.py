"""Replays the 'Ayesha' story against the Step-1 foundation (no AI involved yet).
Run:  PYTHONPATH=src python examples/demo_foundation.py
Uses a temporary folder and SYNTHETIC file bytes only."""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import tempfile
from datetime import date
from decimal import Decimal

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import (
    Appliance, Document, DocumentType, MaintenanceTask, RepairRecord, Warranty,
)

TODAY = date(2026, 2, 20)
USER = "ayesha"

with tempfile.TemporaryDirectory() as tmp:
    app = AppContainer.build(Settings(data_dir=tmp))

    fridge = app.appliances.add(Appliance(
        user_id=USER, name="Fridge", brand="Haier", model="HRF-306",
        purchase_date=date(2024, 3, 14), purchase_price="95000", location="Kitchen"))
    ac = app.appliances.add(Appliance(user_id=USER, name="Split AC", brand="Gree", location="Bedroom"))

    # Upload (synthetic bytes standing in for a PDF manual); the original is stored untouched.
    ref = app.artifacts.save_original(USER, "Haier fridge manual.pdf", b"%PDF-1.4 synthetic manual")
    app.documents.add(Document(
        user_id=USER, appliance_id=fridge.id, original_filename="Haier fridge manual.pdf",
        doc_type=DocumentType.MANUAL, sha256=ref.sha256, storage_path=ref.relative_path,
        size_bytes=ref.size_bytes, mime_type="application/pdf"))

    app.warranties.add(Warranty(user_id=USER, appliance_id=fridge.id, start_date=date(2024, 3, 14),
                                duration_months=24, provider="Haier Pakistan", verified=True))
    for when, problem, cost in ((date(2025, 11, 2), "Ice build-up in freezer", "3500"),
                                (date(2025, 8, 20), "Door gasket loose", "1200")):
        app.repairs.add(RepairRecord(user_id=USER, appliance_id=fridge.id, reported_date=when,
                                     problem_description=problem, cost=cost))
    app.maintenance.add(MaintenanceTask(user_id=USER, appliance_id=ac.id, title="AC service",
                                        next_due_date=date(2026, 2, 1), interval_months=6))

    print(f"Today (demo date): {TODAY}\n")
    print("Warranties:")
    for w in app.warranties.list_for_user(USER):
        name = app.appliances.get(USER, w.appliance_id).display_name
        print(f"  {name}: expires {w.expiry_date} -> {w.status(TODAY).value} ({w.days_remaining(TODAY)} days)")

    print("\nMaintenance:")
    for t in app.maintenance.list_for_user(USER):
        name = app.appliances.get(USER, t.appliance_id).display_name
        print(f"  {name}: {t.title} due {t.next_due_date} -> {t.status(TODAY).value}")

    repairs = app.repairs.list_for_user(USER, since=date(2025, 8, 20))
    print(f"\nRepairs since 2025-08-20: {len(repairs)}, total PKR {sum((r.cost for r in repairs), Decimal('0')):,.2f}")

    print("\nRepair history for the fridge (what a 'same problem again?' check will read):")
    for r in app.repairs.list_for_user(USER, appliance_id=fridge.id):
        print(f"  {r.reported_date}: {r.problem_description}")

    print(f"\nStored original: {ref.relative_path}")
    print(f"Re-upload of same file creates a duplicate? {app.artifacts.save_original(USER, 'copy.pdf', b'%PDF-1.4 synthetic manual').created}")
    app.close()
