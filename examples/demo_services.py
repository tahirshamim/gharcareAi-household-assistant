"""Step-3 demo: the exact-answer questions from the Ayesha story, answered by plain Python.
Run:  PYTHONPATH=src python examples/demo_services.py     (uses a temp folder, synthetic data)"""
import _bootstrap  # noqa: F401  (must come first: use this folder's src/)
import tempfile
from datetime import date

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance, MaintenanceTask, RepairRecord, Warranty
from gharcare.services import ExpenseCategory, build_services, format_money

TODAY, U = date(2026, 2, 20), "ayesha"
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    app = AppContainer.build(Settings(data_dir=tmp))
    svc = build_services(app, clock=lambda: TODAY)
    fridge = app.appliances.add(Appliance(user_id=U, name="Fridge", brand="Haier", model="HRF-306",
                                          purchase_date=date(2024, 3, 14), purchase_price="95000"))
    ac = app.appliances.add(Appliance(user_id=U, name="Split AC", brand="Gree", purchase_date=date(2025, 6, 1), purchase_price="150000"))
    app.warranties.add(Warranty(user_id=U, appliance_id=fridge.id, start_date=date(2024, 3, 14), duration_months=24, verified=True))
    for d, p, c in ((date(2025, 8, 20), "Door gasket loose", "1200"), (date(2025, 11, 2), "Ice build-up in freezer", "3500")):
        app.repairs.add(RepairRecord(user_id=U, appliance_id=fridge.id, reported_date=d, problem_description=p, cost=c,
                                     resolution="Replaced defrost sensor" if "Ice" in p else None))
    app.maintenance.add(MaintenanceTask(user_id=U, appliance_id=ac.id, title="AC service", next_due_date=date(2026, 2, 1), interval_months=6))
    svc.maintenance.log_service(U, ac.id, description="AC service", done_on=date(2025, 12, 1), cost="2500")

    print(f"Today: {TODAY}\n")
    print("Q: Which warranties expire soon?")
    for v in svc.warranties.expiring_soon(U, within_days=60):
        print(f"   {v.appliance.display_name}: {v.expiry_date} ({v.days_remaining} days)")
    print("\nQ: Which maintenance is overdue?")
    for v in svc.maintenance.overview(U).overdue:
        print(f"   {v.appliance.display_name}: {v.task.title}, {-v.days_until} days overdue")
    s = svc.expenses.last_months(U, 6, categories=[ExpenseCategory.REPAIR])
    print(f"\nQ: How much did I spend on repairs in the last 6 months?\n   {format_money(s.total)}  ({len(s.lines)} repairs)")
    c = svc.expenses.compare_recent(U, 6)
    print(f"\nQ: Total spend, last 6 months vs the 6 before?\n   {format_money(c.current.total)} vs {format_money(c.previous.total)} (change {c.pct_change}%)")
    print("\nQ: The fridge has ice build-up again. What do the records say?")
    print("   " + svc.context.build(U, fridge.id, "ice build up in freezer").facts_text().replace("\n", "\n   "))
    ref = svc.reports.generate(U)
    print(f"\nReport saved: {ref.relative_path}")
    print("\n".join(app.artifacts.read(U, ref.relative_path).decode().splitlines()[:8]))
    app.close()
