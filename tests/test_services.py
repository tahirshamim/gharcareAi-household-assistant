import sqlite3
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import (
    Appliance, Document, DocumentType, MaintenanceTask, RepairRecord, TaskStatus, Warranty, WarrantyStatus,
)
from gharcare.errors import DomainValidationError, NotFoundError
from gharcare.repositories import SQLiteDatabase, SQLiteApplianceRepository
from gharcare.repositories.sqlite import _SCHEMA_V1, _SCHEMA_V2, SCHEMA_VERSION
from gharcare.services import CoverageState, ExpenseCategory, build_services, similarity

TODAY = date(2026, 2, 20)
U = "ayesha"


class ServiceTestCase(unittest.TestCase):
    """Ayesha's household, with every expected number computed by hand in the tests below."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.app = AppContainer.build(Settings(data_dir=self._tmp.name))
        self.addCleanup(self.app.close)
        self.svc = build_services(self.app, clock=lambda: TODAY)
        a = self.app
        self.fridge = a.appliances.add(Appliance(user_id=U, name="Fridge", brand="Haier", model="HRF-306",
                                                 purchase_date=date(2024, 3, 14), purchase_price="95000", location="Kitchen"))
        self.ac = a.appliances.add(Appliance(user_id=U, name="Split AC", brand="Gree",
                                             purchase_date=date(2025, 6, 1), purchase_price="150000"))
        self.washer = a.appliances.add(Appliance(user_id=U, name="Washer", brand="Dawlance"))  # no date, no price
        self.w_fridge = a.warranties.add(Warranty(user_id=U, appliance_id=self.fridge.id, start_date=date(2024, 3, 14),
                                                  duration_months=24, verified=True))          # expires 2026-03-14
        self.w_ac = a.warranties.add(Warranty(user_id=U, appliance_id=self.ac.id, start_date=date(2025, 6, 1),
                                              duration_months=12))                               # expires 2026-06-01, unverified
        def repair(app_, d, problem, cost, **kw):
            return a.repairs.add(RepairRecord(user_id=U, appliance_id=app_.id, reported_date=d,
                                              problem_description=problem, cost=cost, **kw))
        repair(self.fridge, date(2024, 9, 10), "Compressor noise", "800")
        repair(self.fridge, date(2025, 8, 20), "Door gasket loose", "1200")
        repair(self.fridge, date(2025, 11, 2), "Ice build-up in freezer", "3500",
               resolution="Replaced defrost sensor", parts_replaced=("Defrost sensor",))
        repair(self.ac, date(2025, 12, 5), "Gas top up", "0")                                    # cost not recorded
        self.t_ac = a.maintenance.add(MaintenanceTask(user_id=U, appliance_id=self.ac.id, title="AC service",
                                                      next_due_date=date(2026, 2, 1), interval_months=6))
        self.t_coil = a.maintenance.add(MaintenanceTask(user_id=U, appliance_id=self.fridge.id,
                                                        title="Clean condenser coils", next_due_date=date(2026, 3, 1)))
        a.maintenance.add(MaintenanceTask(user_id=U, appliance_id=self.washer.id, title="Descale drum",
                                          next_due_date=date(2026, 6, 1), interval_months=3))
        self.svc.maintenance.log_service(U, self.ac.id, description="AC service", done_on=date(2025, 12, 1), cost="2500")
        self.svc.maintenance.log_service(U, self.fridge.id, description="Filter check", done_on=date(2026, 1, 15))  # cost 0


class WarrantyServiceTests(ServiceTestCase):
    def test_list_and_status(self):
        views = self.svc.warranties.list_all(U)
        self.assertEqual([v.appliance.name for v in views], ["Fridge", "Split AC"])       # soonest expiry first
        self.assertEqual((views[0].days_remaining, views[0].status), (22, WarrantyStatus.EXPIRING_SOON))
        self.assertEqual((views[1].days_remaining, views[1].status), (101, WarrantyStatus.ACTIVE))
        self.assertTrue(views[1].needs_review and not views[0].needs_review)

    def test_expiring_soon_window(self):
        self.assertEqual([v.appliance.name for v in self.svc.warranties.expiring_soon(U)], ["Fridge"])
        self.assertEqual(len(self.svc.warranties.expiring_soon(U, within_days=120)), 2)

    def test_coverage_states_and_boundaries(self):
        cov = self.svc.warranties.coverage(U, self.fridge.id)
        self.assertEqual(cov.state, CoverageState.COVERED)
        self.assertIn("2026-03-14", cov.reason)
        self.assertIn("22 days", cov.reason)
        self.assertIn("No coverage terms are recorded", cov.reason)       # never overclaims fault coverage
        self.assertTrue(cov.verified)
        self.assertIn("not been confirmed", self.svc.warranties.coverage(U, self.ac.id).reason)
        self.assertEqual(self.svc.warranties.coverage(U, self.washer.id).state, CoverageState.UNKNOWN)
        self.assertEqual(self.svc.warranties.coverage(U, self.fridge.id, date(2026, 3, 14)).state, CoverageState.COVERED)
        ended = self.svc.warranties.coverage(U, self.fridge.id, date(2026, 3, 15))
        self.assertEqual(ended.state, CoverageState.NOT_COVERED)
        self.assertIn("ended on 2026-03-14", ended.reason)
        before = self.svc.warranties.coverage(U, self.fridge.id, date(2024, 3, 13))
        self.assertEqual(before.state, CoverageState.NOT_COVERED)
        self.assertIn("only starts on 2024-03-14", before.reason)

    def test_coverage_notes_are_quoted_when_recorded(self):
        self.app.warranties.update(Warranty(**{**self.w_fridge.__dict__, "coverage_notes": "Compressor 5 years"}))
        self.assertIn("Compressor 5 years", self.svc.warranties.coverage(U, self.fridge.id).reason)

    def test_foreign_user_cannot_query_or_confirm(self):
        with self.assertRaises(NotFoundError):
            self.svc.warranties.coverage("bilal", self.fridge.id)
        with self.assertRaises(NotFoundError):
            self.svc.warranties.confirm("bilal", self.w_ac.id)
        self.assertEqual(self.svc.warranties.list_all("bilal"), [])

    def test_confirm_applies_corrections_and_marks_verified(self):
        w = self.svc.warranties.confirm(U, self.w_ac.id, duration_months=18)
        self.assertTrue(w.verified)
        self.assertEqual(w.expiry_date, date(2026, 12, 1))
        self.assertEqual(self.app.warranties.get(U, self.w_ac.id).duration_months, 18)

    def test_invalid_correction_leaves_the_record_unchanged(self):
        with self.assertRaises(DomainValidationError):
            self.svc.warranties.confirm(U, self.w_ac.id, duration_months=0)
        stored = self.app.warranties.get(U, self.w_ac.id)
        self.assertEqual((stored.duration_months, stored.verified), (12, False))


class MaintenanceServiceTests(ServiceTestCase):
    def test_overview_groups_and_orders(self):
        o = self.svc.maintenance.overview(U)
        self.assertEqual([(v.task.title, v.days_until) for v in o.overdue], [("AC service", -19)])
        self.assertEqual([(v.task.title, v.days_until) for v in o.due_soon], [("Clean condenser coils", 9)])
        self.assertEqual([(v.task.title, v.days_until) for v in o.upcoming], [("Descale drum", 101)])
        self.assertEqual(o.completed, [])

    def test_complete_recurring_task_reschedules_and_logs_cost(self):
        task, log = self.svc.maintenance.complete_task(U, self.t_ac.id, done_on=date(2026, 2, 18), cost="3000", technician="Imran")
        self.assertEqual(task.next_due_date, date(2026, 8, 18))
        self.assertEqual((log.cost, log.task_id, log.technician), (Decimal("3000.00"), self.t_ac.id, "Imran"))
        self.assertEqual(self.app.maintenance.get(U, self.t_ac.id).next_due_date, date(2026, 8, 18))
        self.assertEqual(self.svc.maintenance.overview(U).overdue, [])

    def test_default_completion_date_is_today(self):
        task, log = self.svc.maintenance.complete_task(U, self.t_ac.id)
        self.assertEqual((log.done_date, task.last_done_date), (TODAY, TODAY))

    def test_one_off_task_completes_once(self):
        _, _ = self.svc.maintenance.complete_task(U, self.t_coil.id)
        self.assertEqual(self.svc.maintenance.overview(U).completed[0].status, TaskStatus.COMPLETED)
        with self.assertRaises(DomainValidationError):
            self.svc.maintenance.complete_task(U, self.t_coil.id)

    def test_future_dates_rejected(self):
        with self.assertRaises(DomainValidationError):
            self.svc.maintenance.complete_task(U, self.t_ac.id, done_on=date(2026, 2, 21))
        with self.assertRaises(DomainValidationError):
            self.svc.maintenance.log_service(U, self.ac.id, description="x", done_on=date(2026, 3, 1))

    def test_completion_is_atomic(self):
        class FailingTasks:
            def __init__(self, inner): self.inner = inner
            def get(self, *a): return self.inner.get(*a)
            def update(self, *a): raise RuntimeError("disk full")
        from gharcare.services import MaintenanceService
        svc = MaintenanceService(self.app.appliances, FailingTasks(self.app.maintenance), self.app.maintenance_logs, self.app.db, lambda: TODAY)
        before = len(self.app.maintenance_logs.list_for_user(U))
        with self.assertRaises(RuntimeError):
            svc.complete_task(U, self.t_ac.id, cost="3000")
        self.assertEqual(len(self.app.maintenance_logs.list_for_user(U)), before)            # log rolled back
        self.assertEqual(self.app.maintenance.get(U, self.t_ac.id).next_due_date, date(2026, 2, 1))

    def test_isolation(self):
        with self.assertRaises(NotFoundError):
            self.svc.maintenance.complete_task("bilal", self.t_ac.id)
        with self.assertRaises(NotFoundError):
            self.svc.maintenance.log_service("bilal", self.ac.id, description="x")
        self.assertEqual(self.svc.maintenance.history("bilal"), [])


class ExpenseServiceTests(ServiceTestCase):
    def test_last_six_months_hand_computed(self):
        s = self.svc.expenses.last_months(U, 6)                      # 2025-08-20 .. 2026-02-20, inclusive
        self.assertEqual(s.since, date(2025, 8, 20))
        self.assertEqual(s.total, Decimal("7200.00"))                 # repairs 1200+3500, maintenance 2500
        self.assertEqual(s.by_category, {ExpenseCategory.PURCHASE: Decimal("0.00"),
                                         ExpenseCategory.REPAIR: Decimal("4700.00"),
                                         ExpenseCategory.MAINTENANCE: Decimal("2500.00")})
        self.assertEqual(s.by_month, {"2025-08": Decimal("1200.00"), "2025-11": Decimal("3500.00"), "2025-12": Decimal("2500.00")})
        self.assertEqual([(t.appliance_name, t.total) for t in s.by_appliance],
                         [("Haier Fridge", Decimal("4700.00")), ("Gree Split AC", Decimal("2500.00"))])
        self.assertEqual(len(s.lines), 3)
        self.assertEqual((s.gaps.records_without_cost, s.gaps.purchases_without_price), (2, 1))  # AC repair 0, fridge log 0; washer

    def test_repairs_only_question(self):
        s = self.svc.expenses.last_months(U, 6, categories=[ExpenseCategory.REPAIR])
        self.assertEqual(s.total, Decimal("4700.00"))
        self.assertEqual(s.gaps.records_without_cost, 1)              # only the zero-cost AC repair counts here
        self.assertEqual(s.gaps.purchases_without_price, 0)           # purchases were not asked about

    def test_period_bounds_are_inclusive(self):
        self.assertEqual(self.svc.expenses.summary(U, since=date(2025, 8, 20), until=date(2025, 8, 20)).total, Decimal("1200.00"))
        self.assertEqual(self.svc.expenses.summary(U, since=date(2025, 8, 21), until=date(2025, 11, 2)).total, Decimal("3500.00"))  # upper bound inclusive
        self.assertEqual(self.svc.expenses.summary(U, since=date(2025, 8, 21), until=date(2025, 11, 1)).total, Decimal("0.00"))     # one day earlier excludes it

    def test_purchase_date_bounds_are_inclusive(self):
        ac_day = date(2025, 6, 1)                                        # the AC purchase (150000)
        self.assertEqual(self.svc.expenses.summary(U, since=ac_day, until=ac_day).total, Decimal("150000.00"))
        self.assertEqual(self.svc.expenses.summary(U, since=date(2025, 6, 2), until=date(2025, 6, 30)).total, Decimal("0.00"))
        self.assertEqual(self.svc.expenses.summary(U, since=date(2025, 5, 1), until=date(2025, 5, 31)).total, Decimal("0.00"))

    def test_purchases_and_whole_history(self):
        s = self.svc.expenses.summary(U)
        self.assertEqual(s.by_category[ExpenseCategory.PURCHASE], Decimal("245000.00"))   # 95000 + 150000
        self.assertEqual(s.total, Decimal("245000") + Decimal("5500") + Decimal("2500"))

    def test_appliance_filter(self):
        s = self.svc.expenses.last_months(U, 6, appliance_id=self.fridge.id)
        self.assertEqual(s.total, Decimal("4700.00"))
        with self.assertRaises(NotFoundError):
            self.svc.expenses.summary(U, appliance_id="nope")

    def test_comparison_with_previous_period(self):
        c = self.svc.expenses.compare_recent(U, 6)
        self.assertEqual((c.previous.since, c.previous.until), (date(2025, 2, 20), date(2025, 8, 19)))
        self.assertEqual(c.previous.total, Decimal("150000.00"))      # the AC purchase on 2025-06-01
        self.assertEqual(c.delta, Decimal("-142800.00"))
        self.assertEqual(c.pct_change, Decimal("-95.2"))

    def test_percent_change_is_none_when_previous_is_zero(self):
        self.assertIsNone(self.svc.expenses.compare_recent(U, 1).pct_change)

    def test_decimal_arithmetic_is_exact(self):
        self.app.appliances.add(Appliance(user_id="dec", name="X", id="x1"))
        for cost in ("0.10", "0.20"):
            self.app.repairs.add(RepairRecord(user_id="dec", appliance_id="x1", reported_date=TODAY, problem_description="p", cost=cost))
        self.assertEqual(self.svc.expenses.last_months("dec", 1).total, Decimal("0.30"))

    def test_isolation_and_validation(self):
        self.assertEqual(self.svc.expenses.summary("bilal").total, Decimal("0.00"))
        with self.assertRaises(DomainValidationError):
            self.svc.expenses.summary(U, since=date(2026, 1, 1), until=date(2025, 1, 1))
        with self.assertRaises(DomainValidationError):
            self.svc.expenses.last_months(U, 0)


class ContextServiceTests(ServiceTestCase):
    def test_finds_the_recurring_problem_and_not_unrelated_ones(self):
        ctx = self.svc.context.build(U, self.fridge.id, "ice build up in the freezer again")
        self.assertEqual([s.repair.problem_description for s in ctx.similar_repairs], ["Ice build-up in freezer"])
        self.assertEqual(ctx.similar_repairs[0].days_ago, 110)
        self.assertGreater(ctx.similar_repairs[0].similarity, 0.5)
        self.assertEqual((ctx.repair_count, ctx.total_repair_cost), (3, Decimal("5500.00")))

    def test_facts_text_contains_only_recorded_facts(self):
        facts = self.svc.context.build(U, self.fridge.id, "ice in freezer").facts_text()
        for expected in ("Haier Fridge", "model HRF-306", "2024-03-14", "2026-03-14", "22 days left", "110 days ago",
                         "PKR 3,500.00", "Replaced defrost sensor", "Defrost sensor", "PKR 5,500.00", "Filter check"):
            self.assertIn(expected, facts)
        self.assertNotIn("Overdue", facts)                              # the fridge has no overdue task

    def test_unknowns_are_stated_not_invented(self):
        facts = self.svc.context.build(U, self.washer.id, "drum does not spin").facts_text()
        self.assertIn("Purchase date: not recorded", facts)
        self.assertIn("No warranty record", facts)
        self.assertIn("No similar past repair", facts)
        ac = self.svc.context.build(U, self.ac.id, "gas leak").facts_text()
        self.assertIn("19 days overdue", ac)

    def test_no_problem_text_means_no_similarity_search(self):
        self.assertEqual(self.svc.context.build(U, self.fridge.id).similar_repairs, ())

    def test_foreign_user_rejected(self):
        with self.assertRaises(NotFoundError):
            self.svc.context.build("bilal", self.fridge.id, "ice")

    def test_similarity_crosses_roman_urdu_urdu_and_synonyms(self):
        record = "Ice build-up in freezer"
        for q in ["fridge mein barf jam rahi hai kya karun", "freezer mein barf jam gayi hai", "frost in the freezer", "فریزر میں برف", "freezer mein baraf"]:
            with self.subTest(q=q):
                self.assertGreaterEqual(similarity(q, record), 0.25)
        self.assertGreaterEqual(similarity("fridge se pani tapak raha hai", "Water leaking from fridge"), 0.25)
        self.assertGreaterEqual(similarity("compressor bahut shor kar raha hai", "Noisy compressor"), 0.25)
        self.assertEqual(similarity("door gasket kharab", "ice build-up in freezer"), 0.0)   # no false bridge

    def test_roman_urdu_question_finds_the_english_repair_record(self):
        ctx = self.svc.context.build(U, self.fridge.id, "freezer mein barf jam gayi hai")
        self.assertEqual([s.repair.problem_description for s in ctx.similar_repairs], ["Ice build-up in freezer"])

    def test_similarity_function(self):
        self.assertEqual(similarity("door gasket loose", "ice build-up in freezer"), 0.0)
        self.assertEqual(similarity("", "anything"), 0.0)
        self.assertEqual(similarity("E4 error", "error E4"), 1.0)       # short error codes are kept


class ReportServiceTests(ServiceTestCase):
    def test_report_content(self):
        self.app.documents.add(Document(user_id=U, appliance_id=self.fridge.id, original_filename="fridge.pdf",
                                        doc_type=DocumentType.MANUAL, sha256="ab" * 32, storage_path="p", size_bytes=1))
        self.app.repairs.add(RepairRecord(user_id=U, appliance_id=self.fridge.id, reported_date=date(2026, 1, 5),
                                          problem_description="Noise | rattle", cost="100"))
        md = self.svc.reports.build_markdown(U, months=6)
        for expected in ("# Household report", "Haier Fridge", "2026-03-14", "PKR 7,300.00", "fridge.pdf (manual, Haier Fridge, sha256 abababab)",
                         "Noise \\| rattle",                                  # table cell escaped
                         "The warranty for Gree Split AC has not been confirmed",
                         "record(s) in this period have no recorded cost",
                         "Dawlance Washer has no purchase date or price",
                         "Maintenance overdue: Gree Split AC - AC service",
                         "Warranty ending soon: Haier Fridge on 2026-03-14"):
            self.assertIn(expected, md)

    def test_generate_saves_without_overwriting_and_is_private(self):
        a = self.svc.reports.generate(U)
        b = self.svc.reports.generate(U)
        self.assertNotEqual(a.relative_path, b.relative_path)
        self.assertTrue(a.relative_path.startswith(f"{U}/reports/"))
        self.assertIn("Haier Fridge", self.app.artifacts.read(U, a.relative_path).decode("utf-8"))
        self.assertNotIn("Haier", self.svc.reports.build_markdown("bilal"))

    def test_empty_household(self):
        md = self.svc.reports.build_markdown("newuser")
        self.assertIn("No appliances recorded.", md)
        self.assertIn("No gaps found.", md)
        self.assertIn("PKR 0.00", md)


class TransactionAndMigrationTests(unittest.TestCase):
    def test_nested_repository_calls_commit_or_roll_back_together(self):
        db = SQLiteDatabase(":memory:")
        self.addCleanup(db.close)
        repo = SQLiteApplianceRepository(db)
        with db.transaction():
            repo.add(Appliance(user_id=U, name="A"))
            repo.add(Appliance(user_id=U, name="B"))
        self.assertEqual(len(repo.list_for_user(U)), 2)
        with self.assertRaises(RuntimeError):
            with db.transaction():
                repo.add(Appliance(user_id=U, name="C"))
                raise RuntimeError("boom")
        self.assertEqual(len(repo.list_for_user(U)), 2)                    # C was rolled back

    def test_v2_database_upgrades_to_v3_keeping_data(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "v2.db"
            old = sqlite3.connect(path)
            old.executescript(_SCHEMA_V1 + _SCHEMA_V2 + "PRAGMA user_version = 2;")
            old.execute("INSERT INTO appliances (id, user_id, name, created_at) VALUES ('a1','ayesha','Fridge','2025-01-01T00:00:00+00:00')")
            old.commit(); old.close()
            db = SQLiteDatabase(path)
            self.assertEqual(db._conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(db._conn.execute("SELECT COUNT(*) FROM maintenance_logs").fetchone()[0], 0)
            self.assertEqual(db._conn.execute("SELECT name FROM appliances").fetchone()[0], "Fridge")
            db.close()


if __name__ == "__main__":
    unittest.main()
