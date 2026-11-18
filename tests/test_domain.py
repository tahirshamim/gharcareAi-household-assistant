import dataclasses
import unittest
from datetime import date, timedelta
from decimal import Decimal

from gharcare.domain import (
    Appliance, Document, DocumentType, MaintenanceTask, RepairRecord, TaskStatus, Warranty,
    WarrantyStatus, add_months,
)
from gharcare.errors import DomainValidationError

SHA = "a" * 64


class AddMonthsTests(unittest.TestCase):
    def test_clamps_to_month_end(self):
        self.assertEqual(add_months(date(2024, 1, 31), 1), date(2024, 2, 29))  # leap year
        self.assertEqual(add_months(date(2023, 1, 31), 1), date(2023, 2, 28))

    def test_crosses_year_boundary(self):
        self.assertEqual(add_months(date(2024, 12, 15), 1), date(2025, 1, 15))

    def test_whole_years(self):
        self.assertEqual(add_months(date(2024, 3, 14), 24), date(2026, 3, 14))


class ApplianceTests(unittest.TestCase):
    def test_normalises_text_and_money(self):
        a = Appliance(user_id="u1", name="  Fridge ", brand=" Haier", model="  ", purchase_price=1500.5)
        self.assertEqual(a.name, "Fridge")
        self.assertEqual(a.brand, "Haier")
        self.assertIsNone(a.model)
        self.assertEqual(a.purchase_price, Decimal("1500.50"))
        self.assertEqual(a.display_name, "Haier Fridge")

    def test_rejects_bad_input(self):
        for kwargs in (
            dict(user_id="u1", name="  "),
            dict(user_id="../etc", name="x"),
            dict(user_id="u1", name="x", purchase_price=-1),
            dict(user_id="u1", name="x", purchase_price="abc"),
            dict(user_id="u1", name="x", purchase_date=date.today() + timedelta(days=30)),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(DomainValidationError):
                Appliance(**kwargs)

    def test_is_immutable(self):
        a = Appliance(user_id="u1", name="Fridge")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a.name = "Other"


class WarrantyTests(unittest.TestCase):
    def setUp(self):
        self.w = Warranty(user_id="u1", appliance_id="a1", start_date=date(2024, 3, 14), duration_months=24)

    def test_expiry(self):
        self.assertEqual(self.w.expiry_date, date(2026, 3, 14))

    def test_exact_edges(self):
        self.assertEqual(self.w.status(date(2025, 1, 1)), WarrantyStatus.ACTIVE)
        self.assertEqual(self.w.status(date(2026, 2, 12)), WarrantyStatus.EXPIRING_SOON)  # 30 days left
        self.assertEqual(self.w.status(date(2026, 2, 11)), WarrantyStatus.ACTIVE)          # 31 days left
        self.assertEqual(self.w.status(date(2026, 3, 14)), WarrantyStatus.EXPIRING_SOON)   # 0 days left
        self.assertEqual(self.w.status(date(2026, 3, 15)), WarrantyStatus.EXPIRED)

    def test_invalid_duration(self):
        for months in (0, -3, 121, True, 2.5):
            with self.subTest(months=months), self.assertRaises(DomainValidationError):
                Warranty(user_id="u1", appliance_id="a1", start_date=date(2024, 1, 1), duration_months=months)


class MaintenanceTests(unittest.TestCase):
    def test_status(self):
        today = date(2026, 5, 1)
        def task(due): return MaintenanceTask(user_id="u1", appliance_id="a1", title="AC service", next_due_date=due)
        self.assertEqual(task(date(2026, 4, 30)).status(today), TaskStatus.OVERDUE)
        self.assertEqual(task(date(2026, 5, 1)).status(today), TaskStatus.DUE_SOON)
        self.assertEqual(task(date(2026, 5, 15)).status(today), TaskStatus.DUE_SOON)
        self.assertEqual(task(date(2026, 5, 16)).status(today), TaskStatus.UPCOMING)

    def test_recurring_reschedules_from_completion_date(self):
        t = MaintenanceTask(user_id="u1", appliance_id="a1", title="AC service",
                            next_due_date=date(2026, 4, 1), interval_months=6)
        done = t.mark_done(date(2026, 4, 10))
        self.assertEqual(done.next_due_date, date(2026, 10, 10))
        self.assertEqual(done.last_done_date, date(2026, 4, 10))
        self.assertEqual(t.next_due_date, date(2026, 4, 1))  # original untouched

    def test_one_off_becomes_completed(self):
        t = MaintenanceTask(user_id="u1", appliance_id="a1", title="Install stabiliser", next_due_date=date(2026, 4, 1))
        done = t.mark_done(date(2026, 4, 2))
        self.assertIsNone(done.next_due_date)
        self.assertEqual(done.status(date(2030, 1, 1)), TaskStatus.COMPLETED)

    def test_invariant_blocks_dateless_active_task(self):
        with self.assertRaises(DomainValidationError):
            MaintenanceTask(user_id="u1", appliance_id="a1", title="x", next_due_date=None)
        with self.assertRaises(DomainValidationError):
            MaintenanceTask(user_id="u1", appliance_id="a1", title="x", next_due_date=None,
                            interval_months=6, last_done_date=date(2026, 1, 1))


class RepairAndDocumentTests(unittest.TestCase):
    def test_repair_defaults_and_cleaning(self):
        r = RepairRecord(user_id="u1", appliance_id="a1", reported_date=date(2025, 11, 2),
                         problem_description="Ice build-up", parts_replaced=["  Defrost sensor ", ""], cost="3500")
        self.assertEqual(r.parts_replaced, ("Defrost sensor",))
        self.assertEqual(r.cost, Decimal("3500.00"))
        self.assertIsNone(r.covered_by_warranty)  # unknown stays unknown

    def test_repair_rejects_negative_cost_and_blank_problem(self):
        with self.assertRaises(DomainValidationError):
            RepairRecord(user_id="u1", appliance_id="a1", reported_date=date(2025, 1, 1),
                         problem_description="x", cost=-5)
        with self.assertRaises(DomainValidationError):
            RepairRecord(user_id="u1", appliance_id="a1", reported_date=date(2025, 1, 1), problem_description=" ")

    def test_document_validation(self):
        d = Document(user_id="u1", original_filename="manual.pdf", doc_type="manual",
                     sha256=SHA, storage_path="u1/originals/aa/x.pdf", size_bytes=10)
        self.assertEqual(d.doc_type, DocumentType.MANUAL)
        for kwargs in (dict(sha256="xyz"), dict(size_bytes=-1), dict(doc_type="banana")):
            args = dict(user_id="u1", original_filename="m.pdf", doc_type="manual", sha256=SHA,
                        storage_path="p", size_bytes=1) | kwargs
            with self.subTest(kwargs=kwargs), self.assertRaises(DomainValidationError):
                Document(**args)


if __name__ == "__main__":
    unittest.main()
