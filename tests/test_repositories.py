import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from gharcare.domain import (
    Appliance, Document, DocumentType, IndexStatus, MaintenanceTask, RepairRecord, Warranty,
)
from gharcare.errors import NotFoundError
from gharcare.repositories import (
    SQLiteApplianceRepository, SQLiteDatabase, SQLiteDocumentRepository,
    SQLiteMaintenanceRepository, SQLiteRepairRepository, SQLiteWarrantyRepository,
)


class RepoTestCase(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteDatabase(":memory:")
        self.appliances = SQLiteApplianceRepository(self.db)
        self.warranties = SQLiteWarrantyRepository(self.db)
        self.tasks = SQLiteMaintenanceRepository(self.db)
        self.repairs = SQLiteRepairRepository(self.db)
        self.documents = SQLiteDocumentRepository(self.db)
        self.fridge = self.appliances.add(Appliance(
            user_id="ayesha", name="Fridge", brand="Haier", model="HRF-306",
            purchase_date=date(2024, 3, 14), purchase_price="95000", location="Kitchen"))

    def tearDown(self):
        self.db.close()

    def make_doc(self, **kw):
        base = dict(user_id="ayesha", original_filename="manual.pdf", doc_type=DocumentType.MANUAL,
                    sha256="b" * 64, storage_path="ayesha/originals/bb/x.pdf", size_bytes=100,
                    appliance_id=self.fridge.id)
        return Document(**(base | kw))


class ApplianceRepoTests(RepoTestCase):
    def test_round_trip_preserves_every_field(self):
        self.assertEqual(self.appliances.get("ayesha", self.fridge.id), self.fridge)
        self.assertEqual(self.appliances.get("ayesha", self.fridge.id).purchase_price, Decimal("95000.00"))

    def test_update(self):
        updated = Appliance(**{**self.fridge.__dict__, "location": "Store room"})
        self.appliances.update(updated)
        self.assertEqual(self.appliances.get("ayesha", self.fridge.id).location, "Store room")

    def test_update_and_delete_of_missing_raise(self):
        ghost = Appliance(user_id="ayesha", name="Ghost")
        with self.assertRaises(NotFoundError):
            self.appliances.update(ghost)
        with self.assertRaises(NotFoundError):
            self.appliances.delete("ayesha", "nope")


class IsolationTests(RepoTestCase):
    def test_other_users_cannot_read_list_modify_or_delete(self):
        self.assertEqual(self.appliances.list_for_user("bilal"), [])
        with self.assertRaises(NotFoundError):
            self.appliances.get("bilal", self.fridge.id)
        with self.assertRaises(NotFoundError):
            self.appliances.delete("bilal", self.fridge.id)
        self.assertEqual(len(self.appliances.list_for_user("ayesha")), 1)  # still there

    def test_cannot_attach_records_to_someone_elses_appliance(self):
        with self.assertRaises(NotFoundError):
            self.warranties.add(Warranty(user_id="bilal", appliance_id=self.fridge.id,
                                         start_date=date(2024, 1, 1), duration_months=12))
        with self.assertRaises(NotFoundError):
            self.repairs.add(RepairRecord(user_id="bilal", appliance_id=self.fridge.id,
                                          reported_date=date(2025, 1, 1), problem_description="x"))
        with self.assertRaises(NotFoundError):
            self.documents.add(self.make_doc(user_id="bilal"))

    def test_records_are_invisible_across_users(self):
        w = self.warranties.add(Warranty(user_id="ayesha", appliance_id=self.fridge.id,
                                         start_date=date(2024, 3, 14), duration_months=24))
        with self.assertRaises(NotFoundError):
            self.warranties.get("bilal", w.id)
        self.assertEqual(self.warranties.list_for_user("bilal"), [])


class ChildRecordTests(RepoTestCase):
    def test_warranty_round_trip_and_status(self):
        w = self.warranties.add(Warranty(
            user_id="ayesha", appliance_id=self.fridge.id, start_date=date(2024, 3, 14),
            duration_months=24, provider="Haier Pakistan", coverage_notes="Compressor 5 years",
            verified=True))
        loaded = self.warranties.get("ayesha", w.id)
        self.assertEqual(loaded, w)
        self.assertTrue(loaded.verified)
        self.assertEqual(loaded.expiry_date, date(2026, 3, 14))

    def test_maintenance_orders_by_due_date_with_completed_last(self):
        a = MaintenanceTask(user_id="ayesha", appliance_id=self.fridge.id, title="Later", next_due_date=date(2026, 9, 1))
        b = MaintenanceTask(user_id="ayesha", appliance_id=self.fridge.id, title="Sooner", next_due_date=date(2026, 6, 1))
        c = MaintenanceTask(user_id="ayesha", appliance_id=self.fridge.id, title="Done", next_due_date=None,
                            last_done_date=date(2026, 1, 1))
        for t in (a, b, c):
            self.tasks.add(t)
        self.assertEqual([t.title for t in self.tasks.list_for_user("ayesha")], ["Sooner", "Later", "Done"])

    def test_maintenance_mark_done_persists(self):
        t = self.tasks.add(MaintenanceTask(user_id="ayesha", appliance_id=self.fridge.id, title="Clean coils",
                                           next_due_date=date(2026, 4, 1), interval_months=6))
        self.tasks.update(t.mark_done(date(2026, 4, 5)))
        self.assertEqual(self.tasks.get("ayesha", t.id).next_due_date, date(2026, 10, 5))

    def test_repair_round_trip_keeps_unknown_warranty_coverage_as_none(self):
        r = self.repairs.add(RepairRecord(
            user_id="ayesha", appliance_id=self.fridge.id, reported_date=date(2025, 11, 2),
            problem_description="Ice build-up in freezer", resolution="Replaced sensor",
            technician="Imran", parts_replaced=("Defrost sensor", "Thermostat"), cost="3500"))
        loaded = self.repairs.get("ayesha", r.id)
        self.assertEqual(loaded, r)
        self.assertIsNone(loaded.covered_by_warranty)
        self.assertEqual(loaded.parts_replaced, ("Defrost sensor", "Thermostat"))

    def test_repair_date_filters_and_ordering(self):
        for d, cost in ((date(2025, 1, 10), "1000"), (date(2025, 6, 10), "2000"), (date(2025, 11, 2), "3500")):
            self.repairs.add(RepairRecord(user_id="ayesha", appliance_id=self.fridge.id, reported_date=d,
                                          problem_description="issue", cost=cost))
        everything = self.repairs.list_for_user("ayesha")
        self.assertEqual([r.reported_date for r in everything],
                         [date(2025, 11, 2), date(2025, 6, 10), date(2025, 1, 10)])
        recent = self.repairs.list_for_user("ayesha", since=date(2025, 6, 10), until=date(2025, 11, 2))
        self.assertEqual(sum(r.cost for r in recent), Decimal("5500.00"))  # bounds are inclusive
        self.assertEqual(len(self.repairs.list_for_user("ayesha", since=date(2025, 12, 1))), 0)

    def test_deleting_appliance_cascades_but_keeps_documents(self):
        self.warranties.add(Warranty(user_id="ayesha", appliance_id=self.fridge.id,
                                     start_date=date(2024, 3, 14), duration_months=24))
        self.repairs.add(RepairRecord(user_id="ayesha", appliance_id=self.fridge.id,
                                      reported_date=date(2025, 1, 1), problem_description="x"))
        self.tasks.add(MaintenanceTask(user_id="ayesha", appliance_id=self.fridge.id, title="t",
                                       next_due_date=date(2026, 1, 1)))
        doc = self.documents.add(self.make_doc())
        self.appliances.delete("ayesha", self.fridge.id)
        self.assertEqual(self.warranties.list_for_user("ayesha"), [])
        self.assertEqual(self.repairs.list_for_user("ayesha"), [])
        self.assertEqual(self.tasks.list_for_user("ayesha"), [])
        kept = self.documents.get("ayesha", doc.id)
        self.assertIsNone(kept.appliance_id)  # original record survives, just unlinked


class DocumentRepoTests(RepoTestCase):
    def test_filters_hash_lookup_and_status_update(self):
        manual = self.documents.add(self.make_doc())
        receipt = self.documents.add(self.make_doc(doc_type=DocumentType.RECEIPT, sha256="c" * 64,
                                                   original_filename="receipt.pdf"))
        self.assertEqual({d.id for d in self.documents.list_for_user("ayesha", doc_type=DocumentType.RECEIPT)},
                         {receipt.id})
        self.assertEqual([d.id for d in self.documents.find_by_hash("ayesha", "b" * 64)], [manual.id])
        self.assertEqual(self.documents.find_by_hash("bilal", "b" * 64), [])
        from dataclasses import replace
        self.documents.update(replace(manual, index_status=IndexStatus.INDEXED))
        self.assertEqual(self.documents.get("ayesha", manual.id).index_status, IndexStatus.INDEXED)

    def test_same_file_can_be_linked_to_two_appliances(self):
        ac2 = self.appliances.add(Appliance(user_id="ayesha", name="AC 2", brand="Gree"))
        self.documents.add(self.make_doc())
        self.documents.add(self.make_doc(appliance_id=ac2.id))
        self.assertEqual(len(self.documents.find_by_hash("ayesha", "b" * 64)), 2)


class PersistenceTests(unittest.TestCase):
    def test_data_survives_reopening_the_database_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "gharcare.db"   # parent folder is created on demand
            db = SQLiteDatabase(path)
            a = SQLiteApplianceRepository(db).add(Appliance(user_id="ayesha", name="Washer", brand="Dawlance"))
            db.close()
            db2 = SQLiteDatabase(path)
            self.assertEqual(SQLiteApplianceRepository(db2).get("ayesha", a.id), a)
            db2.close()


if __name__ == "__main__":
    unittest.main()
