import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance, DocumentType, DraftStatus, FieldConfidence
from gharcare.errors import DomainValidationError, NoExtractableTextError, NotFoundError
from gharcare.extraction import ExtractionPipeline, ExtractionService
from gharcare.extraction.factory import build_extraction
from gharcare.rag import build_rag
from gharcare.repositories import SQLiteDatabase
from gharcare.repositories.sqlite import _SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3, SCHEMA_VERSION
from gharcare.retrieval import HashingEmbedder, InMemoryVectorStore

DOCS = Path(__file__).parent.parent / "examples" / "sample_docs"
U = "ayesha"


class ExtractionTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.app = AppContainer.build(Settings(data_dir=tmp.name))
        self.addCleanup(self.app.close)
        self.rag = build_rag(self.app, HashingEmbedder(), InMemoryVectorStore())
        self.svc = build_extraction(self.app)
        self.fridge = self.app.appliances.add(Appliance(user_id=U, name="Fridge", brand="Haier"))

    def upload(self, sample, doc_type, user=U, appliance=None, text=None):
        content = text.encode() if text is not None else (DOCS / sample).read_bytes()
        return self.rag.ingestion.ingest(user, sample, content, doc_type, appliance_id=appliance.id if appliance else None).document

    @staticmethod
    def values(draft, **overrides):
        return {**{n: f.value for n, f in draft.fields.items()}, **overrides}


class ProposeTests(ExtractionTestCase):
    def test_propose_stores_a_pending_draft_and_the_raw_ai_output(self):
        doc = self.upload("warranty_card_SYNTHETIC.txt", "warranty_card", appliance=self.fridge)
        draft = self.svc.propose(U, doc.id)
        self.assertEqual((draft.status, draft.doc_type), (DraftStatus.PENDING, DocumentType.WARRANTY_CARD))
        self.assertEqual(draft.fields["warranty_duration_months"].value, "24")
        raw = self.app.artifacts.load_latest_extracted(U, f"{doc.id}-fields")
        self.assertEqual((raw["kind"], raw["draft_id"]), ("ai_draft", draft.id))
        self.assertEqual(self.svc.propose(U, doc.id).id, draft.id)                   # idempotent
        self.assertEqual([d.id for d in self.svc.list_pending(U)], [draft.id])

    def test_rejections(self):
        manual = self.upload("m.txt", "manual", text="Safety warnings. Read this manual first.")
        with self.assertRaisesRegex(DomainValidationError, "receipts, warranty cards and service invoices"):
            self.svc.propose(U, manual.id)
        scanned = self.upload("scan.txt", "receipt", text="   \n  ")
        with self.assertRaises(NoExtractableTextError):
            self.svc.propose(U, scanned.id)
        doc = self.upload("receipt_SYNTHETIC.txt", "receipt")
        with self.assertRaises(NotFoundError):
            self.svc.propose("bilal", doc.id)

    def test_drafts_are_private_to_their_owner(self):
        draft = self.svc.propose(U, self.upload("receipt_SYNTHETIC.txt", "receipt").id)
        for call in (lambda: self.svc.get("bilal", draft.id), lambda: self.svc.confirm("bilal", draft.id, {}), lambda: self.svc.reject("bilal", draft.id)):
            with self.assertRaises(NotFoundError):
                call()
        self.assertEqual(self.svc.list_pending("bilal"), [])


class ConfirmTests(ExtractionTestCase):
    def test_confirming_a_warranty_card_creates_a_verified_warranty(self):
        doc = self.upload("warranty_card_SYNTHETIC.txt", "warranty_card", appliance=self.fridge)
        draft = self.svc.propose(U, doc.id)
        result = self.svc.confirm(U, draft.id, self.values(draft))                    # all fields HIGH: no extra review needed
        w = self.app.warranties.get(U, result.created["warranty"])
        self.assertEqual((str(w.expiry_date), w.verified, w.source_document_id, w.appliance_id, w.provider),
                         ("2026-03-14", True, doc.id, self.fridge.id, "Haier Pakistan Service Centre"))
        self.assertEqual(self.app.extraction_drafts.get(U, draft.id).status, DraftStatus.CONFIRMED)
        self.assertEqual(self.app.appliances.get(U, self.fridge.id).model, "HRF-306")   # blanks filled from the card
        final = self.app.artifacts.load_latest_extracted(U, f"{doc.id}-fields")
        self.assertEqual((final["kind"], final["final"]["warranty_duration_months"]), ("user_confirmed", "24"))
        self.assertEqual(final["ai_draft"]["warranty_duration_months"], "24")

    def test_uncertain_fields_must_be_reviewed_but_corrections_count_as_review(self):
        draft = self.svc.propose(U, self.upload("service_invoice_SYNTHETIC.txt", "service_invoice", appliance=self.fridge).id)
        self.assertEqual(draft.fields["service_date"].confidence, FieldConfidence.MEDIUM)
        with self.assertRaisesRegex(DomainValidationError, "Service date"):
            self.svc.confirm(U, draft.id, self.values(draft))
        self.assertEqual(self.app.repairs.list_for_user(U), [])                         # nothing saved
        result = self.svc.confirm(U, draft.id, self.values(draft), reviewed_fields=["service_date"])
        r = self.app.repairs.get(U, result.created["repair"])
        self.assertEqual((str(r.reported_date), r.cost, r.parts_replaced, r.technician), ("2025-12-05", Decimal("4500.00"), ("Capacitor",), "Imran Ali"))
        self.assertEqual(r.problem_description, "AC not cooling, gas low")
        # a user who CHANGES the uncertain value has implicitly reviewed it
        d2 = self.svc.propose(U, self.upload("service2.txt", "service_invoice", appliance=self.fridge, text=(DOCS / "service_invoice_SYNTHETIC.txt").read_text() + "\n").id)
        self.svc.confirm(U, d2.id, self.values(d2, service_date="2025-05-12"))

    def test_receipt_updates_the_appliance_and_reports_what_changed(self):
        self.app.appliances.update(Appliance(**{**self.fridge.__dict__, "purchase_date": __import__("datetime").date(2023, 1, 1)}))
        draft = self.svc.propose(U, self.upload("receipt_SYNTHETIC.txt", "receipt", appliance=self.fridge).id)
        result = self.svc.confirm(U, draft.id, self.values(draft))
        self.assertEqual(result.changes, ("purchase date: 2023-01-01 -> 2024-03-14", "purchase price: not set -> 95000.00"))
        a = self.app.appliances.get(U, self.fridge.id)
        self.assertEqual((str(a.purchase_date), a.purchase_price, a.model, a.serial_number), ("2024-03-14", Decimal("95000.00"), "HRF-306", "HF2403145521"))

    def test_an_unlinked_receipt_creates_a_new_appliance(self):
        doc = self.upload("receipt_SYNTHETIC.txt", "receipt")
        draft = self.svc.propose(U, doc.id)
        self.svc.confirm(U, draft.id, self.values(draft))
        created = [a for a in self.app.appliances.list_for_user(U) if a.id != self.fridge.id][0]
        self.assertEqual((created.name, created.brand, created.display_name, str(created.purchase_date)), ("Refrigerator", "Haier", "Haier Refrigerator", "2024-03-14"))
        self.assertEqual(self.app.documents.get(U, doc.id).appliance_id, created.id)     # the document is now linked

    def test_documents_for_the_same_serial_share_one_appliance(self):
        r = self.svc.propose(U, self.upload("receipt_SYNTHETIC.txt", "receipt").id)
        self.svc.confirm(U, r.id, self.values(r))
        after_receipt = len(self.app.appliances.list_for_user(U))
        w = self.svc.propose(U, self.upload("warranty_card_SYNTHETIC.txt", "warranty_card").id)       # same serial, unlinked
        result = self.svc.confirm(U, w.id, self.values(w))
        self.assertEqual(len(self.app.appliances.list_for_user(U)), after_receipt)                    # no duplicate appliance
        receipt_appliance = [a for a in self.app.appliances.list_for_user(U) if a.serial_number == "HF2403145521"][0]
        self.assertEqual(self.app.warranties.get(U, result.created["warranty"]).appliance_id, receipt_appliance.id)

    def test_a_different_serial_with_the_same_model_is_a_different_appliance(self):
        r = self.svc.propose(U, self.upload("receipt_SYNTHETIC.txt", "receipt").id)
        self.svc.confirm(U, r.id, self.values(r))
        before = len(self.app.appliances.list_for_user(U))
        text = (DOCS / "warranty_card_SYNTHETIC.txt").read_text().replace("HF2403145521", "HF9999999999")
        w = self.svc.propose(U, self.upload("w2.txt", "warranty_card", text=text).id)
        self.svc.confirm(U, w.id, self.values(w))
        self.assertEqual(len(self.app.appliances.list_for_user(U)), before + 1)

    def test_without_a_serial_brand_and_model_match_only_when_unique(self):
        no_serial = (DOCS / "warranty_card_SYNTHETIC.txt").read_text().replace("Serial No: HF2403145521\n", "")
        r = self.svc.propose(U, self.upload("receipt_SYNTHETIC.txt", "receipt").id)
        self.svc.confirm(U, r.id, self.values(r))
        w = self.svc.propose(U, self.upload("w3.txt", "warranty_card", text=no_serial).id)
        self.svc.confirm(U, w.id, self.values(w))                                                     # unique brand+model match: reused
        self.assertEqual(len([a for a in self.app.appliances.list_for_user(U) if a.model == "HRF-306"]), 1)
        self.app.appliances.add(Appliance(user_id=U, name="Fridge 2", brand="Haier", model="HRF-306"))   # now two match
        w2 = self.svc.propose(U, self.upload("w4.txt", "warranty_card", text=no_serial + "\n").id)
        with self.assertRaisesRegex(DomainValidationError, "more than one of your appliances"):
            self.svc.confirm(U, w2.id, self.values(w2))

    def test_validation_blocks_bad_input_without_saving_anything(self):
        draft = self.svc.propose(U, self.upload("receipt_SYNTHETIC.txt", "receipt", appliance=self.fridge).id)
        cases = [({"purchase_date": "14/03/2024"}, "Purchase date must be a date like 2024-03-14"), ({"total_amount": "lots"}, "Total amount"),
                 ({"nonsense": "x"}, "unknown field"), ({"purchase_date": None, "total_amount": None}, "missing required")]
        for override, message in cases:
            with self.subTest(override=override), self.assertRaisesRegex(DomainValidationError, message):
                self.svc.confirm(U, draft.id, self.values(draft, **override))
        self.assertEqual(self.app.extraction_drafts.get(U, draft.id).status, DraftStatus.PENDING)
        self.assertIsNone(self.app.appliances.get(U, self.fridge.id).purchase_date)

    def test_missing_required_warranty_fields(self):
        draft = self.svc.propose(U, self.upload("warranty_card_SYNTHETIC.txt", "warranty_card", appliance=self.fridge).id)
        with self.assertRaisesRegex(DomainValidationError, "Warranty duration"):
            self.svc.confirm(U, draft.id, self.values(draft, warranty_duration_months=None))

    def test_confirmation_is_atomic(self):
        class FailingWarranties:
            def add(self, *a, **k): raise RuntimeError("disk full")
        svc = ExtractionService(self.app.documents, self.app.appliances, FailingWarranties(), self.app.repairs, self.app.extraction_drafts,
                                self.app.artifacts, self.app.db, ExtractionPipeline())
        doc = self.upload("warranty_card_SYNTHETIC.txt", "warranty_card", user="atomic")          # unlinked: a new appliance is created first
        draft = svc.propose("atomic", doc.id)
        with self.assertRaises(RuntimeError):
            svc.confirm("atomic", draft.id, self.values(draft))
        self.assertEqual(self.app.appliances.list_for_user("atomic"), [])                         # the new appliance was rolled back
        self.assertIsNone(self.app.documents.get("atomic", doc.id).appliance_id)
        self.assertEqual(self.app.extraction_drafts.get("atomic", draft.id).status, DraftStatus.PENDING)

    def test_an_unlinked_service_invoice_asks_which_appliance(self):
        draft = self.svc.propose("sam", self.upload("service_invoice_SYNTHETIC.txt", "service_invoice", user="sam").id)
        with self.assertRaisesRegex(DomainValidationError, "choose which appliance"):
            self.svc.confirm("sam", draft.id, self.values(draft), reviewed_fields=["service_date"])
        own = self.app.appliances.add(Appliance(user_id="sam", name="AC", brand="Gree"))
        self.assertIn("repair", self.svc.confirm("sam", draft.id, self.values(draft), reviewed_fields=["service_date"], appliance_id=own.id).created)

    def test_a_draft_resolves_only_once(self):
        draft = self.svc.propose(U, self.upload("warranty_card_SYNTHETIC.txt", "warranty_card", appliance=self.fridge).id)
        self.svc.confirm(U, draft.id, self.values(draft))
        for call in (lambda: self.svc.confirm(U, draft.id, self.values(draft)), lambda: self.svc.reject(U, draft.id)):
            with self.assertRaisesRegex(DomainValidationError, "already confirmed"):
                call()
        self.assertEqual(len(self.app.warranties.list_for_user(U)), 1)

    def test_reject(self):
        draft = self.svc.propose(U, self.upload("receipt_SYNTHETIC.txt", "receipt").id)
        self.assertEqual(self.svc.reject(U, draft.id).status, DraftStatus.REJECTED)
        self.assertEqual(self.svc.list_pending(U), [])
        with self.assertRaises(DomainValidationError):
            self.svc.confirm(U, draft.id, self.values(draft))


class AccuracyTests(ExtractionTestCase):
    def test_accuracy_is_measured_from_user_corrections(self):
        self.assertIsNone(self.svc.accuracy(U).accuracy)
        draft = self.svc.propose(U, self.upload("warranty_card_SYNTHETIC.txt", "warranty_card", appliance=self.fridge).id)
        self.svc.confirm(U, draft.id, self.values(draft, warranty_duration_months="36"))              # the user fixes one field
        acc = self.svc.accuracy(U)
        self.assertEqual((acc.drafts, acc.fields_read, acc.fields_correct, acc.fields_missed), (1, 7, 6, 0))
        self.assertAlmostEqual(acc.accuracy, 6 / 7)
        self.assertEqual(acc.by_confidence, {"high": (6, 7)})
        self.assertEqual(acc.by_field["warranty_duration_months"], (0, 1))

    def test_fields_the_draft_missed_are_counted_separately(self):
        text = (DOCS / "receipt_SYNTHETIC.txt").read_text().replace("Serial No: HF2403145521\n", "")
        draft = self.svc.propose(U, self.upload("r.txt", "receipt", appliance=self.fridge, text=text).id)
        self.assertIsNone(draft.fields["serial_number"].value)
        self.svc.confirm(U, draft.id, self.values(draft, serial_number="SN-777"))
        acc = self.svc.accuracy(U)
        self.assertEqual((acc.fields_read, acc.fields_correct, acc.fields_missed), (6, 6, 1))
        self.assertEqual(self.svc.accuracy("bilal").drafts, 0)


class Migration4Tests(unittest.TestCase):
    def test_v3_database_upgrades_and_keeps_data(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "v3.db"
            old = sqlite3.connect(path)
            old.executescript(_SCHEMA_V1 + _SCHEMA_V2 + _SCHEMA_V3 + "PRAGMA user_version = 3;")
            old.execute("INSERT INTO appliances (id, user_id, name, created_at) VALUES ('a1','ayesha','Fridge','2025-01-01T00:00:00+00:00')")
            old.commit(); old.close()
            db = SQLiteDatabase(path)
            self.assertEqual(db._conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(db._conn.execute("SELECT COUNT(*) FROM extraction_drafts").fetchone()[0], 0)
            self.assertEqual(db._conn.execute("SELECT name FROM appliances").fetchone()[0], "Fridge")
            db.close()


if __name__ == "__main__":
    unittest.main()
