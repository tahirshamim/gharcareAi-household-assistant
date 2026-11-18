import tempfile
import unittest
from datetime import date
from pathlib import Path

from gharcare.api.service import AppService
from gharcare.config import Settings
from gharcare.errors import DomainValidationError, LLMUnavailableError, NotFoundError
from gharcare.llm import FakeLLMProvider, UnavailableLLMProvider
from gharcare.rag import rebuild_vector_index

DOCS = Path(__file__).parent.parent / "examples" / "sample_docs"
MANUAL = DOCS / "xr200_sample_manual_SYNTHETIC.txt"
TODAY = date(2026, 2, 20)
U = "ayesha"


def model(messages):
    system, user = messages[0].content, messages[-1].content
    if system.startswith("You are a strict fact checker"):
        return '{"supported": true, "unsupported_claims": []}'
    if system.startswith("Classify"):
        return "troubleshooting"
    if system.startswith("You extract fields"):
        return "{}"
    return "NOT_FOUND" if "litres" in user.split("MANUAL EXCERPTS")[0] else "Leave the doors open for about 6 hours [1]."


class AppServiceTestCase(unittest.TestCase):
    def make(self, data_dir=None, llm="fake", rebuild=True):
        if data_dir is None:
            tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            self.addCleanup(tmp.cleanup)
            data_dir = tmp.name
        settings = Settings(data_dir=data_dir, embedder="hashing", vector_store="memory")
        service = AppService.create(settings, llm=FakeLLMProvider(responder=model) if llm == "fake" else None,
                                    clock=lambda: TODAY, rebuild_index=rebuild, workflow_backend="local")
        self.addCleanup(service.close)
        self.data_dir = data_dir
        return service

    def setUp(self):
        self.s = self.make()
        self.fridge = self.s.appliance_create(U, {"name": "Fridge", "brand": "Haier", "purchase_date": "2024-03-14", "purchase_price": "95000"})

    def upload_manual(self, user=U, appliance=None):
        return self.s.document_upload(user, MANUAL.name, MANUAL.read_bytes(), "manual", (appliance or self.fridge)["id"])


class AppliancesAndDocumentsTests(AppServiceTestCase):
    def test_appliance_crud_and_validation(self):
        self.assertEqual(self.fridge["display_name"], "Haier Fridge")
        self.assertEqual(self.fridge["purchase_price"], "95000.00")
        updated = self.s.appliance_update(U, self.fridge["id"], {"location": "Kitchen", "purchase_price": None})
        self.assertEqual((updated["location"], updated["purchase_price"]), ("Kitchen", None))
        with self.assertRaisesRegex(DomainValidationError, "not a valid date"):
            self.s.appliance_create(U, {"name": "X", "purchase_date": "14/03/2024"})
        with self.assertRaises(DomainValidationError):
            self.s.appliance_create(U, {"name": "  "})
        self.s.appliance_delete(U, self.fridge["id"])
        self.assertEqual(self.s.appliances(U), [])

    def test_upload_lists_and_hides_internal_fields(self):
        result = self.upload_manual()
        self.assertTrue(result["indexed"] and result["chunk_count"] > 0)
        doc = self.s.documents(U)[0]
        self.assertEqual(set(doc), {"id", "filename", "doc_type", "appliance_id", "index_status", "size_bytes", "uploaded_at"})   # no paths/hashes
        self.assertTrue(self.upload_manual()["duplicate"])
        blank = self.s.document_upload(U, "scan.txt", b"  \n ", "receipt")
        self.assertFalse(blank["indexed"])
        self.assertIn("OCR", blank["warning"])
        with self.assertRaises(DomainValidationError):
            self.s.document_upload(U, "photo.jpg", b"\xff\xd8", "receipt")

    def test_download_returns_the_original_and_users_are_isolated(self):
        doc = self.upload_manual()["document"]
        name, content = self.s.document_file(U, doc["id"])
        self.assertEqual((name, content), (MANUAL.name, MANUAL.read_bytes()))
        for call in (lambda: self.s.document_file("bilal", doc["id"]), lambda: self.s.document_delete("bilal", doc["id"]),
                     lambda: self.s.extract("bilal", doc["id"]), lambda: self.s.appliance_update("bilal", self.fridge["id"], {"name": "x"})):
            with self.assertRaises(NotFoundError):
                call()
        self.assertEqual((self.s.appliances("bilal"), self.s.documents("bilal")), ([], []))
        self.s.document_delete(U, doc["id"])
        self.assertEqual(self.s.documents(U), [])


class AssistantThroughTheServiceTests(AppServiceTestCase):
    def test_grounded_answer_safety_and_exact_answers(self):
        self.upload_manual()
        r = self.s.ask(U, "How do I clear ice build-up in the freezer?")
        self.assertEqual(r["status"], "answered")
        self.assertTrue(r["citations"][0].startswith(MANUAL.name))
        self.assertEqual(self.s.ask(U, "there is smoke coming from the fridge")["status"], "safety_escalation")
        self.s.repair_create(U, {"appliance_id": self.fridge["id"], "reported_date": "2025-11-02", "problem_description": "Ice", "cost": "3500"})
        self.assertIn("PKR 3,500.00", self.s.ask(U, "How much did I spend on repairs in the last 6 months?")["answer"])
        with self.assertRaises(DomainValidationError):
            self.s.ask(U, "   ")

    def test_index_is_rebuilt_after_a_restart(self):
        self.upload_manual()
        probe = lambda svc: svc.rag.vector_store.query(U, svc.rag.embedder.embed_query("ice in the freezer"), 5)
        restarted = self.make(data_dir=self.data_dir)                                   # same database, brand-new (empty) in-memory vector store
        self.assertGreater(len(probe(restarted)), 0)                                    # the rebuild restored the vectors
        self.assertEqual(restarted.ask(U, "How do I clear ice build-up in the freezer?")["status"], "answered")
        without = self.make(data_dir=self.data_dir, rebuild=False)
        self.assertEqual(probe(without), [])                                            # proves the rebuild is what restores them
        self.assertEqual(without.ask(U, "How do I clear ice build-up in the freezer?")["status"], "answered")   # keyword search still works from SQLite
        self.assertGreater(rebuild_vector_index(restarted.app, restarted.rag), 0)

    def test_without_an_llm_the_app_degrades_gracefully(self):
        bare = self.make(llm=None)                                                       # no key/model configured in this environment
        self.assertFalse(bare.status()["llm_configured"])
        bare_fridge = bare.appliance_create(U, {"name": "Fridge", "brand": "Haier"})
        bare.document_upload(U, MANUAL.name, MANUAL.read_bytes(), "manual", bare_fridge["id"])
        r = bare.ask(U, "How do I clear ice build-up in the freezer?")
        self.assertEqual(r["status"], "unavailable")
        self.assertIn(MANUAL.name, r["answer"])                                          # still shows cited passages
        self.assertEqual(bare.ask(U, "which warranties expire soon?")["status"], "answered")
        with self.assertRaises(LLMUnavailableError):
            UnavailableLLMProvider("x").generate([])


class ReviewAndRecordsTests(AppServiceTestCase):
    def test_extraction_review_flow_and_accuracy(self):
        doc = self.s.document_upload(U, "warranty_card_SYNTHETIC.txt", (DOCS / "warranty_card_SYNTHETIC.txt").read_bytes(), "warranty_card", self.fridge["id"])["document"]
        draft = self.s.extract(U, doc["id"])
        self.assertEqual([f["name"] for f in draft["fields"]][:2], ["product_name", "brand"])          # schema order, with labels
        self.assertEqual(draft["fields"][5]["label"], "Warranty duration (months)")
        self.assertEqual(len(self.s.drafts(U)), 1)
        values = {f["name"]: f["value"] for f in draft["fields"]}
        result = self.s.draft_confirm(U, draft["id"], values)
        self.assertIn("warranty", result["created"])
        w = self.s.warranties(U)[0]
        self.assertEqual((w["expiry_date"], w["status"], w["verified"], w["days_remaining"]), ("2026-03-14", "expiring_soon", True, 22))
        self.assertEqual(self.s.drafts(U), [])
        self.assertEqual(self.s.extraction_accuracy(U)["fields_correct"], 7)
        with self.assertRaises(DomainValidationError):
            self.s.drafts(U, status="nonsense")
        with self.assertRaises(DomainValidationError):
            self.s.expenses(U, 6, category="nonsense")

    def test_manual_entry_forms_expenses_maintenance_and_report(self):
        fid = self.fridge["id"]
        self.s.warranty_create(U, {"appliance_id": fid, "start_date": "2024-03-14", "duration_months": 24})
        self.s.repair_create(U, {"appliance_id": fid, "reported_date": "2025-11-02", "problem_description": "Ice build-up", "cost": "3500", "parts_replaced": ["Sensor"]})
        task = self.s.task_create(U, {"appliance_id": fid, "title": "Clean coils", "next_due_date": "2026-02-01", "interval_months": 6})
        self.assertEqual(self.s.maintenance(U)["overdue"][0]["days_until"], -19)
        done = self.s.task_complete(U, task["id"], {"done_on": "2026-02-18", "cost": "800"})
        self.assertEqual(done["next_due_date"], "2026-08-18")
        e = self.s.expenses(U, months=6)
        self.assertEqual((e["total"], e["by_category"]["repair"], e["by_category"]["maintenance"]), ("4300.00", "3500.00", "800.00"))
        self.assertEqual(self.s.expenses(U, 6, category="repair")["total"], "3500.00")
        self.assertIsNotNone(e["comparison"]["delta"])
        report = self.s.report(U, 6)
        self.assertIn("Haier Fridge", report["markdown"])
        self.assertTrue(report["path"].startswith(f"{U}/reports/"))
        with self.assertRaisesRegex(DomainValidationError, "start_date is required"):
            self.s.warranty_create(U, {"appliance_id": fid, "duration_months": 12})

    def test_unconfirmed_warranty_can_be_confirmed(self):
        from gharcare.domain import Warranty
        w = self.s.app.warranties.add(Warranty(user_id=U, appliance_id=self.fridge["id"], start_date=date(2025, 1, 1), duration_months=12))
        self.assertFalse(self.s.warranties(U)[0]["verified"])
        self.assertTrue(self.s.warranty_confirm(U, w.id)["verified"])


if __name__ == "__main__":
    unittest.main()


class SplitAnswerTests(unittest.TestCase):
    def test_parts(self):
        from gharcare.api.service import split_answer
        full = ("Leave the doors open [1].\n\nFrom your records:\nWarranty: active\nSimilar past repair: ice\n\n"
                "Sources:\n[1] m.pdf, p.3 (DEFROSTING)\n[2] m.pdf, p.4")
        self.assertEqual(split_answer(full), {"body": "Leave the doors open [1].", "records": ["Warranty: active", "Similar past repair: ice"],
                                              "sources": [{"n": 1, "citation": "m.pdf, p.3 (DEFROSTING)"}, {"n": 2, "citation": "m.pdf, p.4"}]})
        self.assertEqual(split_answer("Not found.\n\nFrom your records:\nWarranty: x"), {"body": "Not found.", "records": ["Warranty: x"], "sources": []})
        self.assertEqual(split_answer("Sources: are mentioned inline only"), {"body": "Sources: are mentioned inline only", "records": [], "sources": []})

    def test_ask_returns_parts(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        from gharcare.api.service import AppService
        from gharcare.config import Settings
        from gharcare.llm import FakeLLMProvider
        svc = AppService.create(Settings(data_dir=tmp.name, embedder="hashing", vector_store="memory"), llm=FakeLLMProvider(responder=model),
                                clock=lambda: TODAY, workflow_backend="local")
        self.addCleanup(svc.close)
        fridge = svc.appliance_create(U, {"name": "Fridge"})
        manual = Path(__file__).parent.parent / "examples" / "sample_docs" / "xr200_sample_manual_SYNTHETIC.txt"
        svc.document_upload(U, manual.name, manual.read_bytes(), "manual", fridge["id"])
        r = svc.ask(U, "How do I clear ice build-up in the freezer?", fridge["id"])
        self.assertEqual(r["parts"]["body"], "Leave the doors open for about 6 hours [1].")
        self.assertEqual([s["n"] for s in r["parts"]["sources"]], [1])
        self.assertNotIn("Sources:", r["parts"]["body"])
