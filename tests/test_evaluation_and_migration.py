import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from gharcare.domain import Chunk, Document, DocumentType
from gharcare.errors import DomainValidationError, NotFoundError, StorageError
from gharcare.evaluation import EvalCase, evaluate_retrieval, first_relevant_rank, load_eval_cases
from gharcare.repositories import SQLiteChunkRepository, SQLiteDatabase, SQLiteDocumentRepository
from gharcare.repositories.sqlite import _SCHEMA_V1, SCHEMA_VERSION
from gharcare.retrieval import RetrievedChunk
from tests.rag_fixtures import MANUAL_A, RagTestCase


def fake_result(doc, page, text="t"):
    return RetrievedChunk(chunk=Chunk(user_id="u", document_id="d", chunk_index=0, text=text, page_number=page),
                          document_name=doc, score=0.1)


class MetricTests(unittest.TestCase):
    def test_first_relevant_rank(self):
        results = [fake_result("m.pdf", 9), fake_result("m.pdf", 3), fake_result("m.pdf", 4)]
        self.assertEqual(first_relevant_rank(results, frozenset({("m.pdf", 3)})), 2)
        self.assertEqual(first_relevant_rank(results, frozenset({("m.pdf", 3), ("m.pdf", 4)})), 2)
        self.assertIsNone(first_relevant_rank(results, frozenset({("other.pdf", 3)})))   # right page, wrong file
        self.assertIsNone(first_relevant_rank([], frozenset({("m.pdf", 3)})))

    def test_evidence_phrase_makes_the_metric_chunk_level(self):
        results = [fake_result("m.pdf", 4, "water leaking at the bottom"),
                   fake_result("m.pdf", 4, "E4 - Defrost sensor fault")]
        relevant = frozenset({("m.pdf", 4)})
        self.assertEqual(first_relevant_rank(results, relevant), 1)                      # page-level: first chunk counts
        self.assertEqual(first_relevant_rank(results, relevant, "e4 - defrost"), 2)      # chunk-level: needs the evidence
        self.assertIsNone(first_relevant_rank(results, relevant, "not in any chunk"))

    def test_evidence_matching_ignores_line_wrapping_and_case(self):
        results = [fake_result("m.pdf", 4, "Do not use the appliance if the DOOR is\nopened too frequently or too long")]
        self.assertEqual(first_relevant_rank(results, frozenset({("m.pdf", 4)}), "door is opened too frequently"), 1)

    def test_answerable_case_requires_labels(self):
        with self.assertRaises(DomainValidationError):
            EvalCase(question="q")

    def test_load_eval_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps([
                {"question": "a", "relevant": [{"document": "m.txt", "page": 2}]},
                {"question": "b", "relevant": [], "answerable": False}]), encoding="utf-8")
            cases = load_eval_cases(path)
        self.assertEqual(cases[0].relevant, frozenset({("m.txt", 2)}))
        self.assertIsNone(cases[0].must_contain)
        self.assertFalse(cases[1].answerable)
        with self.assertRaises(DomainValidationError):
            load_eval_cases("/definitely/missing.json")


class EvaluationRunTests(RagTestCase):
    def test_hand_computed_metrics(self):
        self.ingest("fridge.txt", MANUAL_A)
        cases = [
            EvalCase("E4 defrost sensor fault", frozenset({("fridge.txt", 2)})),          # rank 1 -> RR 1.0
            EvalCase("ice freezer unplug", frozenset({("fridge.txt", 1)})),               # rank 1 -> RR 1.0
            EvalCase("E4 defrost sensor fault freezer", frozenset({("fridge.txt", 1)})),  # page 2 first, page 1 second -> RR 0.5
            EvalCase("zebra giraffe", frozenset({("fridge.txt", 1)})),                    # nothing matches -> no results -> miss
            EvalCase("zebra giraffe", frozenset(), answerable=False),
        ]
        report = evaluate_retrieval(self.rag.retriever, "ayesha", cases, k_values=(1, 2))
        self.assertEqual((report.n_answerable, report.n_unanswerable), (4, 1))
        self.assertAlmostEqual(report.recall_at[1], 2 / 4)       # cases 1 and 2 hit at rank 1
        self.assertAlmostEqual(report.recall_at[2], 3 / 4)       # case 3 is found at rank 2
        self.assertAlmostEqual(report.mrr, (1 + 1 + 0.5 + 0) / 4)
        self.assertEqual(report.misses, ["zebra giraffe"])
        self.assertEqual(report.unanswerable_top_similarity, [0.0])
        self.assertEqual(report.unanswerable_top_keyword, [0.0])  # nothing matched at all
        self.assertIn("Hit rate @1", report.format())


class MigrationTests(unittest.TestCase):
    def test_v1_database_is_upgraded_in_place_without_losing_data(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "old.db"
            old = sqlite3.connect(path)
            old.executescript(_SCHEMA_V1 + "PRAGMA user_version = 1;")
            old.execute("INSERT INTO appliances (id, user_id, name, created_at) VALUES ('a1','ayesha','Fridge','2025-01-01T00:00:00+00:00')")
            old.commit()
            old.close()
            db = SQLiteDatabase(path)
            self.assertEqual(db._conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(db._conn.execute("SELECT name FROM appliances").fetchone()[0], "Fridge")
            self.assertEqual(db._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)
            db.close()

    def test_database_from_a_newer_app_is_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "future.db"
            future = sqlite3.connect(path)
            future.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
            future.close()
            with self.assertRaises(StorageError):
                SQLiteDatabase(path)
            path.unlink()                    # fails on Windows if the refused database left its connection open

    def test_reopening_a_current_database_is_a_noop(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "x.db"
            SQLiteDatabase(path).close()
            SQLiteDatabase(path).close()


class ChunkRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.db = SQLiteDatabase(":memory:")
        self.addCleanup(self.db.close)
        self.docs, self.chunks = SQLiteDocumentRepository(self.db), SQLiteChunkRepository(self.db)
        self.doc = self.docs.add(Document(user_id="ayesha", original_filename="m.txt", doc_type=DocumentType.MANUAL,
                                          sha256="a" * 64, storage_path="p", size_bytes=1))

    def make(self, i, user="ayesha"):
        return Chunk(user_id=user, document_id=self.doc.id, chunk_index=i, text=f"text {i}", page_number=i + 1, section="S")

    def test_round_trip_ordering_and_filters(self):
        self.chunks.add_many([self.make(1), self.make(0)])
        self.assertEqual([c.chunk_index for c in self.chunks.list_for_user("ayesha")], [0, 1])
        self.assertEqual(self.chunks.list_for_user("ayesha", []), [])                   # empty filter = nothing
        self.assertEqual(len(self.chunks.list_for_user("ayesha", [self.doc.id])), 2)
        self.assertEqual(self.chunks.count_for_document("ayesha", self.doc.id), 2)
        self.assertEqual(set(self.chunks.get_many("ayesha", [f"{self.doc.id}-0", "nope"])), {f"{self.doc.id}-0"})

    def test_cannot_add_chunks_to_someone_elses_document(self):
        with self.assertRaises(NotFoundError):
            self.chunks.add_many([self.make(0, user="bilal")])
        self.assertEqual(self.chunks.list_for_user("bilal"), [])

    def test_duplicate_chunk_ids_are_a_storage_error(self):
        self.chunks.add_many([self.make(0)])
        with self.assertRaises(StorageError):
            self.chunks.add_many([self.make(0)])

    def test_deleting_the_document_cascades_to_its_chunks(self):
        self.chunks.add_many([self.make(0)])
        self.docs.delete("ayesha", self.doc.id)
        self.assertEqual(self.chunks.list_for_user("ayesha"), [])

    def test_delete_for_document_counts(self):
        self.chunks.add_many([self.make(0), self.make(1)])
        self.assertEqual(self.chunks.delete_for_document("bilal", self.doc.id), 0)
        self.assertEqual(self.chunks.delete_for_document("ayesha", self.doc.id), 2)


if __name__ == "__main__":
    unittest.main()
