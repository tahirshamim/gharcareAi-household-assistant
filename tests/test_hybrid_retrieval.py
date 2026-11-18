import unittest
from dataclasses import replace

from gharcare.domain import DocumentType, IndexStatus
from tests.rag_fixtures import MANUAL_A, MANUAL_B, RagTestCase


class HybridRetrievalTests(RagTestCase):
    def setUp(self):
        super().setUp()
        self.fridge_doc = self.ingest("fridge.txt", MANUAL_A, appliance=self.fridge).document
        self.washer_doc = self.ingest("washer.txt", MANUAL_B, appliance=self.washer).document

    def retrieve(self, query, **kw):
        return self.rag.retriever.retrieve("ayesha", query, **kw)

    def test_exact_error_code_is_found_and_cited_with_page_and_section(self):
        top = self.retrieve("defrost sensor E4", appliance_id=self.fridge.id)[0]
        self.assertEqual(top.document_name, "fridge.txt")
        self.assertEqual(top.chunk.page_number, 2)
        self.assertEqual(top.citation, "fridge.txt, p.2 (TROUBLESHOOTING)")
        self.assertIsNotNone(top.keyword_score)
        self.assertIsNotNone(top.vector_similarity)

    def test_appliance_filter_prevents_cross_appliance_answers(self):
        # Both manuals mention E4 with different meanings; the filter must pick the right appliance.
        for appliance, expected in ((self.fridge, "fridge.txt"), (self.washer, "washer.txt")):
            names = {r.document_name for r in self.retrieve("what does E4 mean", appliance_id=appliance.id)}
            self.assertEqual(names, {expected})

    def test_without_filter_both_appear(self):
        names = {r.document_name for r in self.retrieve("what does E4 mean", k=5)}
        self.assertEqual(names, {"fridge.txt", "washer.txt"})

    def test_users_cannot_retrieve_each_others_documents(self):
        self.assertEqual(self.rag.retriever.retrieve("bilal", "E4 defrost sensor"), [])

    def test_foreign_appliance_id_returns_nothing(self):
        self.assertEqual(self.rag.retriever.retrieve("bilal", "E4", appliance_id=self.fridge.id), [])

    def test_appliance_without_documents_returns_nothing_rather_than_searching_everything(self):
        from gharcare.domain import Appliance
        oven = self.app.appliances.add(Appliance(user_id="ayesha", name="Oven"))
        self.assertEqual(self.retrieve("E4 defrost sensor", appliance_id=oven.id), [])

    def test_failed_and_pending_documents_are_not_searchable(self):
        self.app.documents.update(replace(self.fridge_doc, index_status=IndexStatus.FAILED))
        names = {r.document_name for r in self.retrieve("E4 defrost sensor", k=5)}
        self.assertEqual(names, {"washer.txt"})

    def test_document_type_filter(self):
        receipt = self.ingest("receipt.txt", "Haier fridge receipt total 95000 rupees E4", appliance=self.fridge,
                              doc_type="receipt").document
        manuals = self.retrieve("E4 95000", doc_types=[DocumentType.MANUAL], k=10)
        self.assertNotIn(receipt.original_filename, {r.document_name for r in manuals})
        receipts = self.retrieve("E4 95000", doc_types=[DocumentType.RECEIPT], k=10)
        self.assertEqual({r.document_name for r in receipts}, {"receipt.txt"})

    def test_orphan_vectors_are_ignored(self):
        self.app.chunks.delete_for_document("ayesha", self.fridge_doc.id)   # chunks gone, vectors remain
        names = {r.document_name for r in self.retrieve("ice freezer defrost", k=10)}
        self.assertNotIn("fridge.txt", names)

    def test_roman_urdu_keyword_match(self):
        self.ingest("urdu.txt", "URDU GUIDE\n\nFridge mein barf jam jaye to plug nikal dein.", appliance=self.fridge)
        top = self.retrieve("fridge mein barf jam rahi hai", appliance_id=self.fridge.id)[0]
        self.assertEqual(top.document_name, "urdu.txt")

    def test_completely_unrelated_query_returns_nothing(self):
        self.assertEqual(self.retrieve("zebra giraffe"), [])     # no word overlap -> no noise results

    def test_blank_query_and_bad_k(self):
        from gharcare.errors import DomainValidationError
        self.assertEqual(self.retrieve("   "), [])
        with self.assertRaises(DomainValidationError):
            self.retrieve("E4", k=0)

    def test_result_count_respects_k_and_scores_are_ordered(self):
        results = self.retrieve("E4 ice freezer defrost drum", k=2)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_results_found_by_both_methods_carry_both_ranks(self):
        top = self.retrieve("defrost sensor E4", appliance_id=self.fridge.id)[0]
        self.assertIsNotNone(top.vector_rank)
        self.assertIsNotNone(top.keyword_rank)


if __name__ == "__main__":
    unittest.main()
