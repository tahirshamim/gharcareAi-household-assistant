"""One contract, run against every VectorStore implementation. The Chroma class runs only when
chromadb is installed (pip install chromadb) so you can verify the real adapter on your machine."""
import importlib.util
import tempfile
import unittest

from gharcare.retrieval import ChromaVectorStore, InMemoryVectorStore, VectorRecord

X, Y, XY = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]


class VectorStoreContract:
    def make_store(self):
        raise NotImplementedError

    def setUp(self):
        self.store = self.make_store()
        self.store.upsert("ayesha", [VectorRecord("c1", "docA", X), VectorRecord("c2", "docA", Y),
                                     VectorRecord("c3", "docB", XY)])

    def ids(self, hits):
        return [h.chunk_id for h in hits]

    def test_nearest_first_with_similarity_scores(self):
        hits = self.store.query("ayesha", X, k=3)
        self.assertEqual(self.ids(hits), ["c1", "c3", "c2"])
        self.assertAlmostEqual(hits[0].similarity, 1.0, places=3)
        self.assertAlmostEqual(hits[2].similarity, 0.0, places=3)
        self.assertGreater(hits[1].similarity, 0.9)

    def test_k_limits_results(self):
        self.assertEqual(self.ids(self.store.query("ayesha", X, k=2)), ["c1", "c3"])

    def test_users_are_isolated(self):
        self.assertEqual(self.store.query("bilal", X, k=5), [])

    def test_document_filter(self):
        self.assertEqual(self.ids(self.store.query("ayesha", X, k=5, document_ids=["docB"])), ["c3"])
        self.assertEqual(set(self.ids(self.store.query("ayesha", X, k=5, document_ids=["docA", "docB"]))),
                         {"c1", "c2", "c3"})

    def test_empty_document_filter_means_nothing_not_everything(self):
        self.assertEqual(self.store.query("ayesha", X, k=5, document_ids=[]), [])

    def test_unknown_document_filter(self):
        self.assertEqual(self.store.query("ayesha", X, k=5, document_ids=["nope"]), [])

    def test_upsert_replaces_existing_id(self):
        self.store.upsert("ayesha", [VectorRecord("c1", "docA", Y)])
        hits = self.store.query("ayesha", Y, k=1)
        self.assertAlmostEqual(hits[0].similarity, 1.0, places=3)
        self.assertEqual(len(self.store.query("ayesha", X, k=10)), 3)   # still 3 records, not 4

    def test_delete_document_removes_only_that_document(self):
        self.store.delete_document("ayesha", "docA")
        self.assertEqual(self.ids(self.store.query("ayesha", X, k=5)), ["c3"])

    def test_delete_is_scoped_to_the_user(self):
        self.store.delete_document("bilal", "docA")
        self.assertEqual(len(self.store.query("ayesha", X, k=5)), 3)

    def test_query_on_empty_store(self):
        self.assertEqual(self.make_store().query("ayesha", X, k=3), [])


class InMemoryVectorStoreTests(VectorStoreContract, unittest.TestCase):
    def make_store(self):
        return InMemoryVectorStore()


@unittest.skipUnless(importlib.util.find_spec("chromadb"), "chromadb not installed (pip install chromadb)")
class ChromaVectorStoreTests(VectorStoreContract, unittest.TestCase):
    def make_store(self):
        # Chroma keeps files open on Windows, so tolerate cleanup errors.
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        return ChromaVectorStore(tmp.name, collection_name="contract_test")


if __name__ == "__main__":
    unittest.main()
