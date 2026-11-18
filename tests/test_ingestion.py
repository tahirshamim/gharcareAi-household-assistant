import unittest
from dataclasses import replace

from gharcare.domain import IndexStatus
from gharcare.errors import DomainValidationError, EmbeddingError, NotFoundError, StorageError
from gharcare.retrieval import HashingEmbedder
from tests.rag_fixtures import MANUAL_A, RagTestCase


class FlakyEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.fail = True

    def embed_documents(self, texts):
        if self.fail:
            raise EmbeddingError("model unavailable")
        return super().embed_documents(texts)


class WrongCountEmbedder(HashingEmbedder):
    def embed_documents(self, texts):
        return super().embed_documents(texts)[:-1]


class UpsertFailingStore:
    """Vector store whose write fails: chunks are already saved when this happens."""
    def __init__(self, inner):
        self.inner, self.fail = inner, True

    def upsert(self, user_id, records):
        if self.fail:
            raise StorageError("vector database unavailable")
        self.inner.upsert(user_id, records)

    def query(self, *a, **kw):
        return self.inner.query(*a, **kw)

    def delete_document(self, *a, **kw):
        return self.inner.delete_document(*a, **kw)


class IngestionTests(RagTestCase):
    def test_successful_ingest_indexes_everything(self):
        result = self.ingest()
        self.assertTrue(result.indexed)
        self.assertIsNone(result.warning)
        self.assertEqual(result.chunk_count, 2)
        doc = self.app.documents.get("ayesha", result.document.id)
        self.assertEqual(doc.index_status, IndexStatus.INDEXED)
        self.assertEqual(self.app.chunks.count_for_document("ayesha", doc.id), 2)
        self.assertEqual(len(self.vectors.query("ayesha", self.embedder.embed_query("ice"), 10)), 2)
        self.assertTrue(self.app.artifacts.exists("ayesha", doc.storage_path))          # original kept
        extracted = self.app.artifacts.load_latest_extracted("ayesha", doc.id)          # extraction stored
        self.assertEqual(extracted["page_count"], 2)
        self.assertEqual(doc.mime_type, "text/plain")

    def test_pages_and_sections_reach_the_chunks(self):
        doc = self.ingest().document
        chunks = self.app.chunks.list_for_user("ayesha", [doc.id])
        self.assertEqual([(c.page_number, c.section) for c in chunks], [(1, "DEFROSTING"), (2, "TROUBLESHOOTING")])

    def test_duplicate_upload_for_same_appliance_is_detected_and_not_reindexed(self):
        first = self.ingest()
        second = self.ingest()
        self.assertTrue(second.duplicate)
        self.assertEqual(second.document.id, first.document.id)
        self.assertEqual(len(self.app.documents.list_for_user("ayesha")), 1)
        self.assertEqual(len(self.app.chunks.list_for_user("ayesha")), 2)

    def test_same_file_for_a_second_appliance_gets_its_own_document(self):
        a = self.ingest(appliance=self.fridge)
        b = self.ingest(appliance=self.washer)
        self.assertNotEqual(a.document.id, b.document.id)
        self.assertEqual(a.document.storage_path, b.document.storage_path)   # one stored file, two links

    def test_blank_text_file_is_saved_but_reported_not_silently_indexed(self):
        result = self.rag.ingestion.ingest("ayesha", "scan.txt", b"  \n\f \n ", "receipt", self.fridge.id)
        self.assertFalse(result.indexed)
        self.assertIn("OCR", result.warning)
        self.assertEqual(result.document.index_status, IndexStatus.FAILED)
        self.assertEqual(self.app.chunks.list_for_user("ayesha"), [])
        self.assertEqual(len(self.app.documents.list_for_user("ayesha")), 1)   # original still recorded

    def test_unsupported_type_is_rejected_before_anything_is_stored(self):
        with self.assertRaises(DomainValidationError):
            self.rag.ingestion.ingest("ayesha", "photo.jpg", b"\xff\xd8\xff", "receipt")
        self.assertEqual(self.app.documents.list_for_user("ayesha"), [])

    def test_empty_file_and_bad_doc_type_rejected(self):
        with self.assertRaises(DomainValidationError):
            self.rag.ingestion.ingest("ayesha", "a.txt", b"", "manual")
        with self.assertRaises(DomainValidationError):
            self.rag.ingestion.ingest("ayesha", "a.txt", b"text", "banana")

    def test_binary_garbage_in_a_txt_file_fails_cleanly(self):
        from gharcare.errors import IngestionError
        with self.assertRaises(IngestionError):
            self.rag.ingestion.ingest("ayesha", "a.txt", b"\xff\xfe\xfa\x80\x81", "manual")
        doc = self.app.documents.list_for_user("ayesha")[0]
        self.assertEqual(doc.index_status, IndexStatus.FAILED)

    def test_unknown_or_foreign_appliance_rejected(self):
        with self.assertRaises(NotFoundError):
            self.rag.ingestion.ingest("bilal", "a.txt", b"hello world", "manual", appliance_id=self.fridge.id)
        self.assertEqual(self.app.documents.list_for_user("bilal"), [])

    def test_embedding_failure_leaves_no_partial_state_and_reindex_recovers(self):
        flaky = FlakyEmbedder()
        from gharcare.rag import build_rag
        rag = build_rag(self.app, flaky, self.vectors)
        with self.assertRaises(EmbeddingError):
            rag.ingestion.ingest("ayesha", "m.txt", MANUAL_A.encode(), "manual", self.fridge.id)
        doc = self.app.documents.list_for_user("ayesha")[0]
        self.assertEqual(doc.index_status, IndexStatus.FAILED)
        self.assertEqual(self.app.chunks.list_for_user("ayesha"), [])
        self.assertEqual(self.vectors.query("ayesha", flaky.embed_query("ice"), 10), [])
        flaky.fail = False                                   # the model comes back
        result = rag.ingestion.reindex("ayesha", doc.id)
        self.assertTrue(result.indexed)
        self.assertEqual(len(self.app.chunks.list_for_user("ayesha")), 2)

    def test_vector_store_failure_after_chunks_are_saved_rolls_chunks_back(self):
        from gharcare.rag import build_rag
        store = UpsertFailingStore(self.vectors)
        rag = build_rag(self.app, self.embedder, store)
        with self.assertRaises(StorageError):
            rag.ingestion.ingest("ayesha", "m.txt", MANUAL_A.encode(), "manual", self.fridge.id)
        doc = self.app.documents.list_for_user("ayesha")[0]
        self.assertEqual(doc.index_status, IndexStatus.FAILED)
        self.assertEqual(self.app.chunks.list_for_user("ayesha"), [])        # no searchable half-document
        store.fail = False
        self.assertTrue(rag.ingestion.reindex("ayesha", doc.id).indexed)

    def test_wrong_vector_count_from_embedder_is_caught(self):
        from gharcare.rag import build_rag
        rag = build_rag(self.app, WrongCountEmbedder(), self.vectors)
        with self.assertRaises(EmbeddingError):
            rag.ingestion.ingest("ayesha", "m.txt", MANUAL_A.encode(), "manual", self.fridge.id)
        self.assertEqual(self.app.chunks.list_for_user("ayesha"), [])

    def test_reindex_replaces_instead_of_duplicating(self):
        doc = self.ingest().document
        self.rag.ingestion.reindex("ayesha", doc.id)
        self.rag.ingestion.reindex("ayesha", doc.id)
        self.assertEqual(self.app.chunks.count_for_document("ayesha", doc.id), 2)
        self.assertEqual(len(self.vectors.query("ayesha", self.embedder.embed_query("ice"), 10)), 2)

    def test_delete_document_removes_chunks_vectors_and_row_but_keeps_original(self):
        doc = self.ingest().document
        self.rag.ingestion.delete_document("ayesha", doc.id)
        self.assertEqual(self.app.documents.list_for_user("ayesha"), [])
        self.assertEqual(self.app.chunks.list_for_user("ayesha"), [])
        self.assertEqual(self.vectors.query("ayesha", self.embedder.embed_query("ice"), 10), [])
        self.assertTrue(self.app.artifacts.exists("ayesha", doc.storage_path))

    def test_other_users_cannot_delete_or_reindex_a_document(self):
        doc = self.ingest().document
        with self.assertRaises(NotFoundError):
            self.rag.ingestion.delete_document("bilal", doc.id)
        with self.assertRaises(NotFoundError):
            self.rag.ingestion.reindex("bilal", doc.id)
        self.assertEqual(len(self.app.documents.list_for_user("ayesha")), 1)


if __name__ == "__main__":
    unittest.main()
