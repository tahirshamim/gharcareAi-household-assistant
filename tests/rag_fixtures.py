"""Shared test wiring for ingestion/retrieval tests."""
import tempfile
import unittest

from gharcare.config import Settings
from gharcare.container import AppContainer
from gharcare.domain import Appliance
from gharcare.rag import build_rag
from gharcare.retrieval import HashingEmbedder, InMemoryVectorStore

MANUAL_A = ("DEFROSTING\n\nIf ice builds up in the freezer, unplug the fridge and leave the door open for six hours.\f"
            "TROUBLESHOOTING\n\nE4 - Defrost sensor fault. Call an authorised service centre.")
MANUAL_B = ("TROUBLESHOOTING\n\nE4 - The drum is unbalanced. Redistribute the laundry.\f"
            "CARE\n\nClean the detergent drawer monthly.")


class RagTestCase(unittest.TestCase):
    embedder_cls = HashingEmbedder

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.app = AppContainer.build(Settings(data_dir=self._tmp.name, embedder="hashing", vector_store="memory"))
        self.addCleanup(self.app.close)
        self.embedder = self.embedder_cls()
        self.vectors = InMemoryVectorStore()
        self.rag = build_rag(self.app, self.embedder, self.vectors)
        self.fridge = self.app.appliances.add(Appliance(user_id="ayesha", name="Fridge", brand="Haier"))
        self.washer = self.app.appliances.add(Appliance(user_id="ayesha", name="Washer", brand="Dawlance"))

    def ingest(self, name="manual.txt", text=MANUAL_A, user="ayesha", appliance=None, doc_type="manual"):
        appliance = appliance if appliance is not None else self.fridge
        return self.rag.ingestion.ingest(user, name, text.encode("utf-8"), doc_type,
                                         appliance_id=appliance.id if appliance else None)
