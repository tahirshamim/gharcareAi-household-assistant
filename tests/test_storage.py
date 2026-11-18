import hashlib
import stat
import tempfile
import unittest
from pathlib import Path

from gharcare.errors import DomainValidationError, NotFoundError, StorageError
from gharcare.storage import LocalArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = LocalArtifactStore(self.root, max_upload_bytes=1000)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_read_original(self):
        ref = self.store.save_original("ayesha", "Fridge Manual.PDF", b"%PDF manual")
        self.assertTrue(ref.created)
        self.assertEqual(ref.sha256, hashlib.sha256(b"%PDF manual").hexdigest())
        self.assertTrue(ref.relative_path.startswith("ayesha/originals/"))
        self.assertTrue(ref.relative_path.endswith(".pdf"))      # extension lower-cased
        self.assertNotIn("Fridge", ref.relative_path)             # user filename never hits disk
        self.assertEqual(self.store.read("ayesha", ref.relative_path), b"%PDF manual")

    def test_identical_content_is_deduplicated(self):
        first = self.store.save_original("ayesha", "a.pdf", b"same bytes")
        second = self.store.save_original("ayesha", "renamed.pdf", b"same bytes")
        self.assertFalse(second.created)
        self.assertEqual(first.relative_path, second.relative_path)

    def test_same_content_for_two_users_is_stored_separately(self):
        a = self.store.save_original("ayesha", "a.pdf", b"shared manual")
        b = self.store.save_original("bilal", "a.pdf", b"shared manual")
        self.assertNotEqual(a.relative_path, b.relative_path)
        self.assertTrue(b.created)

    def test_originals_are_read_only(self):
        ref = self.store.save_original("ayesha", "a.pdf", b"data")
        mode = stat.S_IMODE((self.root / ref.relative_path).stat().st_mode)
        self.assertEqual(mode, 0o444)

    def test_no_temp_files_left_behind(self):
        self.store.save_original("ayesha", "a.pdf", b"data")
        self.assertEqual([p for p in self.root.rglob(".tmp-*")], [])

    def test_rejects_empty_oversized_and_bad_user(self):
        with self.assertRaises(DomainValidationError):
            self.store.save_original("ayesha", "a.pdf", b"")
        with self.assertRaises(DomainValidationError):
            self.store.save_original("ayesha", "a.pdf", b"x" * 1001)
        with self.assertRaises(DomainValidationError):
            self.store.save_original("../evil", "a.pdf", b"data")

    def test_hostile_filenames_cannot_influence_the_path(self):
        for name in ("../../etc/passwd", "a.p df", "x.pdf/../../y", "", "noext", "a." + "z" * 30):
            with self.subTest(name=name):
                ref = self.store.save_original("ayesha", name, name.encode() or b"empty-name")
                resolved = (self.root / ref.relative_path).resolve()
                self.assertTrue(resolved.is_relative_to((self.root / "ayesha" / "originals").resolve()))

    def test_path_traversal_and_cross_user_reads_are_blocked(self):
        ref = self.store.save_original("ayesha", "a.pdf", b"private")
        with self.assertRaises(StorageError):
            self.store.read("bilal", ref.relative_path)                  # another user's file
        with self.assertRaises(StorageError):
            self.store.read("ayesha", "ayesha/../../outside.txt")        # escape attempt
        self.assertFalse(self.store.exists("bilal", ref.relative_path))
        self.assertTrue(self.store.exists("ayesha", ref.relative_path))

    def test_reading_missing_file(self):
        with self.assertRaises(NotFoundError):
            self.store.read("ayesha", "ayesha/originals/zz/missing.pdf")

    def test_extraction_results_are_versioned_never_overwritten(self):
        r1 = self.store.save_extracted("ayesha", "doc1", {"amount": "9500", "confidence": 0.4})
        r2 = self.store.save_extracted("ayesha", "doc1", {"amount": "95000", "confidence": 1.0, "user_corrected": True})
        self.assertTrue(r1.relative_path.endswith("v0001.json"))
        self.assertTrue(r2.relative_path.endswith("v0002.json"))
        self.assertEqual(self.store.read("ayesha", r1.relative_path).decode().count("9500"), 1)  # v1 intact
        self.assertEqual(self.store.load_latest_extracted("ayesha", "doc1")["amount"], "95000")
        self.assertIsNone(self.store.load_latest_extracted("ayesha", "unknown-doc"))

    def test_extraction_handles_urdu_and_non_json_types(self):
        from datetime import date
        self.store.save_extracted("ayesha", "doc2", {"note": "فریج", "when": date(2024, 3, 14)})
        loaded = self.store.load_latest_extracted("ayesha", "doc2")
        self.assertEqual(loaded["note"], "فریج")
        self.assertEqual(loaded["when"], "2024-03-14")

    def test_reports_never_overwrite_and_names_are_sanitised(self):
        a = self.store.save_report("ayesha", "Year-End Report!! ../x", b"# one")
        b = self.store.save_report("ayesha", "Year-End Report!! ../x", b"# two")
        self.assertNotEqual(a.relative_path, b.relative_path)
        self.assertEqual(self.store.read("ayesha", a.relative_path), b"# one")
        self.assertTrue((self.root / a.relative_path).resolve().is_relative_to((self.root / "ayesha" / "reports").resolve()))
        with self.assertRaises(DomainValidationError):
            self.store.save_report("ayesha", "x", b"data", extension=".exe")


if __name__ == "__main__":
    unittest.main()
