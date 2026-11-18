"""Checks the labeled question set for the REAL Gorenje manual. The PDF is third-party, so it is not shipped: put it at
examples/real_docs/590939.pdf to enable the PDF-dependent tests."""
import re
import unittest
from pathlib import Path

from gharcare.evaluation import load_eval_cases
from gharcare.ingestion import Chunker, PdfTextExtractor

ROOT = Path(__file__).parent.parent / "examples"
QUESTIONS = ROOT / "sample_docs" / "eval_questions_gorenje.json"
PDF = ROOT / "real_docs" / "590939.pdf"


def squash(text):
    return re.sub(r"\s+", " ", text.casefold())


class QuestionSetTests(unittest.TestCase):
    def test_well_formed(self):
        cases = load_eval_cases(QUESTIONS)
        self.assertGreaterEqual(sum(c.answerable for c in cases), 30)
        self.assertGreaterEqual(sum(not c.answerable for c in cases), 8)
        self.assertEqual(len({c.question for c in cases}), len(cases))
        self.assertTrue(all(doc == "590939.pdf" and 1 <= page <= 34 for c in cases for doc, page in c.relevant))


@unittest.skipUnless(PDF.exists(), "put the real manual at examples/real_docs/590939.pdf to run these")
class RealPdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = PdfTextExtractor().extract(PDF.read_bytes())
        cls.chunks = Chunker().chunk_pages("u1", "d1", cls.pages)

    def test_labeled_evidence_really_is_on_the_labeled_pages(self):
        by_page = {p.page_number: squash(p.text) for p in self.pages}
        for case in load_eval_cases(QUESTIONS):
            if case.must_contain:
                with self.subTest(question=case.question):
                    self.assertTrue(any(squash(case.must_contain) in by_page[page] for _, page in case.relevant))

    def test_real_manual_noise_is_gone_and_wrapped_headings_are_whole(self):
        self.assertFalse(any("512960" in c.text for c in self.chunks))                       # the footer code printed on every page
        self.assertTrue(all(sum(ch.isalpha() for ch in c.text) >= 6 for c in self.chunks))   # no stray diagram-label chunks
        sections = {c.section for c in self.chunks}
        for expected in ("BEFORE USING THE APPLIANCE FOR THE FIRST TIME", "ADVICE FOR SAVING POWER WITH REFRIGERATION APPLIANCES"):
            self.assertIn(expected, sections)


if __name__ == "__main__":
    unittest.main()
