"""Regressions from inspecting the real-manual misses (examples/inspect_cases.py, Step 6.3):
reserved slots evicted the best fused chunk at k=4; the retrieval evaluation sliced a k=5 list and hid it;
'[2, 3]' citations were rejected; correct but uncited drafts were thrown away; Roman Urdu questions searched
an English manual untranslated."""
import itertools
import random
import unittest

from gharcare.llm import FakeLLMProvider
from gharcare.errors import LLMUnavailableError
from gharcare.retrieval.hybrid import reserve_slots
from gharcare.workflow import HouseholdAssistant
from gharcare.workflow.assistant import JUDGE_SYSTEM, TRANSLATE_SYSTEM
from gharcare.workflow.intent import needs_translation
from tests.test_workflow import AssistantTestCase, ICE_Q, U


class ReserveCapTests(unittest.TestCase):
    def test_the_real_failure_k4_reserve2(self):
        # fused best 'a' is in neither retriever's top 2 -> it used to be evicted by four reserved chunks
        out = reserve_slots(["a", "b", "c", "d", "e"], ["k1", "k2"], ["v1", "v2"], k=4, reserve=2)
        self.assertEqual(out[:2], ["a", "b"])
        self.assertEqual(set(out[2:]), {"k1", "v1"})

    def test_fused_top_half_is_never_evicted(self):
        rng = random.Random(7)
        ids = [f"c{i}" for i in range(12)]
        for _ in range(500):
            ordered = rng.sample(ids, len(ids))
            kw, vec = rng.sample(ids, 4), rng.sample(ids, 4)
            k, reserve = rng.randint(1, 6), rng.randint(0, 3)
            out = reserve_slots(ordered, kw, vec, k, reserve)
            self.assertEqual(len(out), k)
            self.assertEqual(len(set(out)), k)
            self.assertTrue(set(ordered[: k - k // 2]) <= set(out), (ordered, kw, vec, k, reserve, out))

    def test_k1_is_pure_fusion(self):
        self.assertEqual(reserve_slots(["a", "b"], ["x"], ["y"], k=1, reserve=2), ["a"])


class CitationParsingTests(unittest.TestCase):
    def test_formats(self):
        c = HouseholdAssistant._cited_numbers
        self.assertEqual(c("x [2, 3]. y [1]", 4), [2, 3, 1])
        self.assertEqual(c("x [2,3][3]", 4), [2, 3])
        self.assertIsNone(c("x [2, 9]", 4))         # any out-of-range number rejects the draft
        self.assertIsNone(c("no citations", 4))


class VerificationTests(AssistantTestCase):
    def test_uncited_but_supported_draft_uses_the_judges_excerpts(self):
        judge = '{"supported": true, "unsupported_claims": [], "supporting_excerpts": [1]}'
        r = self.make(["Leave the doors open for about 6 hours.", judge]).ask(U, ICE_Q)
        self.assertEqual(r.status, "answered")
        self.assertEqual(len(r.citations), 1)
        self.assertIn("Sources:", r.answer)

    def test_judge_excerpts_are_validated(self):
        for bad in ('[9]', '["1"]', '[true]', '"1"', '[]'):
            with self.subTest(bad=bad):
                judge = '{"supported": true, "supporting_excerpts": %s}' % bad
                self.assertEqual(self.make(["Leave the doors open.", judge]).ask(U, ICE_Q).status, "unverified")

    def test_uncited_and_unsupported_stays_unverified(self):
        judge = '{"supported": false, "unsupported_claims": ["x"], "supporting_excerpts": [1]}'
        self.assertEqual(self.make(["Leave the doors open.", judge]).ask(U, ICE_Q).status, "unverified")

    def test_out_of_range_citation_skips_the_judge(self):
        r = self.make(["Leave the doors open [9]."]).ask(U, ICE_Q)
        self.assertEqual(r.status, "unverified")
        self.assertEqual(len(self.llm.calls), 1)

    def test_comma_citation_is_accepted(self):
        good = '{"supported": true, "unsupported_claims": []}'
        r = self.make(["Leave the doors open for about 6 hours [1, 2].", good]).ask(U, ICE_Q)
        self.assertEqual(r.status, "answered")

    def test_judge_is_told_safety_advice_is_allowed(self):
        self.assertIn("qualified technician is always acceptable", JUDGE_SYSTEM)


class TranslationTests(AssistantTestCase):
    def test_detection(self):
        for t in ["fridge mein bahut zyada barf jam gayi hai, kya karun?",
                  "fridge pehli bar chalane se pehle kitni der khara rakhna chahiye?", "فریج میں برف جم گئی ہے"]:
            self.assertTrue(needs_translation(t), t)
        for t in ["How do I clear ice build-up in the freezer?", "Can I use a power strip behind the fridge?",
                  "Is the warranty still valid?", "What does error code E4 mean?",
                  "How do I reset model KE-200?"]:          # one stray marker ('ke' from a model code) is not enough
            self.assertFalse(needs_translation(t), t)

    def _ask_and_capture_query(self, script, question):
        seen = []
        real = self.rag.retriever.retrieve

        def spy(user_id, query, **kw):
            seen.append(query)
            return real(user_id, query, **kw)
        self.rag.retriever.retrieve = spy
        r = self.make(script).ask(U, question)
        return r, seen

    def test_roman_urdu_is_searched_in_english_but_answered_from_the_original(self):
        q = "freezer mein barf jam gayi hai, kya karun?"
        r, seen = self._ask_and_capture_query(
            ["How do I clear ice build-up in the freezer?", "Doors khol kar 6 ghante chhor dein [1].",
             '{"supported": true, "unsupported_claims": []}'], q)
        self.assertEqual(seen, ["How do I clear ice build-up in the freezer?"])
        self.assertEqual(self.llm.calls[0][0].content, TRANSLATE_SYSTEM)
        self.assertIn(q, self.llm.calls[1][1].content)            # the answer prompt keeps the user's own words
        self.assertEqual(r.status, "answered")

    def test_translation_failure_falls_back_to_the_original(self):
        q = "freezer mein barf jam gayi hai, kya karun?"
        r, seen = self._ask_and_capture_query([LLMUnavailableError("down"), "NOT_FOUND"], q)
        self.assertEqual(seen, [q])
        self.assertEqual(r.status, "not_found")

    def test_english_questions_make_no_translation_call(self):
        self.make(["NOT_FOUND"]).ask(U, ICE_Q)
        self.assertNotIn(TRANSLATE_SYSTEM, [c[0].content for c in self.llm.calls])


class EvaluationAtKTests(unittest.TestCase):
    def test_hit_at_k_uses_retrieve_at_that_k(self):
        from gharcare.domain import Chunk
        from gharcare.evaluation import evaluate_retrieval
        from gharcare.evaluation.retrieval import EvalCase
        from gharcare.retrieval.hybrid import RetrievedChunk

        def rc(page):
            return RetrievedChunk(chunk=Chunk(id=f"c{page}", user_id="u", document_id="d", chunk_index=page, text="t",
                                              page_number=page), document_name="m.pdf", score=1.0)

        class Retriever:     # the relevant page is 1st at k=5 but missing at k=4, like the real bug
            def retrieve(self, user_id, query, appliance_id=None, k=5):
                return [rc(4), rc(1), rc(2), rc(3), rc(5)][:k] if k == 5 else [rc(1), rc(2), rc(3), rc(5)][:k]
        report = evaluate_retrieval(Retriever(), "u", [EvalCase("q", frozenset({("m.pdf", 4)}))], k_values=(1, 4, 5))
        self.assertEqual(report.recall_at, {1: 0.0, 4: 0.0, 5: 1.0})


if __name__ == "__main__":
    unittest.main()
