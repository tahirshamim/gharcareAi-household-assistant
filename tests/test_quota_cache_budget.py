import json
import tempfile
import unittest
from pathlib import Path

from gharcare.errors import LLMQuotaExhaustedError, LLMRateLimitError, LLMError, LLMUnavailableError
from gharcare.evaluation import EvalCase, evaluate_assistant
from gharcare.llm import CachingLLMProvider, FakeLLMProvider, LLMMessage, MeteredLLMProvider, RetryingLLMProvider
from gharcare.llm.langchain_adapter import map_exception
from gharcare.workflow import HouseholdAssistant
from tests.test_workflow import GOOD_JUDGE, MANUAL, AssistantTestCase, U

M = [LLMMessage("system", "s"), LLMMessage("user", "q")]
GOOGLE_DAILY = ("Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. You exceeded your current quota. "
                "Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash, "
                "description: Generate content requests per day limit exceeded on Free Tier. Please retry in 45m27s.")


class QuotaMappingTests(unittest.TestCase):
    def test_daily_quota_is_distinct_from_a_per_minute_rate_limit(self):
        daily = map_exception(RuntimeError(GOOGLE_DAILY))
        self.assertIsInstance(daily, LLMQuotaExhaustedError)
        self.assertNotIsInstance(daily, LLMRateLimitError)
        self.assertIn("midnight Pacific", str(daily))
        self.assertIsInstance(map_exception(RuntimeError("429 Too Many Requests: requests per minute exceeded")), LLMRateLimitError)

    def test_quota_exhaustion_is_never_retried(self):
        fake = FakeLLMProvider([LLMQuotaExhaustedError("daily"), "never reached"])
        llm = RetryingLLMProvider(fake, sleep=lambda s: self.fail("must not sleep on a daily quota"))
        with self.assertRaises(LLMQuotaExhaustedError):
            llm.generate(M)
        self.assertEqual(len(fake.calls), 1)


class CacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "sub" / "cache.jsonl"

    def test_hit_miss_and_persistence_across_instances(self):
        fake = FakeLLMProvider(["answer one"])
        cache = CachingLLMProvider(fake, self.path, namespace="m1")
        self.assertEqual(cache.generate(M).text, "answer one")
        self.assertEqual(cache.generate(M).text, "answer one")
        self.assertEqual((cache.hits, cache.misses, len(fake.calls)), (1, 1, 1))
        again = CachingLLMProvider(FakeLLMProvider([]), self.path, namespace="m1")      # a new process: empty fake would raise
        self.assertEqual(again.generate(M).text, "answer one")

    def test_prompt_or_model_changes_miss_the_cache(self):
        CachingLLMProvider(FakeLLMProvider(["a"]), self.path, namespace="m1").generate(M)
        other_model = CachingLLMProvider(FakeLLMProvider(["b"]), self.path, namespace="m2")
        self.assertEqual(other_model.generate(M).text, "b")
        changed = CachingLLMProvider(FakeLLMProvider(["c"]), self.path, namespace="m1")
        self.assertEqual(changed.generate([LLMMessage("system", "s"), LLMMessage("user", "different")]).text, "c")

    def test_failures_are_not_cached_and_corrupt_lines_are_skipped(self):
        cache = CachingLLMProvider(FakeLLMProvider([LLMUnavailableError("down"), "ok"]), self.path)
        with self.assertRaises(LLMUnavailableError):
            cache.generate(M)
        self.assertEqual(cache.generate(M).text, "ok")
        self.path.write_text(self.path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
        self.assertEqual(CachingLLMProvider(FakeLLMProvider([]), self.path).generate(M).text, "ok")


class EvalBudgetTests(AssistantTestCase):
    CASE = EvalCase("How do I clear ice build-up in the freezer?", frozenset({(MANUAL.name, 3)}))

    def assistant(self, llm):
        return HouseholdAssistant(self.app.appliances, self.svc, self.rag.retriever, llm, backend="local")

    @staticmethod
    def model(messages):
        return GOOD_JUDGE if messages[0].content.startswith("You are a strict") else "Do this [1]."

    def test_stops_early_when_the_provider_keeps_failing(self):
        llm = MeteredLLMProvider(FakeLLMProvider([LLMQuotaExhaustedError("daily quota used up")] * 50))
        report = evaluate_assistant(self.assistant(llm), U, [self.CASE] * 6, appliance_id=self.fridge.id, meter=llm)
        self.assertEqual(len(report.results), 2)                                   # stopped after 2 in a row, not 6
        self.assertIn("2 questions in a row", report.aborted)
        text = report.format()
        self.assertIn("RUN STOPPED EARLY", text)
        self.assertIn("2 of 6 planned", text)
        self.assertEqual(report.to_dict()["planned_questions"], 6)

    def test_call_budget_stops_before_exceeding_it(self):
        llm = MeteredLLMProvider(FakeLLMProvider(responder=self.model))
        report = evaluate_assistant(self.assistant(llm), U, [self.CASE] * 6, appliance_id=self.fridge.id, meter=llm, max_llm_calls=5)
        self.assertEqual((len(report.results), llm.calls), (2, 4))                 # each question uses 2 calls; a 3rd would risk >5
        self.assertIn("budget", report.aborted)

    def test_an_intermittent_failure_does_not_abort(self):
        llm = MeteredLLMProvider(FakeLLMProvider([LLMUnavailableError("blip")] + ["Do this [1].", GOOD_JUDGE] * 3))
        report = evaluate_assistant(self.assistant(llm), U, [self.CASE] * 3, appliance_id=self.fridge.id, meter=llm)
        self.assertIsNone(report.aborted)
        self.assertEqual(len(report.results), 3)


if __name__ == "__main__":
    unittest.main()
