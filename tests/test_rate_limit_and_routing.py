"""Regressions from the first end-to-end run on the real Gorenje manual (Step 6):
(1) 'how much' manual questions were routed to the expense report; (2) unmatched manual questions went to
the LLM classifier, which sometimes picked 'general'; (3) Gemini's per-minute 429 says how long to wait,
but retries ignored it and failed again."""
import unittest

from gharcare.errors import DomainValidationError, LLMRateLimitError
from gharcare.llm import FakeLLMProvider, LLMMessage, PacedLLMProvider, RetryingLLMProvider
from gharcare.llm.factory import requests_per_minute
from gharcare.llm.langchain_adapter import map_exception, parse_retry_after
from gharcare.workflow.intent import Intent, classify_by_rules

# Trimmed copy of the real error text from the user's run.
GEMINI_PER_MINUTE_429 = (
    "Error calling model 'gemini-3.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
    "'message': 'You exceeded your current quota, please check your plan and billing details. \\n* Quota exceeded for "
    "metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, model: "
    "gemini-3.5-flash-lite, description: Generate content requests per minute limit exceeded on Free Tier.\\nPlease "
    "retry in 36.316450519s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '15'}, {'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '36s'}]}}")


class RoutingRegressionTests(unittest.TestCase):
    MANUAL_QUESTIONS = [
        "How much space is needed above the appliance?", "How much room space is needed for the amount of refrigerant?",
        "How much can the mains voltage vary?", "How much weight can the bottle holder take?",
        "How much does the appliance weigh?", "Can I store spray cans in the fridge?",
        "How long can eggs be kept in the fridge?", "Can I use a power strip behind the fridge?",
        "What is the retail price of this model?", "What information should I give the service centre?",
        "fridge pehli bar chalane se pehle kitni der khara rakhna chahiye?", "kitni der lagegi?",
        "Does turning the knob to off disconnect the power?"]
    EXPENSE_QUESTIONS = [
        "How much did I spend on repairs in the last 6 months?", "How much did the AC repair cost?",
        "How much did I pay for the fridge?", "Pichle 6 mahine mein kitna kharcha hua", "AC service kitne ki hui",
        "repair pe kitne paise lage"]

    def test_manual_questions_go_to_the_documents_without_an_llm_call(self):
        for q in self.MANUAL_QUESTIONS:
            with self.subTest(q=q):
                self.assertEqual(classify_by_rules(q), Intent.TROUBLESHOOTING)

    def test_money_questions_still_go_to_expenses(self):
        for q in self.EXPENSE_QUESTIONS:
            with self.subTest(q=q):
                self.assertEqual(classify_by_rules(q), Intent.EXPENSES)

    def test_non_questions_still_fall_back(self):
        self.assertIsNone(classify_by_rules("banana"))
        self.assertEqual(classify_by_rules("thanks!"), Intent.GENERAL)


class RetryAfterTests(unittest.TestCase):
    def test_parses_gemini_wait(self):
        self.assertAlmostEqual(parse_retry_after(GEMINI_PER_MINUTE_429), 36.316450519)
        self.assertEqual(parse_retry_after("retryDelay': '20s'"), 20.0)
        self.assertIsNone(parse_retry_after("429 Too Many Requests"))

    def test_per_minute_limit_is_retryable_with_the_wait(self):
        err = map_exception(RuntimeError(GEMINI_PER_MINUTE_429))
        self.assertIs(type(err), LLMRateLimitError)      # NOT the non-retryable daily-quota error
        self.assertAlmostEqual(err.retry_after, 36.316450519)

    def test_retry_waits_as_long_as_the_provider_asked(self):
        sleeps = []
        calls = iter([LLMRateLimitError("slow down", retry_after=36.3), None])

        class Flaky(FakeLLMProvider):
            def generate(self, messages, **kw):
                exc = next(calls)
                if exc:
                    raise exc
                return super().generate(messages, **kw)

        llm = RetryingLLMProvider(Flaky(["ok"]), sleep=sleeps.append)
        self.assertEqual(llm.generate([LLMMessage("user", "q")]).text, "ok")
        self.assertEqual(sleeps, [37.3])

    def test_requested_wait_is_still_capped(self):
        sleeps = []

        class AlwaysLimited(FakeLLMProvider):
            def generate(self, messages, **kw):
                raise LLMRateLimitError("x", retry_after=500)

        with self.assertRaises(LLMRateLimitError):
            RetryingLLMProvider(AlwaysLimited([]), max_attempts=2, max_delay=65, sleep=sleeps.append).generate([LLMMessage("user", "q")])
        self.assertEqual(sleeps, [65])


class PacingTests(unittest.TestCase):
    def _paced(self, rpm):
        self.t, self.sleeps = 100.0, []

        def sleep(s):
            self.sleeps.append(round(s, 6))
            self.t += s
        return PacedLLMProvider(FakeLLMProvider(responder=lambda m: "ok"), requests_per_minute=rpm,
                                clock=lambda: self.t, sleep=sleep)

    def test_spaces_calls_to_the_limit(self):
        llm = self._paced(12)          # one call per 5 s
        for _ in range(3):
            llm.generate([LLMMessage("user", "q")])
        self.assertEqual(self.sleeps, [5.0, 5.0])

    def test_no_wait_when_calls_are_already_far_apart(self):
        llm = self._paced(12)
        llm.generate([LLMMessage("user", "q")])
        self.t += 10
        llm.generate([LLMMessage("user", "q")])
        self.assertEqual(self.sleeps, [])

    def test_rpm_setting(self):
        self.assertEqual(requests_per_minute("gemini", {}), 12.0)
        self.assertIsNone(requests_per_minute("ollama", {}))
        self.assertEqual(requests_per_minute("gemini", {"GHARCARE_LLM_RPM": "5"}), 5.0)
        self.assertIsNone(requests_per_minute("gemini", {"GHARCARE_LLM_RPM": "0"}))
        for bad in ("fast", "-1"):
            with self.assertRaises(DomainValidationError):
                requests_per_minute("gemini", {"GHARCARE_LLM_RPM": bad})
        with self.assertRaises(ValueError):
            PacedLLMProvider(FakeLLMProvider(["x"]), requests_per_minute=0)


if __name__ == "__main__":
    unittest.main()
