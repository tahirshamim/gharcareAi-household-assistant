import unittest

from gharcare.evaluation import EvalCase, classify_outcome, evaluate_assistant, load_eval_cases, parse_citation, wilson_interval
from gharcare.llm import FakeLLMProvider, MeteredLLMProvider
from gharcare.workflow import AssistantResponse, HouseholdAssistant
from tests.test_workflow import GOOD_JUDGE, MANUAL, AssistantTestCase, U

D = "xr200_sample_manual_SYNTHETIC.txt"


def response(status, intent="troubleshooting", citations=()):
    return AssistantResponse(status, "text", tuple(citations), intent, ())


class StatsTests(unittest.TestCase):
    def test_wilson_interval_known_values(self):
        lo, hi = wilson_interval(9, 10)
        self.assertAlmostEqual(lo, 0.596, delta=0.002)
        self.assertAlmostEqual(hi, 0.982, delta=0.002)
        self.assertEqual(wilson_interval(0, 0)[0], 0.0)
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)
        self.assertLess(wilson_interval(0, 5)[0], 1e-9)

    def test_citation_parsing(self):
        self.assertEqual(parse_citation("a, b.txt, p.12 (Section X)"), ("a, b.txt", 12))
        self.assertEqual(parse_citation("manual.pdf, p.3"), ("manual.pdf", 3))
        self.assertIsNone(parse_citation("no page here"))


class OutcomeTests(unittest.TestCase):
    ANS = EvalCase("q", frozenset({(D, 3)}))
    UNANS = EvalCase("q", answerable=False)

    def test_answerable_outcomes(self):
        cases = [(response("answered", citations=[f"{D}, p.3 (X)"]), "correct"), (response("answered", citations=[f"{D}, p.4 (X)"]), "wrong_citation"),
                 (response("answered", citations=[f"other.pdf, p.3"]), "wrong_citation"), (response("unverified"), "unverified"),
                 (response("not_found"), "abstained"), (response("unavailable"), "unavailable"),
                 (response("needs_clarification"), "other"), (response("answered", intent="expenses"), "routed_elsewhere"),
                 (response("safety_escalation", intent="safety"), "safety_escalation")]
        for resp, expected in cases:
            with self.subTest(status=resp.status, intent=resp.intent, cites=resp.citations):
                self.assertEqual(classify_outcome(self.ANS, resp)[0], expected)

    def test_unanswerable_outcomes(self):
        cases = [(response("not_found"), "correct_abstention"), (response("unverified"), "safe_fallback"),
                 (response("answered", citations=[f"{D}, p.3"]), "answered"), (response("answered", intent="general"), "routed_elsewhere"),
                 (response("unavailable"), "unavailable"), (response("safety_escalation", intent="safety"), "safety_escalation")]
        for resp, expected in cases:
            with self.subTest(status=resp.status, intent=resp.intent):
                self.assertEqual(classify_outcome(self.UNANS, resp)[0], expected)


class HarnessTests(AssistantTestCase):
    @staticmethod
    def model(messages):
        system, user = messages[0].content, messages[-1].content
        if system.startswith("You are a strict fact checker"):
            return GOOD_JUDGE
        if system.startswith("Classify"):
            return "troubleshooting"
        return "NOT_FOUND" if "litres" in user.split("MANUAL EXCERPTS")[0] else "Do this as the manual says [1]."

    def run_eval(self, cases, pause=0.0, sleeps=None):
        meter = MeteredLLMProvider(FakeLLMProvider(responder=self.model))
        assistant = HouseholdAssistant(self.app.appliances, self.svc, self.rag.retriever, meter, backend="local")
        return evaluate_assistant(assistant, U, cases, appliance_id=self.fridge.id, meter=meter, pause_seconds=pause,
                                  sleep=(sleeps.append if sleeps is not None else (lambda s: None)), meta={"llm": "fake"}), meter

    def test_scores_correct_wrong_and_abstained_cases(self):
        cases = [EvalCase("How do I clear ice build-up in the freezer?", frozenset({(MANUAL.name, 3)})),       # [1] is page 3 -> correct
                 EvalCase("How do I clear ice build-up in the freezer?", frozenset({(MANUAL.name, 2)})),       # cites p3, wants p2 -> wrong
                 EvalCase("What is the freezer capacity in litres?", answerable=False),                         # model refuses -> correct
                 EvalCase("How do I clear ice build-up in the freezer?", answerable=False)]                     # model answers -> hallucination risk
        report, meter = self.run_eval(cases)
        self.assertEqual([r.outcome for r in report.results], ["correct", "wrong_citation", "correct_abstention", "answered"])
        d = report.to_dict()
        self.assertEqual((d["answerable"]["correct"]["count"], d["answerable"]["correct"]["total"]), (1, 2))
        self.assertEqual(d["unanswerable"]["answered"]["count"], 1)
        self.assertEqual(sum(r.llm_calls for r in report.results), meter.calls)               # per-question metering adds up
        text = report.format()
        for expected in ("1/2", "ANSWERED anyway (hallucination risk)", "Cases needing a look", "wide", "llm: fake"):
            self.assertIn(expected, text.replace("WIDE", "wide"))

    def test_progress_callback_and_call_timing_hooks(self):
        seen, timings = [], []
        meter = MeteredLLMProvider(FakeLLMProvider(responder=self.model), on_call=lambda secs, ok: timings.append(ok))
        assistant = HouseholdAssistant(self.app.appliances, self.svc, self.rag.retriever, meter, backend="local")
        cases = [EvalCase("How do I clear ice build-up in the freezer?", frozenset({(MANUAL.name, 3)}))] * 2
        evaluate_assistant(assistant, U, cases, appliance_id=self.fridge.id, meter=meter, on_result=lambda i, n, r: seen.append((i, n, r.outcome)))
        self.assertEqual(seen, [(1, 2, "correct"), (2, 2, "correct")])
        self.assertEqual(timings, [True] * meter.calls)

    def test_a_hazard_question_is_reported_as_a_designed_escalation_not_a_problem(self):
        case = EvalCase("What should I do if I smell gas?", frozenset({(MANUAL.name, 1)}))
        report, _ = self.run_eval([case])
        self.assertEqual(report.results[0].outcome, "safety_escalation")
        text = report.format()
        self.assertIn("safety escalation (by design)", text)
        self.assertNotIn("Cases needing a look", text)

    def test_pacing_between_questions(self):
        sleeps = []
        self.run_eval([EvalCase("q1 ice", frozenset({(MANUAL.name, 3)}))] * 3, pause=1.5, sleeps=sleeps)
        self.assertEqual(sleeps, [1.5, 1.5])                                                   # between questions, not before the first

    def test_bundled_question_set_is_well_formed(self):
        cases = load_eval_cases(MANUAL.parent / "eval_questions.json")
        self.assertGreaterEqual(sum(c.answerable for c in cases), 15)
        self.assertGreaterEqual(sum(not c.answerable for c in cases), 8)
        self.assertEqual(len({c.question for c in cases}), len(cases))                         # no duplicate questions
        known_pages = {1, 2, 3, 4, 5}
        for c in cases:
            self.assertTrue(all(doc == MANUAL.name and page in known_pages for doc, page in c.relevant))
            if c.must_contain:                                                                 # the labeled evidence really is in the manual
                self.assertIn(c.must_contain.casefold(), MANUAL.read_text(encoding="utf-8").casefold())


if __name__ == "__main__":
    unittest.main()
