import unittest
from pathlib import Path

from gharcare.domain import Appliance
from gharcare.errors import DomainValidationError, LLMUnavailableError, NotFoundError, WorkflowError
from gharcare.llm import FakeLLMProvider
from gharcare.rag import build_rag
from gharcare.retrieval import HashingEmbedder, InMemoryVectorStore
from gharcare.workflow import HouseholdAssistant, LocalRunner, WorkflowSpec, check_hazard, langgraph_available, make_runner, END
from gharcare.workflow.intent import Intent, classify_by_rules, extract_categories, extract_months, match_appliances
from gharcare.services import ExpenseCategory
from tests.test_services import ServiceTestCase, U

MANUAL = Path(__file__).parent.parent / "examples" / "sample_docs" / "xr200_sample_manual_SYNTHETIC.txt"
GOOD_JUDGE = '{"supported": true, "unsupported_claims": []}'
ICE_Q = "How do I clear ice build-up in the freezer?"
GROUNDED = "Unplug the fridge, remove the food and leave the doors open for about 6 hours [1]."


class AssistantTestCase(ServiceTestCase):
    backend = "local"

    def setUp(self):
        super().setUp()
        self.rag = build_rag(self.app, HashingEmbedder(), InMemoryVectorStore())
        self.rag.ingestion.ingest(U, MANUAL.name, MANUAL.read_bytes(), "manual", self.fridge.id)

    def make(self, script=()):
        self.llm = FakeLLMProvider(list(script))
        return HouseholdAssistant(self.app.appliances, self.svc, self.rag.retriever, self.llm, backend=self.backend)


class SafetyTests(unittest.TestCase):
    HAZARD = ["AC se jalne ki smell aa rahi hai aur spark nikla", "I can smell gas near the stove", "there is a burning smell from the fridge",
              "the plug is sparking", "fridge se dhuan nikal raha hai", "geyser mein se chingari nikli", "I got an electric shock",
              "water on the socket behind the washer", "فریج سے دھواں نکل رہا ہے", "AC کے پلگ سے چنگاری نکلی", "gas leak ho rahi hai"]
    BENIGN = ["fridge mein barf jam rahi hai", "AC needs a gas top up", "What does error code E4 mean?", "Is it safe to use an extension cord?",
              "water leaking at the bottom of the fridge", "How much did I spend on repairs?", "washer is making a loud noise"]

    def test_hazards_are_detected(self):
        for text in self.HAZARD:
            with self.subTest(text=text):
                self.assertIsNotNone(check_hazard(text))

    def test_normal_questions_are_not_escalated(self):
        for text in self.BENIGN:
            with self.subTest(text=text):
                self.assertIsNone(check_hazard(text))


class IntentTests(unittest.TestCase):
    def test_rules(self):
        cases = {"How much did I spend on repairs in the last 6 months?": Intent.EXPENSES, "Pichle 6 mahine mein kitna kharcha hua": Intent.EXPENSES,
                 "Which warranties will expire soon?": Intent.WARRANTY, "Is my fridge still under warranty?": Intent.WARRANTY,
                 "warranty claim ke liye kya chahiye": Intent.TROUBLESHOOTING, "Which maintenance is overdue?": Intent.MAINTENANCE,
                 "Generate my household report": Intent.REPORT, "hello": Intent.GENERAL, "fridge mein barf jam rahi hai": Intent.TROUBLESHOOTING,
                 "What does error code E4 mean?": Intent.TROUBLESHOOTING, "banana": None}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify_by_rules(text), expected)

    def test_parameters(self):
        self.assertEqual(extract_months("last 6 months"), 6)
        self.assertEqual(extract_months("pichle 3 mahine"), 3)
        self.assertEqual(extract_months("in the last year"), 12)
        self.assertEqual(extract_months("2 years"), 24)
        self.assertIsNone(extract_months("how much did I spend"))
        self.assertEqual(extract_categories("repair costs"), [ExpenseCategory.REPAIR])
        self.assertEqual(set(extract_categories("repairs and maintenance")), {ExpenseCategory.REPAIR, ExpenseCategory.MAINTENANCE})
        self.assertIsNone(extract_categories("everything"))

    def test_appliance_matching(self):
        a = [Appliance(user_id="u", name="Fridge", brand="Haier"), Appliance(user_id="u", name="Split AC", brand="Gree"),
             Appliance(user_id="u", name="Window AC", brand="Orient")]
        names = lambda text: sorted(x.name for x in match_appliances(a, text))
        self.assertEqual(names("ice in the freezer"), ["Fridge"])             # synonym match
        self.assertEqual(names("my haier is leaking"), ["Fridge"])             # brand match
        self.assertEqual(names("my AC is noisy"), ["Split AC", "Window AC"])   # tie -> must ask
        self.assertEqual(names("my split AC is noisy"), ["Split AC"])          # exact word breaks the tie
        self.assertEqual(names("how much did I spend"), [])


class AssistantTests(AssistantTestCase):
    def test_hazard_short_circuits_before_any_model_call(self):
        r = self.make().ask(U, "AC se jalne ki smell aa rahi hai aur spark nikla")
        self.assertEqual((r.status, r.trace), ("safety_escalation", ("safety",)))
        self.assertIn("qualified technician", r.answer)
        self.assertEqual(self.llm.calls, [])

    def test_hazard_is_escalated_even_if_the_llm_is_down(self):
        a = self.make([LLMUnavailableError("down")] * 5)
        self.assertEqual(a.ask(U, "there is smoke coming from the fridge").status, "safety_escalation")

    def test_expenses_answer_is_exact_and_uses_no_llm(self):
        r = self.make().ask(U, "How much did I spend on repairs in the last 6 months?")
        self.assertEqual(r.status, "answered")
        self.assertIn("PKR 4,700.00", r.answer)
        self.assertIn("2025-08-20 to 2026-02-20", r.answer)
        self.assertEqual((r.trace, self.llm.calls), (("safety", "classify", "resolve", "structured"), []))

    def test_missing_period_is_stated_as_an_assumption(self):
        r = self.make().ask(U, "How much did I spend on repairs?")
        self.assertIn("didn't give a period", r.answer)
        self.assertIn("PKR 4,700.00", r.answer)

    def test_warranty_answers(self):
        a = self.make()
        self.assertIn("22 days left", a.ask(U, "Which warranties will expire soon?").answer)
        r = a.ask(U, "Is my fridge still under warranty?")
        self.assertIn("A warranty period is active until 2026-03-14", r.answer)
        self.assertIn("No coverage terms are recorded", r.answer)           # never claims fault coverage

    def test_maintenance_answer(self):
        r = self.make().ask(U, "Which maintenance tasks are overdue?")
        self.assertIn("Gree Split AC: AC service", r.answer)
        self.assertIn("19 days overdue", r.answer)

    def test_report_is_generated_as_an_artifact(self):
        r = self.make().ask(U, "Generate my household report")
        self.assertIn("household report", r.answer)
        rel = r.answer.split("(")[1].split(")")[0]
        self.assertTrue(self.app.artifacts.exists(U, rel))

    def test_grounded_troubleshooting_answer_with_citations_and_records(self):
        r = self.make([GROUNDED, GOOD_JUDGE]).ask(U, ICE_Q)
        self.assertEqual(r.status, "answered")
        self.assertEqual(r.trace, ("safety", "classify", "resolve", "retrieve", "history", "generate", "verify"))
        self.assertTrue(r.citations[0].startswith("xr200_sample_manual_SYNTHETIC.txt, p.3"))
        self.assertIn("Sources:", r.answer)
        self.assertIn("Similar past repair: 2025-11-02 (110 days ago)", r.answer)   # from records, not the model
        self.assertIn("Warranty:", r.answer)
        self.assertEqual(len(self.llm.calls), 2)
        prompt = self.llm.calls[0]
        self.assertIn("untrusted", prompt[0].content)
        self.assertIn('<excerpt id="1"', prompt[1].content)

    def test_uncited_answer_is_not_presented_as_grounded(self):
        # The judge says "supported" but names no supporting excerpt, so there is nothing to cite.
        r = self.make(["Just defrost it.", GOOD_JUDGE]).ask(U, ICE_Q)
        self.assertEqual(r.status, "unverified")
        self.assertIn("couldn't fully verify", r.answer)
        self.assertEqual(len(self.llm.calls), 2)

    def test_out_of_range_citation_is_rejected(self):
        r = self.make(["Do this [9].", GOOD_JUDGE]).ask(U, ICE_Q)
        self.assertEqual((r.status, len(self.llm.calls)), ("unverified", 1))

    def test_judge_can_reject_an_answer(self):
        r = self.make([GROUNDED, '{"supported": false, "unsupported_claims": ["6 hours"]}']).ask(U, ICE_Q)
        self.assertEqual(r.status, "unverified")

    def test_unparseable_judge_output_means_unverified(self):
        for junk in ("I think it is fine", "{not json}", '{"supported": "yes"}'):
            with self.subTest(junk=junk):
                self.assertEqual(self.make([GROUNDED, junk]).ask(U, ICE_Q).status, "unverified")

    def test_model_can_declare_not_found(self):
        # A question is routed to the documents by rule now, so no classifier call precedes the draft.
        r = self.make(["NOT_FOUND"]).ask(U, "What is the freezer capacity in litres?")
        self.assertEqual(r.status, "not_found")
        self.assertIn("couldn't find this in your documents", r.answer)

    def test_llm_outage_degrades_to_raw_passages(self):
        r = self.make([LLMUnavailableError("down")]).ask(U, ICE_Q)
        self.assertEqual(r.status, "unavailable")
        self.assertIn("unavailable", r.answer)
        self.assertIn("xr200_sample_manual_SYNTHETIC.txt", r.answer)

    def test_appliance_without_documents_never_calls_the_model(self):
        r = self.make().ask(U, "Why is my washer not draining?")
        self.assertEqual((r.status, self.llm.calls), ("not_found", []))
        self.assertIn("From your records", r.answer)

    def test_ambiguous_appliance_asks_instead_of_guessing(self):
        self.app.appliances.add(Appliance(user_id=U, name="Window AC", brand="Orient", location="Lounge"))
        r = self.make().ask(U, "my AC is making noise")
        self.assertEqual((r.status, self.llm.calls), ("needs_clarification", []))
        self.assertIn("Gree Split AC", r.answer)
        self.assertIn("Orient Window AC (Lounge)", r.answer)
        specific = self.make().ask(U, "my split AC is making noise")
        self.assertEqual(specific.status, "not_found")                      # resolved, but no AC manual uploaded

    def test_unmatched_question_with_several_appliances_asks_which(self):
        self.assertEqual(self.make().ask(U, "it is not working").status, "needs_clarification")

    def test_single_appliance_is_used_automatically_and_empty_account_is_told_so(self):
        self.app.appliances.add(Appliance(user_id="solo", name="Fridge"))
        r = self.make().ask("solo", "Why is it not cooling?")
        self.assertEqual((r.status, "retrieve" in r.trace), ("not_found", True))
        self.assertIn("haven't added any appliances", self.make().ask("nobody", ICE_Q).answer)

    def test_explicit_appliance_overrides_text_matching_and_is_ownership_checked(self):
        a = self.make(["Loud rattling usually means the appliance is not level [1].", GOOD_JUDGE])
        self.assertEqual(a.ask(U, "Why is it making noise?", appliance_id=self.fridge.id).status, "answered")
        with self.assertRaises(NotFoundError):
            a.ask("bilal", "Why is it making noise?", appliance_id=self.fridge.id)

    def test_other_users_documents_are_never_reachable(self):
        self.app.appliances.add(Appliance(user_id="bilal", name="Fridge"))
        r = self.make().ask("bilal", ICE_Q)
        self.assertEqual((r.status, self.llm.calls), ("not_found", []))

    def test_greeting_and_unclear_message_fallbacks(self):
        self.assertEqual(self.make().ask(U, "hello").trace, ("safety", "classify", "resolve", "help"))
        self.assertIn("I can help", self.make(["general"]).ask(U, "banana").answer)
        self.assertEqual(self.make(["xyz"]).ask(U, "banana").status, "needs_clarification")      # garbage -> grounded path
        self.assertEqual(self.make([LLMUnavailableError("down")]).ask(U, "banana").status, "needs_clarification")

    def test_input_validation(self):
        a = self.make()
        for bad in ("", "   ", "x" * 2001):
            with self.subTest(len=len(bad)), self.assertRaises(DomainValidationError):
                a.ask(U, bad)
        with self.assertRaises(DomainValidationError):
            a.ask("../evil", "hello")


class FailureVisibilityTests(AssistantTestCase):
    def test_swallowed_llm_errors_are_logged_with_the_real_reason(self):
        with self.assertLogs("gharcare.workflow.assistant", level="WARNING") as logs:
            r = self.make([LLMUnavailableError("404 model not found")]).ask(U, ICE_Q)
        self.assertEqual(r.status, "unavailable")
        self.assertTrue(any("404 model not found" in line for line in logs.output))

    def test_judge_failures_are_logged_too(self):
        with self.assertLogs("gharcare.workflow.assistant", level="WARNING") as logs:
            self.make([GROUNDED, "not json at all"]).ask(U, ICE_Q)
        self.assertTrue(any("grounding judge" in line for line in logs.output))


class WorkflowSpecTests(unittest.TestCase):
    def test_validation_errors(self):
        ok = lambda s: {"x": 1}
        with self.assertRaises(WorkflowError):
            WorkflowSpec(entry="missing", nodes={"a": ok}, edges={"a": END}).validate()
        with self.assertRaises(WorkflowError):
            WorkflowSpec(entry="a", nodes={"a": ok}).validate()                          # no outgoing edge
        with self.assertRaises(WorkflowError):
            WorkflowSpec(entry="a", nodes={"a": ok}, edges={"a": "nowhere"}).validate()
        with self.assertRaises(WorkflowError):
            WorkflowSpec(entry="a", nodes={"a": ok}, edges={"a": END}, branches={"a": (lambda s: "x", {"x": END})}).validate()

    def test_branching_and_bad_router_key(self):
        spec = WorkflowSpec(entry="a", nodes={"a": lambda s: {"n": 1}, "b": lambda s: {"hit": "b"}},
                            edges={"b": END}, branches={"a": (lambda s: "go", {"go": "b"})})
        self.assertEqual(LocalRunner(spec).invoke({})["hit"], "b")
        bad = WorkflowSpec(entry="a", nodes={"a": lambda s: {}}, branches={"a": (lambda s: "nope", {"go": END})})
        with self.assertRaises(WorkflowError):
            LocalRunner(bad).invoke({})

    def test_loops_are_stopped(self):
        spec = WorkflowSpec(entry="a", nodes={"a": lambda s: {}}, edges={"a": "a"})
        with self.assertRaisesRegex(WorkflowError, "exceeded"):
            LocalRunner(spec, max_steps=5).invoke({})

    def test_backend_selection(self):
        spec = WorkflowSpec(entry="a", nodes={"a": lambda s: {}}, edges={"a": END})
        self.assertEqual(make_runner(spec, dict, "local")[1], "local")
        self.assertEqual(make_runner(spec, dict, "auto")[1], "langgraph" if langgraph_available() else "local")
        with self.assertRaises(WorkflowError):
            make_runner(spec, dict, "magic")
        if not langgraph_available():
            with self.assertRaisesRegex(WorkflowError, "pip install langgraph"):
                make_runner(spec, dict, "langgraph")


@unittest.skipUnless(langgraph_available(), "langgraph not installed (pip install langgraph)")
class LangGraphParityTests(AssistantTestCase):
    """The SAME scenarios through the real LangGraph backend must give identical responses."""
    backend = "langgraph"

    def test_matches_the_local_backend(self):
        scenarios = [("there is smoke coming from the fridge", []), ("How much did I spend on repairs in the last 6 months?", []),
                     (ICE_Q, [GROUNDED, GOOD_JUDGE]), (ICE_Q, ["no citation here", GOOD_JUDGE]), ("hello", []), ("it is not working", [])]
        for question, script in scenarios:
            with self.subTest(question=question):
                graph = self.make(script)
                self.assertEqual(graph.backend, "langgraph")
                local = HouseholdAssistant(self.app.appliances, self.svc, self.rag.retriever, FakeLLMProvider(list(script)), backend="local")
                self.assertEqual(graph.ask(U, question), local.ask(U, question))


if __name__ == "__main__":
    unittest.main()
