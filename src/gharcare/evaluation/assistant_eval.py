"""End-to-end evaluation of the assistant (retrieval + LLM + grounding checks) on labeled questions.

Each answerable case names the (document, page) locations that answer it; unanswerable cases must be
refused. Outcomes (only the grounded troubleshooting path is scored; anything routed elsewhere is
reported separately so routing errors are visible instead of being counted as successes):

  answerable    correct          answered AND cited a relevant (document, page)
                wrong_citation   answered, but cited no relevant page          (grounding failure)
                unverified       safe fallback (raw passages) - not presented as an answer
                abstained        said it could not find it                      (missed answer)
  unanswerable  correct_abstention   said it could not find it
                safe_fallback        unverified -> raw passages, nothing asserted
                answered             answered with citations anyway            (HALLUCINATION RISK)
  either        safety_escalation  the hazard screen escalated (by design; not counted as a failure)
                routed_elsewhere / unavailable / other

Rates come with 95% Wilson confidence intervals: with ~25 questions the intervals are WIDE, and the
report says so instead of presenting point estimates as precise."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from gharcare.evaluation.retrieval import EvalCase, Location
from gharcare.workflow.assistant import AssistantResponse, HouseholdAssistant

_CITATION = re.compile(r"^(?P<doc>.+?), p\.(?P<page>\d+)")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_citation(citation: str) -> Location | None:
    m = _CITATION.match(citation)
    return (m.group("doc"), int(m.group("page"))) if m else None


def classify_outcome(case: EvalCase, response: AssistantResponse) -> tuple[str, tuple[Location, ...]]:
    cited = tuple(loc for loc in (parse_citation(c) for c in response.citations) if loc)
    if response.status == "safety_escalation":     # deterministic hazard screen fired: by design, not a failure
        return "safety_escalation", cited
    if response.intent != "troubleshooting":
        return ("unavailable" if response.status == "unavailable" else "routed_elsewhere"), cited
    status = response.status
    if status == "unavailable":
        return "unavailable", cited
    if case.answerable:
        if status == "answered":
            return ("correct" if any(loc in case.relevant for loc in cited) else "wrong_citation"), cited
        return {"unverified": "unverified", "not_found": "abstained"}.get(status, "other"), cited
    return {"not_found": "correct_abstention", "unverified": "safe_fallback", "answered": "answered"}.get(status, "other"), cited


@dataclass(frozen=True)
class CaseResult:
    question: str
    answerable: bool
    status: str
    intent: str
    outcome: str
    cited: tuple[Location, ...]
    llm_calls: int
    seconds: float


@dataclass
class AssistantEvalReport:
    results: list[CaseResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    planned: int = 0
    aborted: str | None = None      # why the run stopped early, if it did

    def _count(self, answerable: bool, outcome: str) -> int:
        return sum(1 for r in self.results if r.answerable == answerable and r.outcome == outcome)

    @property
    def n_answerable(self) -> int:
        return sum(1 for r in self.results if r.answerable)

    @property
    def n_unanswerable(self) -> int:
        return len(self.results) - self.n_answerable

    def rate(self, answerable: bool, outcome: str) -> tuple[int, int, float, float]:
        n = self.n_answerable if answerable else self.n_unanswerable
        k = self._count(answerable, outcome)
        lo, hi = wilson_interval(k, n)
        return k, n, lo, hi

    def to_dict(self) -> dict:
        def pack(answerable: bool, outcomes: Sequence[str]) -> dict:
            return {o: dict(zip(("count", "total", "ci_low", "ci_high"), self.rate(answerable, o))) for o in outcomes}
        n = max(len(self.results), 1)
        return {"meta": self.meta, "planned_questions": self.planned, "aborted": self.aborted,
                "answerable": pack(True, ["correct", "wrong_citation", "unverified", "abstained", "safety_escalation", "routed_elsewhere", "unavailable", "other"]),
                "unanswerable": pack(False, ["correct_abstention", "safe_fallback", "answered", "safety_escalation", "routed_elsewhere", "unavailable", "other"]),
                "avg_llm_calls_per_question": sum(r.llm_calls for r in self.results) / n,
                "avg_seconds_per_question": sum(r.seconds for r in self.results) / n,
                "cases": [dict(question=r.question, answerable=r.answerable, status=r.status, intent=r.intent,
                               outcome=r.outcome, cited=[list(c) for c in r.cited]) for r in self.results]}

    def format(self) -> str:
        def line(label: str, answerable: bool, outcome: str) -> str:
            k, n, lo, hi = self.rate(answerable, outcome)
            return f"  {label:<34}{k:>3}/{n:<3} {100 * k / n if n else 0:5.1f}%   (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"
        d = self.to_dict()
        out = [f"End-to-end evaluation: {self.n_answerable} answerable, {self.n_unanswerable} unanswerable questions"
               f" ({len(self.results)} of {self.planned} planned)"]
        if self.aborted:
            out += [f"*** RUN STOPPED EARLY: {self.aborted}",
                    "*** Results below cover only the completed questions. Re-run the same command later (or with another model): "
                    "answers already received are cached, so only the remaining questions use quota."]
        out += [f"  {k}: {v}" for k, v in self.meta.items()]
        out += ["", "ANSWERABLE questions",
                line("correct (cited a relevant page)", True, "correct"), line("wrong citation (grounding failure)", True, "wrong_citation"),
                line("unverified (safe fallback)", True, "unverified"), line("abstained (missed answer)", True, "abstained"),
                line("safety escalation (by design)", True, "safety_escalation"), line("routed elsewhere", True, "routed_elsewhere"), "", "UNANSWERABLE questions",
                line("correct abstention", False, "correct_abstention"), line("safe fallback (unverified)", False, "safe_fallback"),
                line("ANSWERED anyway (hallucination risk)", False, "answered"),
                line("safety escalation (by design)", False, "safety_escalation"), line("routed elsewhere", False, "routed_elsewhere"), "",
                f"Average LLM calls per question: {d['avg_llm_calls_per_question']:.1f}; average time: {d['avg_seconds_per_question']:.1f}s",
                f"Provider errors (unavailable): {self._count(True, 'unavailable') + self._count(False, 'unavailable')}"]
        bad = [r for r in self.results if r.outcome in ("wrong_citation", "abstained", "answered", "routed_elsewhere", "unavailable", "other")]
        if bad:
            out += ["", "Cases needing a look:"] + [f"  [{r.outcome}] {r.question}" for r in bad]
        out += ["", "Note: small samples give wide confidence intervals; treat these as a first measurement, not a guarantee."]
        return "\n".join(out)


def evaluate_assistant(assistant: HouseholdAssistant, user_id: str, cases: Sequence[EvalCase], *, appliance_id: str | None = None,
                       meter=None, pause_seconds: float = 0.0, sleep: Callable[[float], None] = time.sleep,
                       meta: dict | None = None, max_llm_calls: int | None = None, max_consecutive_unavailable: int = 2,
                       on_result: Callable[[int, int, CaseResult], None] | None = None) -> AssistantEvalReport:
    report = AssistantEvalReport(meta=dict(meta or {}), planned=len(cases))
    unavailable_in_a_row = 0
    for index, case in enumerate(cases):
        if max_llm_calls is not None and meter is not None and meter.calls + 3 > max_llm_calls:   # a question needs up to 3 calls
            report.aborted = f"LLM call budget reached ({meter.calls} of {max_llm_calls} calls used)"
            break
        if index and pause_seconds:
            sleep(pause_seconds)             # be polite to free-tier rate limits
        calls_before = meter.calls if meter else 0
        started = time.perf_counter()
        response = assistant.ask(user_id, case.question, appliance_id=appliance_id)
        outcome, cited = classify_outcome(case, response)
        result = CaseResult(case.question, case.answerable, response.status, response.intent, outcome, cited,
                            (meter.calls - calls_before) if meter else 0, time.perf_counter() - started)
        report.results.append(result)
        if on_result:
            on_result(index + 1, len(cases), result)
        unavailable_in_a_row = unavailable_in_a_row + 1 if outcome == "unavailable" else 0
        if unavailable_in_a_row >= max_consecutive_unavailable:
            report.aborted = (f"the LLM provider failed {unavailable_in_a_row} questions in a row (daily quota used up, outage or bad "
                              "key: see the WARNING lines above for the real reason)")
            break
    return report
