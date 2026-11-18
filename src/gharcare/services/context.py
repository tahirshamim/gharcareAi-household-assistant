"""Repair-history + coverage context for one appliance: the deterministic 'history node' input.

Everything here comes from stored records. Similar-repair detection is a KEYWORD heuristic (Dice
overlap of meaningful words). Words are first mapped to shared CONCEPTS through a small glossary of
appliance-fault vocabulary in English, Roman Urdu and Urdu script, so 'barf jam gayi' matches an 'ice
build-up' record and 'frost' matches 'ice'. Paraphrases outside the glossary are still missed. The assistant
must present matches as 'similar to', never as proof of the same fault."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable

from gharcare.domain import Appliance, MaintenanceLog, RepairRecord
from gharcare.domain.validation import validate_identifier
from gharcare.repositories.base import ApplianceRepository, MaintenanceLogRepository, RepairRepository
from gharcare.retrieval.text import tokenize
from gharcare.services.formatting import format_money
from gharcare.services.maintenance import MaintenanceService, TaskView
from gharcare.services.warranty import Coverage, CoverageState, WarrantyService

_STOP = frozenset("a an and the in on of to is it my our for with at by from has have had not no was were be "
                  "again still kar rahi raha raha hai hain mein ki ka ke se ko ye wo bhi aur pe par kya "
                  "karun karoon karein karna gayi gaya gaye rahe hota hoti hogi bahut zyada kuch abhi".split())


# word -> concept. Deliberately small and reviewable; extend it from real repair records.
_GLOSSARY_GROUPS = {
    # history is matched within ONE appliance, so its compartment words should agree rather than compete
    "fridge": "fridge refrigerator freezer فریج فریزر",
    "ice": "ice frost frosting frozen barf baraf برف",
    "cold": "cold cool cooling chill thanda thandi thande ٹھنڈا ٹھنڈی",
    "hot": "hot warm heat heating overheating garam garm garmi گرم",
    "water": "water pani paani پانی",
    "leak": "leak leaks leaking leakage drip dripping tapak tapakna tapak-raha ٹپک",
    "noise": "noise noisy sound sounds loud awaz aawaz awaaz shor آواز شور",
    "power": "power electricity electric bijli current بجلی",
    "smell": "smell smells odour odor stink badbu بدبو",
    "door": "door doors darwaza darwaaza دروازہ",
    "gasket": "gasket seal seals rubber",
    "broken": "broken faulty fault damaged kharab خراب",
    "stopped": "stopped dead band",
    "light": "light lights bulb lamp batti بتی",
}
_CONCEPT = {word: concept for concept, words in _GLOSSARY_GROUPS.items() for word in words.split()}


def _keywords(text: str) -> set[str]:
    return {_CONCEPT.get(t, t) for t in tokenize(text)
            if t not in _STOP and (len(t) >= 3 or any(c.isdigit() for c in t) or t in _CONCEPT)}


def similarity(a: str, b: str) -> float:
    """Dice coefficient of the two texts' keyword sets (0.0 to 1.0)."""
    ka, kb = _keywords(a), _keywords(b)
    return 2 * len(ka & kb) / (len(ka) + len(kb)) if ka and kb else 0.0


@dataclass(frozen=True)
class SimilarRepair:
    repair: RepairRecord
    similarity: float
    days_ago: int


@dataclass(frozen=True)
class ApplianceContext:
    appliance: Appliance
    today: date
    coverage: Coverage
    similar_repairs: tuple[SimilarRepair, ...]
    repair_count: int
    total_repair_cost: Decimal
    overdue_tasks: tuple[TaskView, ...]
    last_service: MaintenanceLog | None

    def facts_text(self) -> str:
        """Compact, factual summary for the LLM prompt. Unknown things are stated as 'not recorded',
        never filled in."""
        a = self.appliance
        head = f"Appliance: {a.display_name}"
        if a.model:
            head += f" (model {a.model})"
        lines = [head + (f", location: {a.location}" if a.location else "") + ".",
                 f"Purchase date: {a.purchase_date or 'not recorded'}.",
                 f"Warranty: {self.coverage.reason}"]
        if self.similar_repairs:
            for s in self.similar_repairs:
                r = s.repair
                parts = f"; parts replaced: {', '.join(r.parts_replaced)}" if r.parts_replaced else ""
                cost = f"cost {format_money(r.cost)}" if r.cost > 0 else "cost not recorded"
                lines.append(f"Similar past repair: {r.reported_date} ({s.days_ago} days ago): "
                             f"{r.problem_description}; resolution: {r.resolution or 'not recorded'}{parts}; {cost}.")
        else:
            lines.append("No similar past repair found in the records.")
        lines.append(f"Recorded repairs for this appliance: {self.repair_count} "
                     f"(total {format_money(self.total_repair_cost)}).")
        if self.last_service:
            lines.append(f"Last recorded service: {self.last_service.done_date} ({self.last_service.description}).")
        for v in self.overdue_tasks:
            lines.append(f"Overdue maintenance: {v.task.title} (due {v.task.next_due_date}, {-v.days_until} days overdue).")
        return "\n".join(lines)


class ApplianceContextService:
    def __init__(self, appliances: ApplianceRepository, repairs: RepairRepository, logs: MaintenanceLogRepository,
                 warranties: WarrantyService, maintenance: MaintenanceService,
                 clock: Callable[[], date] = date.today, min_similarity: float = 0.25, max_similar: int = 3) -> None:
        self._appliances, self._repairs, self._logs = appliances, repairs, logs
        self._warranties, self._maintenance, self._clock = warranties, maintenance, clock
        self._min_similarity, self._max_similar = min_similarity, max_similar

    def build(self, user_id: str, appliance_id: str, problem: str | None = None) -> ApplianceContext:
        validate_identifier(user_id, "user_id")
        appliance = self._appliances.get(user_id, appliance_id)       # NotFoundError for foreign ids
        today = self._clock()
        repairs = list(self._repairs.list_for_user(user_id, appliance_id=appliance_id))
        similar: list[SimilarRepair] = []
        if problem and problem.strip():
            for r in repairs:
                score = similarity(problem, f"{r.problem_description} {' '.join(r.parts_replaced)}")
                if score >= self._min_similarity:
                    similar.append(SimilarRepair(r, round(score, 3), (today - r.reported_date).days))
            similar.sort(key=lambda s: (-s.similarity, s.days_ago))
        logs = self._logs.list_for_user(user_id, appliance_id=appliance_id)
        return ApplianceContext(
            appliance=appliance, today=today, coverage=self._warranties.coverage(user_id, appliance_id, today),
            similar_repairs=tuple(similar[: self._max_similar]), repair_count=len(repairs),
            total_repair_cost=sum((r.cost for r in repairs), Decimal("0.00")),
            overdue_tasks=tuple(self._maintenance.overview(user_id, appliance_id).overdue),
            last_service=logs[0] if logs else None)
