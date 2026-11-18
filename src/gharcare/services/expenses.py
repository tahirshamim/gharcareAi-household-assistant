"""Expense tracking: exact Decimal arithmetic over the user's saved records.

Three sources, all from the database: appliance purchases (price on purchase date), repair costs,
and maintenance-log costs. The LLM never computes any of this.

Honesty about gaps: a cost of 0 is ambiguous (free? covered by warranty? simply not entered?), so
zero-cost records are COUNTED and reported in `gaps` rather than silently treated as certain.
Appliances without a price/date are likewise reported, so a total can be described as
'at least X' when records are incomplete.

Periods are inclusive on both ends. 'Last N months' = from (today minus N calendar months) to today."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Callable, Sequence

from gharcare.domain import add_months
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError, NotFoundError
from gharcare.repositories.base import (
    ApplianceRepository, MaintenanceLogRepository, RepairRepository,
)
from gharcare.services.formatting import CURRENCY

ZERO = Decimal("0.00")


class ExpenseCategory(str, Enum):
    PURCHASE = "purchase"
    REPAIR = "repair"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True)
class ExpenseLine:
    date: date
    category: ExpenseCategory
    appliance_id: str
    appliance_name: str
    amount: Decimal
    description: str


@dataclass(frozen=True)
class ApplianceTotal:
    appliance_id: str
    appliance_name: str
    total: Decimal


@dataclass(frozen=True)
class DataGaps:
    records_without_cost: int = 0      # repairs / service logs recorded with cost 0
    purchases_without_price: int = 0   # appliances in scope with no purchase price
    purchases_without_date: int = 0    # appliances with a price but no purchase date (cannot be placed in a period)

    @property
    def any(self) -> bool:
        return bool(self.records_without_cost or self.purchases_without_price or self.purchases_without_date)


@dataclass(frozen=True)
class ExpenseSummary:
    since: date | None
    until: date | None
    total: Decimal
    by_category: dict[ExpenseCategory, Decimal]
    by_appliance: tuple[ApplianceTotal, ...]       # largest first
    by_month: dict[str, Decimal]                   # 'YYYY-MM' -> total, chronological
    lines: tuple[ExpenseLine, ...]                 # chronological, every amount traceable
    gaps: DataGaps
    currency: str = CURRENCY


@dataclass(frozen=True)
class PeriodComparison:
    current: ExpenseSummary
    previous: ExpenseSummary
    delta: Decimal                                 # current - previous
    pct_change: Decimal | None                     # None when the previous total is zero


class ExpenseService:
    def __init__(self, appliances: ApplianceRepository, repairs: RepairRepository,
                 logs: MaintenanceLogRepository, clock: Callable[[], date] = date.today) -> None:
        self._appliances, self._repairs, self._logs, self._clock = appliances, repairs, logs, clock

    def summary(self, user_id: str, *, since: date | None = None, until: date | None = None,
                appliance_id: str | None = None,
                categories: Sequence[ExpenseCategory] | None = None) -> ExpenseSummary:
        validate_identifier(user_id, "user_id")
        if since and until and since > until:
            raise DomainValidationError("'since' must not be after 'until'")
        wanted = set(ExpenseCategory(c) for c in categories) if categories else set(ExpenseCategory)
        appliances = {a.id: a for a in self._appliances.list_for_user(user_id)}
        if appliance_id and appliance_id not in appliances:
            raise NotFoundError(f"appliance {appliance_id} not found")
        in_scope = [appliances[appliance_id]] if appliance_id else list(appliances.values())

        def in_range(d: date) -> bool:
            return (since is None or d >= since) and (until is None or d <= until)

        lines: list[ExpenseLine] = []
        no_cost = no_price = no_date = 0

        if ExpenseCategory.PURCHASE in wanted:
            for a in in_scope:
                if a.purchase_date is None:
                    if a.purchase_price is None:
                        no_price += 1
                    else:
                        no_date += 1
                    continue
                if not in_range(a.purchase_date):
                    continue
                if a.purchase_price is None:
                    no_price += 1
                elif a.purchase_price > 0:
                    lines.append(ExpenseLine(a.purchase_date, ExpenseCategory.PURCHASE, a.id, a.display_name,
                                             a.purchase_price, f"Purchase of {a.display_name}"))
        if ExpenseCategory.REPAIR in wanted:
            for r in self._repairs.list_for_user(user_id, appliance_id=appliance_id, since=since, until=until):
                if r.cost > 0 and r.appliance_id in appliances:
                    lines.append(ExpenseLine(r.reported_date, ExpenseCategory.REPAIR, r.appliance_id,
                                             appliances[r.appliance_id].display_name, r.cost, r.problem_description))
                elif r.cost == 0:
                    no_cost += 1
        if ExpenseCategory.MAINTENANCE in wanted:
            for m in self._logs.list_for_user(user_id, appliance_id=appliance_id, since=since, until=until):
                if m.cost > 0 and m.appliance_id in appliances:
                    lines.append(ExpenseLine(m.done_date, ExpenseCategory.MAINTENANCE, m.appliance_id,
                                             appliances[m.appliance_id].display_name, m.cost, m.description))
                elif m.cost == 0:
                    no_cost += 1

        lines.sort(key=lambda l: (l.date, l.category.value, l.description))
        by_category = {c: sum((l.amount for l in lines if l.category == c), ZERO) for c in ExpenseCategory if c in wanted}
        per_appliance: dict[str, Decimal] = {}
        names: dict[str, str] = {}
        by_month: dict[str, Decimal] = {}
        for l in lines:
            per_appliance[l.appliance_id] = per_appliance.get(l.appliance_id, ZERO) + l.amount
            names[l.appliance_id] = l.appliance_name
            key = f"{l.date.year:04d}-{l.date.month:02d}"
            by_month[key] = by_month.get(key, ZERO) + l.amount
        by_appliance = tuple(sorted((ApplianceTotal(i, names[i], t) for i, t in per_appliance.items()),
                                    key=lambda x: (-x.total, x.appliance_name)))
        return ExpenseSummary(
            since=since, until=until, total=sum((l.amount for l in lines), ZERO), by_category=by_category,
            by_appliance=by_appliance, by_month=dict(sorted(by_month.items())), lines=tuple(lines),
            gaps=DataGaps(no_cost, no_price, no_date))

    def last_months(self, user_id: str, months: int, **filters) -> ExpenseSummary:
        if months < 1:
            raise DomainValidationError("months must be at least 1")
        today = self._clock()
        return self.summary(user_id, since=add_months(today, -months), until=today, **filters)

    def compare_recent(self, user_id: str, months: int, **filters) -> PeriodComparison:
        """The last N months versus the N months immediately before them."""
        current = self.last_months(user_id, months, **filters)
        previous_until = current.since - timedelta(days=1)
        previous = self.summary(user_id, since=add_months(current.since, -months), until=previous_until, **filters)
        delta = current.total - previous.total
        pct = (delta / previous.total * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) if previous.total else None
        return PeriodComparison(current, previous, delta, pct)
