"""Household report as Markdown, generated from saved records only (no AI-written content), then
stored as an artifact. Every figure comes from the same services the assistant uses, and a final
section lists data gaps so incomplete records are visible instead of hidden."""
from __future__ import annotations

from datetime import date
from typing import Callable, Sequence

from gharcare.domain import add_months
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError
from gharcare.repositories.base import ApplianceRepository, DocumentRepository, RepairRepository
from gharcare.services.expenses import ExpenseService
from gharcare.services.formatting import format_money
from gharcare.services.maintenance import MaintenanceService
from gharcare.services.warranty import WarrantyService
from gharcare.storage.base import ArtifactRef, ArtifactStore


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip() or "-"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


class ReportService:
    def __init__(self, appliances: ApplianceRepository, repairs: RepairRepository, documents: DocumentRepository,
                 artifacts: ArtifactStore, warranties: WarrantyService, maintenance: MaintenanceService,
                 expenses: ExpenseService, clock: Callable[[], date] = date.today) -> None:
        self._appliances, self._repairs, self._documents, self._artifacts = appliances, repairs, documents, artifacts
        self._warranties, self._maintenance, self._expenses, self._clock = warranties, maintenance, expenses, clock

    def build_markdown(self, user_id: str, months: int = 6) -> str:
        validate_identifier(user_id, "user_id")
        if months < 1:
            raise DomainValidationError("months must be at least 1")
        today = self._clock()
        since = add_months(today, -months)
        appliances = sorted(self._appliances.list_for_user(user_id), key=lambda a: a.display_name)
        names = {a.id: a.display_name for a in appliances}
        wviews = self._warranties.list_all(user_id)
        overview = self._maintenance.overview(user_id)
        spend = self._expenses.summary(user_id, since=since, until=today)
        repairs = self._repairs.list_for_user(user_id, since=since, until=today)
        docs = self._documents.list_for_user(user_id)
        out = [f"# Household report", "",
               f"Generated {today}. Expense and repair figures cover {since} to {today} ({months} months). "
               f"Currency: {spend.currency}. Built only from your saved records.", ""]

        out += [f"## Appliances ({len(appliances)})", ""]
        out += [_table(["Appliance", "Model", "Location", "Purchased", "Price"],
                       [[a.display_name, a.model, a.location, a.purchase_date,
                         format_money(a.purchase_price) if a.purchase_price is not None else None]
                        for a in appliances]) if appliances else "No appliances recorded.", ""]

        out += ["## Warranty status", ""]
        out += [_table(["Appliance", "Expires", "Days left", "Status", "Confirmed by you"],
                       [[v.appliance.display_name, v.expiry_date, v.days_remaining, v.status.value,
                         "yes" if v.warranty.verified else "no"] for v in wviews]) if wviews
                else "No warranties recorded.", ""]

        out += ["## Maintenance", ""]
        shown = False
        for title, group in (("Overdue", overview.overdue), ("Due soon", overview.due_soon), ("Upcoming", overview.upcoming)):
            if group:
                shown = True
                out += [f"**{title}**", "", _table(["Appliance", "Task", "Due", "Days"],
                        [[v.appliance.display_name, v.task.title, v.task.next_due_date, v.days_until] for v in group]), ""]
        if not shown:
            out += ["No scheduled maintenance.", ""]

        out += [f"## Repairs in period ({len(repairs)})", ""]
        out += [_table(["Date", "Appliance", "Problem", "Resolution", "Cost", "Under warranty"],
                       [[r.reported_date, names.get(r.appliance_id), r.problem_description, r.resolution,
                         format_money(r.cost) if r.cost > 0 else None,
                         {True: "yes", False: "no", None: "unknown"}[r.covered_by_warranty]] for r in repairs])
                if repairs else "No repairs recorded in this period.", ""]

        out += ["## Expenses in period", "", f"**Total: {format_money(spend.total)}**", ""]
        if spend.lines:
            out += [_table(["Category", "Amount"], [[c.value, format_money(t)] for c, t in spend.by_category.items()]), "",
                    _table(["Appliance", "Amount"], [[t.appliance_name, format_money(t.total)] for t in spend.by_appliance]), "",
                    _table(["Month", "Amount"], [[m, format_money(t)] for m, t in spend.by_month.items()]), ""]

        out += ["## Source documents", ""]
        if docs:
            for d in docs:
                owner = names.get(d.appliance_id, "not linked to an appliance")
                out.append(f"- {_cell(d.original_filename)} ({d.doc_type.value}, {owner}, sha256 {d.sha256[:8]})")
        else:
            out.append("No documents uploaded.")
        out.append("")

        gaps: list[str] = [f"The warranty for {v.appliance.display_name} has not been confirmed by you yet."
                           for v in wviews if v.needs_review]
        if spend.gaps.records_without_cost:
            gaps.append(f"{spend.gaps.records_without_cost} repair/service record(s) in this period have no recorded cost, "
                        "so the totals above may be understated.")
        gaps += [f"{a.display_name} has no purchase date or price recorded." for a in appliances
                 if a.purchase_date is None or a.purchase_price is None]
        gaps += [f"Maintenance overdue: {v.appliance.display_name} - {v.task.title} (due {v.task.next_due_date})."
                 for v in overview.overdue]
        gaps += [f"Warranty ending soon: {v.appliance.display_name} on {v.expiry_date}."
                 for v in wviews if v.status.value == "expiring_soon"]
        linked = {d.appliance_id for d in docs}
        gaps += [f"No documents uploaded for {a.display_name}." for a in appliances if a.id not in linked]
        out += ["## Data gaps and items to review", ""] + ([f"- {g}" for g in gaps] if gaps else ["No gaps found."])
        return "\n".join(out) + "\n"

    def generate(self, user_id: str, months: int = 6) -> ArtifactRef:
        """Builds the report and stores it (never overwriting earlier reports)."""
        return self._artifacts.save_report(user_id, "household-report", self.build_markdown(user_id, months).encode("utf-8"), ".md")
