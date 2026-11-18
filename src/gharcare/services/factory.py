"""Builds every deterministic service from an AppContainer, sharing one clock (so 'today' is
consistent across a request and fixed in tests)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

from gharcare.container import AppContainer
from gharcare.services.context import ApplianceContextService
from gharcare.services.expenses import ExpenseService
from gharcare.services.maintenance import MaintenanceService
from gharcare.services.reports import ReportService
from gharcare.services.warranty import WarrantyService


@dataclass
class Services:
    warranties: WarrantyService
    maintenance: MaintenanceService
    expenses: ExpenseService
    context: ApplianceContextService
    reports: ReportService


def build_services(app: AppContainer, clock: Callable[[], date] = date.today) -> Services:
    warranties = WarrantyService(app.appliances, app.warranties, clock)
    maintenance = MaintenanceService(app.appliances, app.maintenance, app.maintenance_logs, app.db, clock)
    expenses = ExpenseService(app.appliances, app.repairs, app.maintenance_logs, clock)
    return Services(
        warranties=warranties, maintenance=maintenance, expenses=expenses,
        context=ApplianceContextService(app.appliances, app.repairs, app.maintenance_logs, warranties, maintenance, clock),
        reports=ReportService(app.appliances, app.repairs, app.documents, app.artifacts, warranties, maintenance, expenses, clock))
