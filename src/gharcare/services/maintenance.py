"""Maintenance scheduling: what is due, completing tasks, and recording service history."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Callable

from gharcare.domain import Appliance, MaintenanceLog, MaintenanceTask, TaskStatus
from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError
from gharcare.repositories.base import (
    ApplianceRepository, MaintenanceLogRepository, MaintenanceRepository, TransactionManager,
)


@dataclass(frozen=True)
class TaskView:
    task: MaintenanceTask
    appliance: Appliance
    status: TaskStatus
    days_until: int | None       # negative = days overdue; None for a completed one-off task


@dataclass(frozen=True)
class MaintenanceOverview:
    overdue: list[TaskView] = field(default_factory=list)       # most overdue first
    due_soon: list[TaskView] = field(default_factory=list)      # soonest first
    upcoming: list[TaskView] = field(default_factory=list)
    completed: list[TaskView] = field(default_factory=list)     # one-off tasks already done


class MaintenanceService:
    def __init__(self, appliances: ApplianceRepository, tasks: MaintenanceRepository,
                 logs: MaintenanceLogRepository, tx: TransactionManager,
                 clock: Callable[[], date] = date.today, soon_days: int = 14) -> None:
        self._appliances, self._tasks, self._logs, self._tx = appliances, tasks, logs, tx
        self._clock, self._soon_days = clock, soon_days

    def overview(self, user_id: str, appliance_id: str | None = None) -> MaintenanceOverview:
        validate_identifier(user_id, "user_id")
        today = self._clock()
        appliances = {a.id: a for a in self._appliances.list_for_user(user_id)}
        groups: dict[TaskStatus, list[TaskView]] = {s: [] for s in TaskStatus}
        for task in self._tasks.list_for_user(user_id, appliance_id):
            if task.appliance_id not in appliances:
                continue
            days = (task.next_due_date - today).days if task.next_due_date else None
            status = task.status(today, self._soon_days)
            groups[status].append(TaskView(task, appliances[task.appliance_id], status, days))
        for status in (TaskStatus.OVERDUE, TaskStatus.DUE_SOON, TaskStatus.UPCOMING):
            groups[status].sort(key=lambda v: (v.days_until, v.task.title))
        groups[TaskStatus.COMPLETED].sort(key=lambda v: v.task.last_done_date or date.min, reverse=True)
        return MaintenanceOverview(groups[TaskStatus.OVERDUE], groups[TaskStatus.DUE_SOON],
                                   groups[TaskStatus.UPCOMING], groups[TaskStatus.COMPLETED])

    def complete_task(self, user_id: str, task_id: str, *, done_on: date | None = None,
                      cost: Decimal | str | int | None = None, technician: str | None = None,
                      notes: str | None = None) -> tuple[MaintenanceTask, MaintenanceLog]:
        """Records that the work was done AND reschedules the task, atomically: either both happen
        or neither does (no log without a rescheduled task, no rescheduled task without a log)."""
        validate_identifier(user_id, "user_id")
        today = self._clock()
        done_on = done_on or today
        if done_on > today:
            raise DomainValidationError("completion date cannot be in the future")
        with self._tx.transaction():
            task = self._tasks.get(user_id, task_id)
            if task.next_due_date is None:
                raise DomainValidationError("this one-off task is already completed")
            log = self._logs.add(MaintenanceLog(
                user_id=user_id, appliance_id=task.appliance_id, done_date=done_on, description=task.title,
                cost=cost, technician=technician, task_id=task.id, notes=notes))
            updated = self._tasks.update(task.mark_done(done_on))
        return updated, log

    def log_service(self, user_id: str, appliance_id: str, *, description: str, done_on: date | None = None,
                    cost: Decimal | str | int | None = None, technician: str | None = None,
                    notes: str | None = None) -> MaintenanceLog:
        """An ad-hoc service record that is not tied to a scheduled task."""
        validate_identifier(user_id, "user_id")
        today = self._clock()
        done_on = done_on or today
        if done_on > today:
            raise DomainValidationError("service date cannot be in the future")
        return self._logs.add(MaintenanceLog(user_id=user_id, appliance_id=appliance_id, done_date=done_on,
                                             description=description, cost=cost, technician=technician, notes=notes))

    def history(self, user_id: str, *, appliance_id: str | None = None, since: date | None = None,
                until: date | None = None) -> list[MaintenanceLog]:
        validate_identifier(user_id, "user_id")
        return list(self._logs.list_for_user(user_id, appliance_id=appliance_id, since=since, until=until))
