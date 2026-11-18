"""Warranty logic. Pure date arithmetic on stored records: nothing here is guessed or generated.

A warranty PERIOD being active is not the same as a specific FAULT being covered (that depends on
terms we may not have). Coverage answers therefore state the period facts and say plainly when
coverage terms are not recorded."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from typing import Callable

from gharcare.domain import Appliance, Warranty, WarrantyStatus
from gharcare.domain.validation import validate_identifier
from gharcare.repositories.base import ApplianceRepository, WarrantyRepository


class CoverageState(str, Enum):
    COVERED = "covered"          # a warranty period includes the date
    NOT_COVERED = "not_covered"  # warranty records exist but none includes the date
    UNKNOWN = "unknown"          # no warranty record: we cannot say either way


@dataclass(frozen=True)
class WarrantyView:
    warranty: Warranty
    appliance: Appliance
    status: WarrantyStatus
    days_remaining: int          # negative once expired

    @property
    def expiry_date(self) -> date:
        return self.warranty.expiry_date

    @property
    def needs_review(self) -> bool:
        """True while the user has not yet confirmed the values (e.g. AI-extracted ones)."""
        return not self.warranty.verified


@dataclass(frozen=True)
class Coverage:
    state: CoverageState
    on_date: date
    warranty: Warranty | None
    reason: str

    @property
    def verified(self) -> bool:
        return bool(self.warranty and self.warranty.verified)


class WarrantyService:
    def __init__(self, appliances: ApplianceRepository, warranties: WarrantyRepository,
                 clock: Callable[[], date] = date.today, soon_days: int = 30) -> None:
        self._appliances, self._warranties = appliances, warranties
        self._clock, self._soon_days = clock, soon_days

    def list_all(self, user_id: str, appliance_id: str | None = None) -> list[WarrantyView]:
        """Soonest-expiring first (already-expired ones come first, being the most urgent to know about)."""
        validate_identifier(user_id, "user_id")
        today = self._clock()
        appliances = {a.id: a for a in self._appliances.list_for_user(user_id)}
        views = [WarrantyView(w, appliances[w.appliance_id], w.status(today, self._soon_days), w.days_remaining(today))
                 for w in self._warranties.list_for_user(user_id, appliance_id) if w.appliance_id in appliances]
        return sorted(views, key=lambda v: (v.expiry_date, v.appliance.name))

    def expiring_soon(self, user_id: str, within_days: int | None = None) -> list[WarrantyView]:
        """Warranties that are still valid today but end within `within_days` (default: the soon window)."""
        limit = self._soon_days if within_days is None else within_days
        return [v for v in self.list_all(user_id) if 0 <= v.days_remaining <= limit]

    def coverage(self, user_id: str, appliance_id: str, on_date: date | None = None) -> Coverage:
        validate_identifier(user_id, "user_id")
        self._appliances.get(user_id, appliance_id)            # NotFoundError for foreign/unknown ids
        on_date = on_date or self._clock()
        warranties = self._warranties.list_for_user(user_id, appliance_id)
        if not warranties:
            return Coverage(CoverageState.UNKNOWN, on_date, None,
                            "No warranty record has been entered for this appliance.")
        active = [w for w in warranties if w.start_date <= on_date <= w.expiry_date]
        if active:
            best = max(active, key=lambda w: w.expiry_date)
            left = (best.expiry_date - on_date).days
            terms = (f" Recorded coverage notes: {best.coverage_notes}" if best.coverage_notes
                     else " No coverage terms are recorded, so it is unknown whether a specific problem is covered.")
            caveat = "" if best.verified else " This warranty record has not been confirmed by you yet."
            return Coverage(CoverageState.COVERED, on_date, best,
                            f"A warranty period is active until {best.expiry_date} ({left} days left).{terms}{caveat}")
        ended = [w for w in warranties if w.expiry_date < on_date]
        if ended:
            latest = max(ended, key=lambda w: w.expiry_date)
            return Coverage(CoverageState.NOT_COVERED, on_date, latest,
                            f"The most recent warranty period ended on {latest.expiry_date}.")
        upcoming = min(warranties, key=lambda w: w.start_date)
        return Coverage(CoverageState.NOT_COVERED, on_date, upcoming,
                        f"The recorded warranty only starts on {upcoming.start_date}.")

    def confirm(self, user_id: str, warranty_id: str, *, start_date: date | None = None,
                duration_months: int | None = None, provider: str | None = None,
                coverage_notes: str | None = None) -> Warranty:
        """The human-in-the-loop step: the user reviews (and optionally corrects) a warranty, which
        is then marked verified. Corrections are validated by the domain model; on any error the
        stored warranty is left unchanged."""
        validate_identifier(user_id, "user_id")
        current = self._warranties.get(user_id, warranty_id)
        changes = {k: v for k, v in dict(start_date=start_date, duration_months=duration_months,
                                         provider=provider, coverage_notes=coverage_notes).items() if v is not None}
        return self._warranties.update(replace(current, verified=True, **changes))
