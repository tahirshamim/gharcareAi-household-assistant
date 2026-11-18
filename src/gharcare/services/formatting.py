"""Shared presentation helpers. Money is always Decimal and always shown with its currency."""
from __future__ import annotations

from decimal import Decimal

CURRENCY = "PKR"


def format_money(amount: Decimal, currency: str = CURRENCY) -> str:
    return f"{currency} {amount:,.2f}"
