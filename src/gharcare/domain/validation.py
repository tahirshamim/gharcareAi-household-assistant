"""Small, reusable validators. Domain models and the storage layer share them so that
rules such as 'what is a safe user_id' live in exactly one place."""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from gharcare.errors import DomainValidationError

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CENTS = Decimal("0.01")


def validate_identifier(value: object, field_name: str) -> str:
    """IDs become folder names and SQL keys, so only a conservative character set is allowed.
    This also blocks path traversal ('../') at the source."""
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise DomainValidationError(
            f"{field_name} must be 1-64 characters: letters, digits, '_' or '-'"
        )
    return value


def clean_text(value: object, field_name: str, *, required: bool = False, max_length: int = 500) -> str | None:
    """Strip whitespace; treat blank as None; enforce a length cap."""
    if value is None:
        if required:
            raise DomainValidationError(f"{field_name} is required")
        return None
    if not isinstance(value, str):
        raise DomainValidationError(f"{field_name} must be text")
    text = value.strip()
    if not text:
        if required:
            raise DomainValidationError(f"{field_name} is required")
        return None
    if len(text) > max_length:
        raise DomainValidationError(f"{field_name} is too long (max {max_length} characters)")
    return text


def to_money(value: object, field_name: str) -> Decimal | None:
    """Money is Decimal, never float. Floats are routed through str() to avoid binary artifacts."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise DomainValidationError(f"{field_name} must be a number")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise DomainValidationError(f"{field_name} must be a number") from exc
    if not amount.is_finite():
        raise DomainValidationError(f"{field_name} must be a finite number")
    if amount < 0:
        raise DomainValidationError(f"{field_name} cannot be negative")
    return amount.quantize(_CENTS, rounding=ROUND_HALF_UP)
