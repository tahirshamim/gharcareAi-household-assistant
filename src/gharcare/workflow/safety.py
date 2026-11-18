"""Deterministic hazard screen. Runs FIRST, before any model call, so it works when the LLM is down
and cannot be talked out of escalating. It is intentionally conservative: a false alarm costs a
sentence; a miss could cost safety. It is still a keyword screen and WILL miss unusual phrasings,
so the answer-generation prompt also tells the model to escalate dangerous steps."""
from __future__ import annotations

import re

_PATTERNS = [
    # English
    r"\bsmell(s|ing)?\s+(of\s+)?(gas|burn\w*|smoke|plastic|fumes?)\b", r"\bgas\s+(leak\w*|smell\w*|odou?r)\b",
    r"\bburn(t|ing)\s+(smell|odou?r|plastic|wire|wires)\b", r"\bsmok(e|ing)\b", r"\bsparks?\b", r"\bsparking\b",
    r"\belectric(al)?\s+shock\b", r"\bgot\s+(a\s+)?shock(ed)?\b", r"\b(caught|on)\s+fire\b", r"\bfire\b",
    r"\bexposed\s+wires?\b", r"\bwater\s+(near|on|in|inside)\s+(the\s+)?(socket|wiring|plug|outlet|switch)\b",
    r"\bexplo(de|ded|sion|ding)\b",
    # Roman Urdu
    r"\bjalne\s+(ki|ka)\s+(smell|boo|bu)\b", r"\bjal\s+(raha|rahi|gaya|gayi)\b", r"\bchingari\b", r"\baag\b",
    r"\bdhuan\b", r"\bgas\s+(ki|ka)\s+(smell|boo|bu)\b", r"\bgas\s+leak\b", r"\bkarant\b", r"\bcurrent\s+(lag|laga|lagna)\b",
    r"\bbijli\s+ka\s+jhatka\b",
    # Urdu script
    "چنگاری", "آگ", "دھواں", "جلنے کی بو", "جلنے کی", "گیس کی بو", "گیس لیک", "کرنٹ",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _PATTERNS]

SAFETY_MESSAGE = (
    "This could be dangerous, so please do not try to repair it yourself.\n"
    "1. If it is safe to reach, switch the appliance off at the main switch or breaker. Do not touch it with wet hands.\n"
    "2. If you smell gas, do not use any switches or flames, open the windows and leave the area.\n"
    "3. If there is fire or smoke, or anyone has had an electric shock, call your local emergency services right away.\n"
    "4. Then contact a qualified technician before using the appliance again.\n"
    "(Roman Urdu: Mehrbani karke khud marammat ki koshish na karein. Main switch band karein, aur kisi mahir technician ko bulayein.)"
)


def check_hazard(text: str) -> str | None:
    """Returns the matched pattern text (for logging/tests) or None."""
    for rx in _COMPILED:
        m = rx.search(text or "")
        if m:
            return m.group(0)
    return None
