"""Tokenizer shared by keyword search and the lexical embedder.

Works on English, Roman Urdu and Urdu script (Unicode word characters). Hyphenated codes such as
'HRF-306' or 'F-01' are kept whole AND split ('hrf-306', 'hrf', '306'), so both the exact code and
a sloppy 'hrf306'-style query can match. Error codes like 'E4' survive as single tokens.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.casefold()):
        token = match.group(0)
        tokens.append(token)
        if "-" in token:
            tokens.extend(token.split("-"))
    return tokens
