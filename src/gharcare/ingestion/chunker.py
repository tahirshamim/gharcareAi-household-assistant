"""Structure-aware chunking for appliance manuals.

Rules (each one exists to make citations and retrieval trustworthy):
* A chunk never crosses a page boundary  -> `page_number` is exact.
* A chunk never crosses a heading        -> each chunk has one `section`.
* Paragraphs/lists/table rows are kept together; text is only split when a single block is
  larger than `max_chars`, and then on line, then word boundaries (never mid-word unless a
  single 'word' exceeds the limit).
* The section heading is prepended to the chunk text so the embedding and keyword search both
  see the context ("DEFROSTING ... ice builds up").
* Noise is dropped: bare page numbers, running headers/footers repeated on half the pages, and stray fragments of fewer
  than 6 letters (diagram labels). Everything else is kept.

Heading detection is a heuristic (ALL-CAPS line without digits, standalone numbered line, or a
standalone line ending with ':'). It can misjudge unusual layouts; that is a documented limit.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterator, Sequence

from gharcare.domain import Chunk
from gharcare.errors import DomainValidationError
from gharcare.ingestion.extractors import PageText

_PAGE_NUMBER_ONLY = re.compile(r"^\W*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\W*$", re.IGNORECASE)
_NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")
_TERMINAL = (".", "!", "?", "\u06d4", "\u061f")  # latin and Urdu full stop / question mark
_MIN_LETTERS = 6
_WHITESPACE_RUN = re.compile(r"[ \t]+")


def _caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return (len(letters) >= 3 and all(c.isupper() for c in letters) and len(line) <= 60
            and not any(c.isdigit() for c in line) and not line.endswith(_TERMINAL))


def _standalone_heading(line: str) -> bool:
    if len(line) > 80 or line.endswith(_TERMINAL):
        return False
    return _caps_heading(line) or bool(_NUMBERED.match(line)) or (line.endswith(":") and len(line) <= 60)


def _segments(page_text: str) -> Iterator[tuple[str, str]]:
    """Yield ('heading'|'text', content). Blocks are separated by blank lines."""
    for block in re.split(r"\n\s*\n", page_text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [_WHITESPACE_RUN.sub(" ", ln).strip() for ln in block.split("\n")]
        lines = [ln for ln in lines if ln and not _PAGE_NUMBER_ONLY.match(ln)]
        if not lines:
            continue
        if len(lines) == 1 and _standalone_heading(lines[0]):
            yield "heading", lines[0].rstrip(":").strip()
            continue
        current: list[str] = []
        heading: list[str] = []
        for line in lines:
            if _caps_heading(line):          # all-caps heading inside a hard-wrapped block
                if current:
                    yield "text", "\n".join(current)
                    current = []
                heading.append(line)         # consecutive caps lines are ONE wrapped heading
            else:
                if heading:
                    yield "heading", " ".join(heading)
                    heading = []
                current.append(line)
        if heading:
            yield "heading", " ".join(heading)
        if current:
            yield "text", "\n".join(current)


class Chunker:
    def __init__(self, max_chars: int = 1000) -> None:
        if max_chars < 100:
            raise DomainValidationError("max_chars must be at least 100")
        self.max_chars = max_chars

    def chunk_pages(self, user_id: str, document_id: str, pages: Sequence[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        section: str | None = None
        pages = self._strip_repeated_lines(pages)

        def flush(parts: list[str], page_number: int) -> None:
            body = "\n\n".join(parts).strip()
            if sum(ch.isalpha() for ch in body) >= _MIN_LETTERS:      # drops stray diagram labels such as 'EN'

                text = f"{section}\n{body}" if section else body
                chunks.append(Chunk(user_id=user_id, document_id=document_id, chunk_index=len(chunks),
                                    text=text, page_number=page_number, section=section))

        for page in pages:
            parts: list[str] = []
            size = 0
            for kind, content in _segments(page.text):
                if kind == "heading":
                    flush(parts, page.page_number)
                    parts, size = [], 0
                    section = content
                    continue
                for piece in self._fit(content):
                    if parts and size + len(piece) + 2 > self.max_chars:
                        flush(parts, page.page_number)
                        parts, size = [], 0
                    parts.append(piece)
                    size += len(piece) + 2
            flush(parts, page.page_number)
        return chunks

    @staticmethod
    def _strip_repeated_lines(pages: Sequence[PageText]) -> list[PageText]:
        """Removes running headers/footers: short lines that appear (ignoring digits, so page numbers match) on at least
        half of the pages of a document with 5+ pages. A real manual repeats its document code on every page; left in,
        it pollutes every chunk and every keyword search."""
        if len(pages) < 5:
            return list(pages)
        key = lambda line: re.sub(r"\d+", "#", line.strip())
        counts: Counter[str] = Counter()
        for page in pages:
            counts.update({key(l) for l in page.text.splitlines() if l.strip()})
        threshold = max(3, math.ceil(len(pages) * 0.5))
        noisy = {k for k, n in counts.items() if n >= threshold and len(k) <= 40}
        if not noisy:
            return list(pages)
        return [PageText(p.page_number, "\n".join(l for l in p.text.splitlines() if key(l) not in noisy)) for p in pages]

    # -- splitting oversized blocks -----------------------------------------------------
    def _fit(self, block: str) -> list[str]:
        if len(block) <= self.max_chars:
            return [block]
        pieces: list[str] = []
        current = ""
        for unit in self._units(block):
            if current and len(current) + 1 + len(unit) > self.max_chars:
                pieces.append(current)
                current = unit
            else:
                current = f"{current}\n{unit}" if current else unit
        if current:
            pieces.append(current)
        return pieces

    def _units(self, block: str) -> Iterator[str]:
        for line in block.split("\n"):
            if len(line) <= self.max_chars:
                yield line
                continue
            current = ""
            for word in line.split(" "):
                while len(word) > self.max_chars:      # pathological token (e.g. a very long URL)
                    if current:
                        yield current
                        current = ""
                    yield word[: self.max_chars]
                    word = word[self.max_chars:]
                if current and len(current) + 1 + len(word) > self.max_chars:
                    yield current
                    current = word
                else:
                    current = f"{current} {word}" if current else word
            if current:
                yield current
