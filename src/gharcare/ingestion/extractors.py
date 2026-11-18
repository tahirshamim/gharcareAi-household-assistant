"""Turn uploaded bytes into per-page text. No OCR in v1: image-only files yield no text and are
reported to the user instead of failing silently."""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass

from gharcare.errors import IngestionError


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based, matches what the user sees in the PDF viewer
    text: str


class TextExtractor(ABC):
    name = "base"

    @abstractmethod
    def extract(self, content: bytes) -> list[PageText]:
        """Return one PageText per page (text may be empty for pages without a text layer)."""


class PlainTextExtractor(TextExtractor):
    """.txt/.md files. A form-feed character (\\f) marks a page break, as in text exported from PDFs."""
    name = "plain-text"

    def extract(self, content: bytes) -> list[PageText]:
        for encoding in ("utf-8-sig", "utf-16"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeError:
                continue
        else:
            raise IngestionError("text file is not valid UTF-8 or UTF-16")
        return [PageText(i, page) for i, page in enumerate(text.split("\f"), start=1)]


class PdfTextExtractor(TextExtractor):
    """Reads the embedded text layer with pypdf (optional dependency, imported lazily)."""
    name = "pypdf"

    def extract(self, content: bytes) -> list[PageText]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise IngestionError("PDF support needs pypdf: pip install pypdf") from exc
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted and not reader.decrypt(""):
                raise IngestionError("PDF is password-protected")
            return [PageText(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
        except IngestionError:
            raise
        except Exception as exc:  # third-party parser boundary: any parse failure becomes one clean error
            raise IngestionError(f"could not read PDF: {exc}") from exc


def default_extractors() -> dict[str, TextExtractor]:
    plain, pdf = PlainTextExtractor(), PdfTextExtractor()
    return {".txt": plain, ".md": plain, ".pdf": pdf}
