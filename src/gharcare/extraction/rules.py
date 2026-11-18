"""Rule-based field readers (the deterministic reader). They only return what a LABEL in the document supports.

Pakistan-specific handling: day-first dates (flagged when day/month order is ambiguous), Rs./PKR amounts with
commas, Eastern-Arabic/Persian digits converted to ASCII, a few Roman Urdu labels. These are heuristics: they can
miss unusual layouts, which is why every result carries a confidence and the user reviews it."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from gharcare.domain import DocumentType, FieldConfidence

HIGH, MEDIUM, LOW = FieldConfidence.HIGH, FieldConfidence.MEDIUM, FieldConfidence.LOW
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(text: str) -> str:
    return text.translate(_DIGITS)


@dataclass(frozen=True)
class RuleResult:
    value: str | None = None
    confidence: FieldConfidence = LOW
    evidence: str | None = None
    note: str | None = None
    alternatives: tuple[str, ...] = ()


# ---------------- dates ----------------
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_ISO = re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b")
_NUM = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})\b")
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-]+{_MON}\.?,?[\s\-]+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)


def _valid(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


@dataclass(frozen=True)
class DateHit:
    value: date
    alt: date | None       # the other reading when day/month order is ambiguous
    ambiguous: bool
    line: str
    before: str            # text before the date on the same line (used to find its label)


def find_dates(text: str) -> list[DateHit]:
    hits: list[DateHit] = []
    for line in text.splitlines():
        found: list[tuple[int, int, date, date | None, bool]] = []
        for m in _ISO.finditer(line):
            v = _valid(int(m[1]), int(m[2]), int(m[3]))
            if v:
                found.append((m.start(), m.end(), v, None, False))
        for m in _NUM.finditer(line):
            a, b, y = int(m[1]), int(m[2]), int(m[3])
            y = y + 2000 if y < 100 else y
            day_first, month_first = _valid(y, b, a), _valid(y, a, b)
            if day_first and month_first and a != b:
                found.append((m.start(), m.end(), day_first, month_first, True))       # read day-first, flag it
            elif day_first:
                found.append((m.start(), m.end(), day_first, None, False))
            elif month_first:
                found.append((m.start(), m.end(), month_first, None, True))            # only a month-first reading exists
        for m in _DMY.finditer(line):
            v = _valid(int(m[3]), _MONTHS[m[2].lower()[:3]], int(m[1]))
            if v:
                found.append((m.start(), m.end(), v, None, False))
        for m in _MDY.finditer(line):
            v = _valid(int(m[3]), _MONTHS[m[1].lower()[:3]], int(m[2]))
            if v:
                found.append((m.start(), m.end(), v, None, False))
        end_of_last = -1
        for start, end, v, alt, amb in sorted(found, key=lambda f: f[0]):
            if start >= end_of_last:                                                   # drop overlapping matches
                hits.append(DateHit(v, alt, amb, line.strip(), line[:start]))
                end_of_last = end
    return hits


def all_date_strings(text: str) -> set[str]:
    """Every date the text could mean (both readings of ambiguous ones), as ISO strings."""
    out: set[str] = set()
    for h in find_dates(text):
        out.add(h.value.isoformat())
        if h.alt:
            out.add(h.alt.isoformat())
    return out


def pick_date(text: str, label: re.Pattern, exclude: re.Pattern | None = None) -> RuleResult:
    hits = [h for h in find_dates(text) if not (exclude and exclude.search(h.before[-45:]))]
    labeled = [h for h in hits if label.search(h.before[-45:])]
    pool, is_labeled = (labeled, True) if labeled else (hits, False)
    if not pool:
        return RuleResult(note="no date found in the document")
    distinct = list(dict.fromkeys(h.value for h in pool))
    first = pool[0]
    if len(distinct) > 1:
        return RuleResult(distinct[0].isoformat(), LOW, first.line, "several different dates were found; please choose",
                          tuple(d.isoformat() for d in distinct[1:]))
    if any(h.ambiguous for h in pool):
        return RuleResult(first.value.isoformat(), MEDIUM, first.line,
                          "day/month order is ambiguous; read as day-first (the usual order in Pakistan)",
                          (first.alt.isoformat(),) if first.alt else ())
    if not is_labeled:
        return RuleResult(first.value.isoformat(), MEDIUM, first.line, "the date has no label such as 'Date:'")
    return RuleResult(first.value.isoformat(), HIGH, first.line)


# ---------------- money ----------------
_AMT = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EXCLUDE = re.compile(r"sub\s*-?\s*total|discount|gst|\btax\b|vat|advance|balance|change|received|tendered|installment|per\s*month", re.I)
_PRIORITY = [(3, re.compile(r"grand\s*total|net\s*(amount|total|payable)|amount\s*(paid|payable)|total\s*payable|kul\s*raqam", re.I)),
             (2, re.compile(r"\btotal\b|invoice\s*(amount|total)", re.I)),
             (1, re.compile(r"\b(price|amount|charges?|bill)\b", re.I))]


def number_values(text: str) -> set[Decimal]:
    out: set[Decimal] = set()
    for token in _AMT.findall(text):
        try:
            out.add(Decimal(token.replace(",", "")))
        except Exception:
            continue
    return out


def pick_amount(text: str) -> RuleResult:
    candidates: list[tuple[int, Decimal, str]] = []
    for line in text.splitlines():
        priority = next((p for p, rx in _PRIORITY if rx.search(line)), 0)
        if not priority or (priority < 3 and _EXCLUDE.search(line)):
            continue
        values = []
        for token in _AMT.findall(line):
            try:
                v = Decimal(token.replace(",", ""))
            except Exception:
                continue
            if 0 < v < 10**9 and len(token.replace(",", "").split(".")[0]) < 10:      # skips phone-number-like tokens
                values.append(v)
        if values:
            candidates.append((priority, values[-1], line.strip()))
    if not candidates:
        return RuleResult(note="no labeled amount found")
    top = max(p for p, _, _ in candidates)
    best = [(v, line) for p, v, line in candidates if p == top]
    distinct = list(dict.fromkeys(v for v, _ in best))
    fmt = lambda v: f"{v.quantize(Decimal('0.01'))}"
    if len(distinct) > 1:
        return RuleResult(fmt(distinct[-1]), LOW, best[-1][1], "several different totals were found; please choose",
                          tuple(fmt(v) for v in distinct[:-1]))
    if top == 1:
        return RuleResult(fmt(distinct[0]), MEDIUM, best[0][1], "the amount has no 'Total' label")
    return RuleResult(fmt(distinct[0]), HIGH, best[0][1])


# ---------------- durations ----------------
_DUR = re.compile(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|saal|سال|months?|mahine|mahinay|ماہ)", re.I)
_WARRANTY_WORD = re.compile(r"warrant|guarant|ضمانت", re.I)


def _months(match: re.Match) -> int | None:
    n = Decimal(match[1])
    months = n * 12 if match[2].lower().startswith(("y", "s", "س")) else n
    return int(months) if months == months.to_integral_value() and 1 <= months <= 120 else None


def all_durations(text: str) -> set[int]:
    return {m for m in (_months(x) for x in _DUR.finditer(text)) if m}


def pick_duration(text: str) -> RuleResult:
    found: list[tuple[int, str]] = []
    for line in text.splitlines():
        if _WARRANTY_WORD.search(line):
            found += [(m, line.strip()) for m in (_months(x) for x in _DUR.finditer(line)) if m]
    if not found:
        return RuleResult(note="no warranty duration found")
    distinct = list(dict.fromkeys(m for m, _ in found))
    if len(distinct) > 1:
        return RuleResult(str(distinct[0]), LOW, found[0][1],
                          "several warranty durations were found (e.g. product vs. a part); please choose",
                          tuple(str(m) for m in distinct[1:]))
    return RuleResult(str(distinct[0]), HIGH, found[0][1])


# ---------------- labeled text ----------------
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-/\.]{1,39}$")


def pick_text(text: str, label_pattern: str, token: bool = False) -> RuleResult:
    rx = re.compile(rf"^\W*(?:{label_pattern})\s*[:\-–]\s*(\S.*?)\s*$", re.I)
    found: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = rx.match(line)
        if not m:
            continue
        value = m.group(1).strip(" .,;")
        if token:
            parts = value.split()
            value = parts[0].strip(" .,;()") if parts else ""
            if not _TOKEN.match(value):
                continue
        if value and len(value) <= 200:
            found.append((value, line.strip()))
    if not found:
        return RuleResult(note="not found in the document")
    distinct = list({v.casefold(): v for v, _ in found}.values())
    if len(distinct) > 1:
        return RuleResult(distinct[0], LOW, found[0][1], "several different values were found; please choose", tuple(distinct[1:]))
    return RuleResult(distinct[0], HIGH, found[0][1])


BRANDS = ("Haier", "Dawlance", "PEL", "Orient", "Gree", "Samsung", "LG", "Sony", "TCL", "Panasonic", "Kenwood", "Changhong",
          "Ruba", "Waves", "Whirlpool", "Hitachi", "Daikin", "Mitsubishi", "Toshiba", "Philips", "Anex", "Nobel", "Homage", "Super Asia")


def pick_brand(text: str) -> RuleResult:
    found = [b for b in BRANDS if re.search(rf"\b{re.escape(b)}\b", text, re.I)]
    if not found:
        return RuleResult(note="no known brand name found")
    evidence = next((ln.strip() for ln in text.splitlines() if re.search(rf"\b{re.escape(found[0])}\b", ln, re.I)), None)
    if len(found) > 1:
        return RuleResult(found[0], LOW, evidence, "several brand names appear; please choose", tuple(found[1:]))
    return RuleResult(found[0], HIGH, evidence)


# ---------------- per-document-type readers ----------------
_PURCHASE_LABEL = re.compile(r"\bdate(d)?\b|sold\s*on|purchase|invoice|bill", re.I)
_START_LABEL = re.compile(r"start|from|commenc|begin|purchase|installation|activation|\bdate\b", re.I)
_EXPIRY_LABEL = re.compile(r"expir|valid\s*(till|until|upto|up\s*to|to)|end\s*date|\bends?\b", re.I)
_SERVICE_LABEL = re.compile(r"\bdate(d)?\b|visit|serviced\s*on|completed", re.I)
_MODEL = r"model(?:\s*(?:no\.?|number|#|name))?"
_SERIAL = r"serial(?:\s*(?:no\.?|number|#))?|s/n|sr\.?\s*no\.?"


def read_fields(doc_type: DocumentType, text: str) -> dict[str, RuleResult]:
    t = normalize_digits(text)
    if doc_type == DocumentType.RECEIPT:
        return {"store_name": pick_text(t, r"shop|store|seller|dealer|sold\s*by|merchant|outlet"),
                "purchase_date": pick_date(t, _PURCHASE_LABEL), "total_amount": pick_amount(t),
                "product_name": pick_text(t, r"item|product|description|appliance"), "brand": pick_brand(t),
                "model_number": pick_text(t, _MODEL, token=True), "serial_number": pick_text(t, _SERIAL, token=True)}
    if doc_type == DocumentType.WARRANTY_CARD:
        return {"product_name": pick_text(t, r"item|product|description|appliance"), "brand": pick_brand(t),
                "model_number": pick_text(t, _MODEL, token=True), "serial_number": pick_text(t, _SERIAL, token=True),
                "warranty_start_date": pick_date(t, _START_LABEL, exclude=_EXPIRY_LABEL),
                "warranty_duration_months": pick_duration(t),
                "provider": pick_text(t, r"issued\s*by|warranty\s*provider|service\s*provider|authori[sz]ed\s*(?:dealer|service\s*(?:centre|center))|provider")}
    return {"service_date": pick_date(t, _SERVICE_LABEL), "total_amount": pick_amount(t),
            "problem_description": pick_text(t, r"problem|complaint|issue|fault"),
            "work_done": pick_text(t, r"work\s*done|action\s*taken|resolution|remarks"),
            "technician": pick_text(t, r"technician|engineer|serviced\s*by|attended\s*by"),
            "parts_replaced": pick_text(t, r"parts?\s*(?:replaced|used|changed)|spare\s*parts")}
