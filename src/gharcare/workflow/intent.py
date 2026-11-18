"""Rule-based intent detection and parameter extraction (English + Roman Urdu). Rules cover the
clear cases deterministically; only genuinely unclear messages fall back to the LLM."""
from __future__ import annotations

import re
from enum import Enum

from gharcare.retrieval.text import tokenize
from gharcare.services.expenses import ExpenseCategory


class Intent(str, Enum):
    TROUBLESHOOTING = "troubleshooting"   # appliance problems and any question answered from manuals/documents
    WARRANTY = "warranty"
    MAINTENANCE = "maintenance"
    EXPENSES = "expenses"
    REPORT = "report"
    GENERAL = "general"


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


_GREETING = _rx(r"^\s*(hi|hello|hey|salam|assalam\w*|help|what can you do|thanks|thank you|shukriya)\b")
_REPORT = _rx(r"\b(report|summary|summari[sz]e)\b")
_EXPENSE = _rx(r"\b(spent|spend|spending|expenses?|kharcha|kharch|total cost|paisa|paise)\b")
# "how much" alone is NOT an expense question ("how much space / weight / voltage"); it needs money context.
_HOW_MUCH = _rx(r"\b(how much|kitn[aei])\b")
_MONEY = _rx(r"\b(cost|costs|costed|paid|pay|paying|rupees?|rs|pkr|bill|bills|did (i|we)|kitn[aei]\s+(ka|ki|ke)|lag[aei]+)\b")
_WARRANTY_WORD = _rx(r"\b(warrant(y|ies)|guarantees?)\b")
_WARRANTY_STATUS = _rx(r"\b(expir\w*|valid|left|still|under|covers?|covered|status|kab|khatam|active|which|soon|ends?)\b")
_MAINTENANCE = _rx(r"\b(maintenance|overdue|servicing|service\s+(is\s+)?due|due\s+(for\s+)?service|service\s+kab|upcoming\s+tasks?)\b")
_PROBLEM = _rx(r"\b(not|isn'?t|won'?t|doesn'?t|stopped|leak\w*|noise|noisy|error|code|ice|frost|barf|smell|broken|"
               r"kharab|nahi|band|garam|thanda|jam|how do i|how to|kaise|kya karun|what does|why|clean|reset|fix|"
               r"claim|manual)\b")


# A question that matched none of the record intents is a question for the user's documents: the grounded
# path either answers it with citations or says it is not in the documents. Sending it to the LLM classifier
# instead cost a call per question and sometimes routed manual questions ("Can I store spray cans?") to help.
_QUESTION = _rx(r"(\?\s*$|^\s*(what|how|why|when|where|which|who|can|could|should|shall|does|do|did|is|are|will|"
                r"would|may|kya|kaise|kab|kyun|kyon|kahan|kitn[aei])\b|\b(kya|kaise|kitn[aei]|kab|kyun|chahiye)\b)")


def classify_by_rules(text: str) -> Intent | None:
    if _GREETING.search(text):
        return Intent.GENERAL
    if _REPORT.search(text):
        return Intent.REPORT
    if _EXPENSE.search(text) or (_HOW_MUCH.search(text) and _MONEY.search(text)):
        return Intent.EXPENSES
    if _WARRANTY_WORD.search(text) and _WARRANTY_STATUS.search(text):
        return Intent.WARRANTY
    if _MAINTENANCE.search(text):
        return Intent.MAINTENANCE
    if _PROBLEM.search(text) or _QUESTION.search(text):
        return Intent.TROUBLESHOOTING
    return None


def extract_months(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(month|months|mahine|mahinon|maah)\b", text, re.I)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"(\d+)\s*(year|years|saal)\b", text, re.I)
    if m:
        return max(1, int(m.group(1))) * 12
    if re.search(r"\b(last|past)\s+year\b|pichle\s+saal", text, re.I):
        return 12
    if re.search(r"\b(last|past)\s+month\b|pichle\s+mahine", text, re.I):
        return 1
    return None


def extract_categories(text: str) -> list[ExpenseCategory] | None:
    found = []
    if re.search(r"\b(repairs?|marammat|technician|fix\w*)\b", text, re.I):
        found.append(ExpenseCategory.REPAIR)
    if re.search(r"\b(maintenance|servicing|service)\b", text, re.I):
        found.append(ExpenseCategory.MAINTENANCE)
    if re.search(r"\b(purchases?|bought|buy|kharida|price)\b", text, re.I):
        found.append(ExpenseCategory.PURCHASE)
    return found or None


_ALIASES = [{"fridge", "refrigerator", "freezer"}, {"ac", "aircon", "conditioner", "split"},
            {"washer", "washing", "laundry"}, {"microwave", "oven"}, {"tv", "television"}, {"heater", "geyser"}]
_FILLER = frozenset("my the a an is of to and in it hai ka ki ke mein meri mera".split())


def _expand(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for group in _ALIASES:
        if tokens & group:
            out |= group
    return out


def match_appliances(appliances, text: str) -> list:
    """Appliances the text refers to, best match first-tier only. Exact word matches (name, brand, model)
    outweigh synonym matches, so 'split AC' picks the split AC even when another AC exists. If several
    appliances tie at the top score, ALL are returned and the caller must ask the user which one."""
    q = set(tokenize(text)) - _FILLER
    q_expanded = _expand(q)
    scored = []
    for a in appliances:
        a_tokens = set(tokenize(f"{a.name} {a.brand or ''} {a.model or ''}")) - _FILLER
        exact = q & a_tokens
        alias_only = (q_expanded & _expand(a_tokens)) - exact
        score = 2 * len(exact) + len(alias_only)
        if score:
            scored.append((score, a))
    if not scored:
        return []
    top = max(s for s, _ in scored)
    return [a for s, a in scored if s == top]


# Words that mark a message as Roman Urdu (none is a common English word). Two distinct markers are required.
_ROMAN_URDU = frozenset("""hai hain kya kyun kyon kaise karun karoon karna karein kar raha rahi rahe mein mera meri mere
kitni kitna kitne chahiye nahi nahin gaya gayi gaye pehle pehli baad kab se ko ka ki ke aur bahut zyada barf thanda
thandi garam khara chalane chalana wala wali kahan kuch abhi hota hoti""".split())
_URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")


def needs_translation(text: str) -> bool:
    """True for Urdu script or Roman Urdu, which English manuals cannot be searched with directly."""
    if _URDU_SCRIPT.search(text):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & _ROMAN_URDU) >= 2
