"""Retrieval evaluation: turns 'it seems to work' into numbers.

A labeled case says which (document, page) locations answer a question, and optionally a phrase
(`must_contain`) that the retrieved chunk must actually contain. The phrase makes the metric
chunk-level: without it, the right page but the WRONG chunk would count as a hit even though the
answer never reaches the LLM. Metrics are computed on
ANSWERABLE cases; UNANSWERABLE cases are used to record how high the best (wrong) result scores,
which is how Step 4 will choose its 'I couldn't find this in your documents' threshold.

* hit rate @k  (reported as recall_at[k]): share of answerable questions with at least one correct
  location in the top k. With one correct location per question this equals classic Recall@k.
* MRR: mean of 1/rank of the first correct result (0 when none is found).
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from gharcare.errors import DomainValidationError
from gharcare.retrieval.hybrid import HybridRetriever, RetrievedChunk

Location = tuple[str, int]  # (document filename, page number)


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant: frozenset[Location] = frozenset()
    answerable: bool = True
    must_contain: str | None = None   # evidence phrase the retrieved chunk must include

    def __post_init__(self) -> None:
        if self.answerable and not self.relevant:
            raise DomainValidationError(f"answerable case needs relevant locations: {self.question!r}")


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainValidationError(f"cannot read evaluation file {path}: {exc}") from exc
    return [EvalCase(
        question=item["question"], answerable=bool(item.get("answerable", True)),
        relevant=frozenset((r["document"], int(r["page"])) for r in item.get("relevant", [])),
        must_contain=item.get("must_contain"),
    ) for item in raw]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def first_relevant_rank(results: Sequence[RetrievedChunk], relevant: frozenset[Location],
                        must_contain: str | None = None) -> int | None:
    """1-based rank of the first result in a relevant (document, page) whose text also contains
    `must_contain` (case-insensitive) when one is given; else None."""
    needle = _squash(must_contain) if must_contain else None
    for rank, r in enumerate(results, start=1):
        if (r.document_name, r.chunk.page_number) in relevant and (
                needle is None or needle in _squash(r.chunk.text)):
            return rank
    return None


@dataclass
class RetrievalReport:
    k_values: tuple[int, ...]
    n_answerable: int = 0
    n_unanswerable: int = 0
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    misses: list[str] = field(default_factory=list)                    # answerable, not found in top max-k
    answerable_top_similarity: list[float] = field(default_factory=list)
    unanswerable_top_similarity: list[float] = field(default_factory=list)
    answerable_top_keyword: list[float] = field(default_factory=list)
    unanswerable_top_keyword: list[float] = field(default_factory=list)

    def format(self) -> str:
        lines = [f"Answerable questions: {self.n_answerable}   Unanswerable: {self.n_unanswerable}"]
        lines += [f"  Hit rate @{k}: {self.recall_at[k]:.2f}" for k in self.k_values]
        lines.append(f"  MRR: {self.mrr:.3f}")
        if self.misses:
            lines.append("  Missed: " + "; ".join(self.misses))

        def span(values: list[float]) -> str:
            return (f"min {min(values):.3f} / median {statistics.median(values):.3f} / max {max(values):.3f}"
                    if values else "n/a")
        lines.append("Best-result signals (for calibrating the 'I don't know' threshold in Step 4):")
        lines.append(f"  vector similarity  answerable: {span(self.answerable_top_similarity)}")
        lines.append(f"  vector similarity  unanswerable: {span(self.unanswerable_top_similarity)}")
        lines.append(f"  keyword (BM25)     answerable: {span(self.answerable_top_keyword)}")
        lines.append(f"  keyword (BM25)     unanswerable: {span(self.unanswerable_top_keyword)}")
        return "\n".join(lines)


def evaluate_retrieval(retriever: HybridRetriever, user_id: str, cases: Sequence[EvalCase],
                       k_values: Sequence[int] = (1, 3, 5), appliance_id: str | None = None) -> RetrievalReport:
    ks = tuple(sorted(set(k_values)))
    report = RetrievalReport(k_values=ks)
    hits_at = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []
    for case in cases:
        results = retriever.retrieve(user_id, case.question, appliance_id=appliance_id, k=max(ks))
        top_sim = max((r.vector_similarity or 0.0 for r in results), default=0.0)
        top_kw = max((r.keyword_score or 0.0 for r in results), default=0.0)
        if not case.answerable:
            report.n_unanswerable += 1
            report.unanswerable_top_similarity.append(top_sim)
            report.unanswerable_top_keyword.append(top_kw)
            continue
        report.n_answerable += 1
        report.answerable_top_similarity.append(top_sim)
        report.answerable_top_keyword.append(top_kw)
        rank = first_relevant_rank(results, case.relevant, case.must_contain)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            # Retrieve AT k: the result set at k is not always a prefix of a larger k (reserved slots depend on k),
            # so slicing a k=5 list once hid a bug that only showed at the assistant's k=4.
            at_k = results if k == max(ks) else retriever.retrieve(user_id, case.question, appliance_id=appliance_id, k=k)
            r_k = first_relevant_rank(at_k, case.relevant, case.must_contain)
            if r_k is not None and r_k <= k:
                hits_at[k] += 1
        if rank is None:
            report.misses.append(case.question)
    if report.n_answerable:
        report.recall_at = {k: hits_at[k] / report.n_answerable for k in ks}
        report.mrr = sum(reciprocal_ranks) / report.n_answerable
    else:
        report.recall_at = {k: 0.0 for k in ks}
    return report
