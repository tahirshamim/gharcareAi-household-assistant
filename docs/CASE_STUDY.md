# GharCare AI: case study

A household assistant for appliances, manuals, warranties, maintenance, repairs and expenses, built for
households in Pakistan (English, Roman Urdu, Urdu script). This document is the summary for reviewers:
what was built, how it meets the brief, how it was evaluated, what the numbers are, and what is still open.
Detail lives in [ARCHITECTURE.md](ARCHITECTURE.md) and the [README](../README.md).

## 1. Problem

Household information is scattered: paper manuals, warranty cards, receipts, technician invoices, memory.
People ask practical questions ("why is my fridge leaking?", "is the AC still under warranty?", "how much did
repairs cost this year?") and a generic chatbot answers them by guessing. The goal was an assistant that
**answers from the household's own documents and records, cites them, and says so when it does not know**.

Principles agreed at the start:

* Never invent specifications, warranty coverage, document contents or repair history.
* Document answers cite the page they came from; "not found" is a valid answer.
* Hazards (gas, burning smell, sparks, smoke, shock) are escalated to a technician, never walked through as DIY.
* Nothing extracted by AI is saved until the user reviews and confirms it.
* Success is measured with labeled questions and confidence intervals, not demos.

## 2. How the brief is met

| Required | Where it is, and what real job it does |
|---|---|
| **RAG** | `ingestion/` (structure-aware chunking: chunks never cross a page or heading) and `retrieval/` (hybrid vector + BM25 with Reciprocal Rank Fusion). Every troubleshooting answer is drafted only from retrieved passages, must cite them, and is checked by a second model call before it is shown. |
| **Vector database** | Chroma (`retrieval/vectorstore.py`) with the `intfloat/multilingual-e5-small` embedder. An in-memory store implements the same interface, and both pass one shared contract test. SQLite holds the chunk text, so the index can always be rebuilt. |
| **LangGraph / LangChain** | `workflow/`: the assistant is a graph (safety screen -> intent -> appliance resolution -> deterministic record answer, or retrieve -> history -> draft -> verify). It runs on LangGraph. A built-in runner executes the same spec offline, and a parity test proves both give identical results. LangChain chat models are reached through one adapter with error mapping. |
| **Object-oriented design** | Frozen, validated domain dataclasses; abstract repository, storage, LLM, embedder and vector-store interfaces with swappable implementations; services that receive dependencies through constructors; decorators for retry, pacing, caching and metering. |
| **Artifact / document storage** | `storage/`: originals are content-addressed (SHA-256), read-only and deduplicated, and the user's filename never touches the disk path. Extractions are versioned (`v0001`, `v0002`, ...) and never overwritten, keeping both the AI draft and the user-confirmed values. Reports are versioned Markdown. |

Around these: deterministic warranty, maintenance, expense and report services (exact `Decimal` money, honest
"data gaps"); receipt, warranty-card and invoice extraction (rules + AI reader, where every AI value must appear
in the document text); a FastAPI backend and a web page; and an evaluation harness.

## 3. Key design decisions

* **The LLM never computes.** Totals, dates, warranty status and due maintenance come from code. The model has
  four narrow jobs: fallback intent for non-questions, translating Roman Urdu / Urdu questions for search,
  drafting a cited answer, and judging whether that answer is supported.
* **Safety runs before any model call**, in English, Roman Urdu and Urdu, so it still works when the LLM is down
  and cannot be argued out of escalating.
* **Degrade, don't fail.** If the LLM is unavailable or an answer cannot be verified, the user gets the raw cited
  passages instead of an unverified claim.
* **Free-tier realities are designed in.** Requests are paced under the per-minute limit, Gemini's requested
  retry delay is honoured, a daily-quota error is not retried, and evaluations are cached and budgeted so a
  stopped run resumes without spending quota again.

## 4. Evaluation method

* **Document:** a real, text-based 34-page Gorenje refrigerator-freezer manual (third-party, not redistributed).
* **Questions:** 42 hand-labeled, with 33 answerable and 9 deliberately unanswerable (capacity, price, WiFi,
  energy use: things this manual does not state). Each answerable question has its correct page(s) and an evidence
  phrase verified against the PDF text, so a hit means the right chunk, not just the right page. Two questions are
  in Roman Urdu.
* **Retrieval metrics:** hit rate at k and MRR, measured at the same k the assistant uses (4).
* **End-to-end outcomes** per question: correct with a relevant citation, wrong citation, unverified (passages
  shown instead), abstained, safety escalation by design; for unanswerable questions: correctly refused, or
  answered anyway (hallucination risk). Wilson 95% confidence intervals for every rate.
* **Model:** Gemini Flash-Lite on the free tier. **Limitation:** the judge is the same model family as the writer.

## 5. Results

All numbers below were produced on the developer's machine with the final code (Step 7), on the real manual.

### Retrieval (no LLM; multilingual-e5-small + Chroma, 60 chunks)

| hit@1 | hit@3 | **hit@4 (what the assistant reads)** | hit@5 | MRR |
|---|---|---|---|---|
| 0.58 | 0.82 | **0.91** | 0.91 | 0.709 |

Remaining retrieval misses at k=4: "Can I use a power strip behind the fridge?" (the manual says "portable socket-outlets or
portable power supplies") and the two Roman Urdu questions, which the assistant now translates before searching.
Earlier retrieval figures in the project history (for example hit@1 0.70) came from older code versions with a
different chunk count, including runs where a stale install was loaded, so they are not comparable and are not quoted here.

### End-to-end (Gemini Flash-Lite, LangGraph, all 42 questions)

| | Before inspection fixes | **Final** | Final 95% CI |
|---|---|---|---|
| Answerable, correct with a relevant citation | 25 / 33 (75.8%) | **29 / 33 (87.9%)** | 73-95% |
| Answerable, **wrong citation** | 0 / 33 | **0 / 33** | 0-10% |
| Answerable, unverified (passages shown instead) | 3 / 33 | 1 / 33 | 1-15% |
| Answerable, abstained | 4 / 33 | 2 / 33 | 2-20% |
| Answerable, safety escalation by design | 1 / 33 | 1 / 33 | |
| Unanswerable, correctly refused | 9 / 9 | **9 / 9** | 70-100% |
| Unanswerable, **answered anyway** | 0 / 9 | **0 / 9** | 0-30% |
| LLM calls per question | 1.6 | 1.3 | |

The fixes recovered 4 of the 7 misses, including one of the two Roman Urdu questions. The confidence intervals of the
two runs overlap, so with 33 questions the improvement is consistent with the targeted fixes but not statistically
conclusive. The safety properties matter most: in both runs **no answer cited a wrong page, and no unanswerable question
was answered**.

Remaining misses: the power-strip question (retrieval, vocabulary gap), the Roman Urdu "how long should it stand
upright" question (translated, but the model still declined), and "how much can the mains voltage vary" (correct
draft without a citation, and the judge did not name a supporting passage, so the system showed passages instead).

## 6. What evaluation found (and what it fixed)

Most of the improvement came from measuring, then reading individual failures, not from adding features.

| Finding | How it was found | Fix |
|---|---|---|
| A footer code on every page polluted keyword search; wrapped headings split; diagram labels became chunks | First run on the real manual | Strip repeated headers/footers, merge wrapped headings, drop fragments |
| Rank fusion buried an exact keyword match | Retrieval misses on real questions | Guarantee each retriever's best results a place |
| Semantic retrieval: hit@1 0.58 -> 0.70 | Retrieval evaluation | Switched from the lexical baseline to multilingual-e5 + Chroma |
| 6 of 20 manual questions answered with the expense report or help text | First end-to-end run | "How much" needs a money word to mean expenses; questions go to documents by rule (also saves a call) |
| Retries ignored Gemini's "retry in 36s" and failed again | Rate-limit errors in the run log | Honour the provider's delay; pace requests under 15/min |
| The guaranteed slots could fill all 4 places and evict the best match | Replaying the misses (`examples/inspect_cases.py`, zero LLM calls) | Reserve at most half the slots |
| The retrieval evaluation hid that bug: it measured k=5 and sliced | Same replay | Retrieve separately at each k |
| Correct answers discarded: citation `[2, 3]` rejected; uncited but correct draft dropped; the judge rejected the safety advice our own prompt requires | Same replay | Accept comma citations; the judge names supporting passages; the judge is told safety advice is allowed |
| Roman Urdu questions searched an English manual untranslated | Same replay | Translate for search only; answer from the original question |
| Inline styles silently blocked by the CSP; favicon 404 | First real-browser test run | CSS classes, `h()` refuses `style`, data-URI favicon |
| Repair history did not match "fridge mein barf jam" to an "Ice build-up in freezer" record | Running the demo | Small English / Roman Urdu / Urdu fault glossary and Roman Urdu filler words |
| The answer card repeated its sources twice and mixed answer, records and citations in one text block; an empty value printed as "null" | Screenshots of the redesign | Structured answer parts from the API (`parts`); a browser check that fails on printed null/undefined |
| A refused database (from a newer app version) stayed open, locking the file on Windows | ResourceWarning in the user's test run | Close the connection when migration fails |
| New scripts ran an old copy of the library (editable install in another folder) | A run that made 0 LLM calls when new prompts required them | Scripts and tests load the code next to them; `doctor` warns about a mismatched install |

Mutation testing (re-introducing each bug deliberately) was used on the critical logic, and caught several tests
that passed for the wrong reasons.

## 7. Verification status

Verified on the developer's machine: the full test suite (357 tests, including Chroma, PDF, LangGraph parity and FastAPI
HTTP tests; only the 4 browser tests skipped there); live Gemini answers and extraction; sentence-transformers + Chroma;
the LangGraph backend; the web app starting and serving the page; the numbers in section 5.

Verified in the development sandbox: real-browser tests in headless Chrome via Playwright (every tab renders with no console errors; add an appliance, upload a manual,
ask and get a cited answer; safety escalation; hostile document text is shown as text, never run as HTML). The
browser tests found two bugs the static checks had missed: inline styles silently blocked by the Content Security
Policy, and a favicon 404.

Not yet verified: the browser tests on the developer's machine; Ollama;
extraction on real receipts and warranty cards (only three synthetic documents exist, and the rules were tuned on
them); the CI workflow (`.github/workflows/tests.yml`) on GitHub.

## 8. Limitations

* No OCR: scanned or photographed documents are stored but not searchable, and the user is told so.
* One labeled manual and 42 questions: confidence intervals are wide, and results may not transfer to other brands.
* Single local user with an optional bearer token; no accounts or sessions. Free LLM tiers may use submitted data,
  so only sample or public documents should be used on them.
* The safety screen and similar-repair matching are keyword/glossary based and miss unusual phrasings.
* The judge shares the writer's model family and can share its blind spots.

## 9. Next steps

1. Close the remaining vocabulary gaps (query expansion or a stronger multilingual model such as bge-m3) and re-measure.
2. Collect real (anonymised) receipts and warranty cards; build a labeled extraction set and harden the rules.
3. Label a second manual from another brand to test generalisation.
4. Use a different model as the judge; calibrate an "I don't know" threshold from the exposed retrieval scores.
5. Production: accounts and sessions, HTTPS, Docker, a paid or local model, and artifact retention / erase.
