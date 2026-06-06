# Corpus Doppelganger — Unit ❸ Memory — Design Spec

_Design spec · 2026-06-05 · status: draft-pending-review_

## 0. One-paragraph summary

**Memory** is the time-gated view of the subject's dated utterances that the doppelganger reasons over at query time. The soul holds the *apparatus* that generates views; memory holds the *specific dated opinions*. Because the subject's full corpus (~200k tokens) fits inside Opus's context with room to spare at every walk-forward step, v1 does **no retrieval** — it filters evidence to `≤ T` behind a hard leakage firewall and hands the doppelganger *all* of it, letting the model select natively. A `query` parameter is accepted but ignored: the seam where retrieval drops in later if (and only if) the eval shows the full corpus dilutes answers.

## 1. Purpose & scope

### Goal
`load_memory(slug, t0) -> MemoryView` — return the subject's evidence dated `≤ t0`, formatted for the doppelganger's prompt, with an auditable guarantee that nothing after `t0` is included.

### Why feed-all, not RAG (concretizes engine spec §4 ❸, defers its retrieval assumption)
The engine spec assumed hybrid RAG because "the corpus is bigger than the context window." Measured, that premise is false: Eddy's ≤T0 corpus is ~40k tokens and his *full* corpus ~200k, vs. Opus's 1M context. So:
- Our primary query is a **survey** ("what sectors/tokens is he excited/concerned about") — diffuse intent, no term to retrieve against. **Completeness beats focus**, and any retrieval step risks dropping a view (recall failure) for a capacity benefit we don't need.
- RAG only plausibly wins on **narrow lookups** ("his take on rollups"), which aren't the primary use.
- Therefore feed-all is the best *reasoned* bet for our query shape **and** the simplest. Whether long-context dilution beats retrieval recall on narrow queries is an **empirical question the eval (Unit 5) settles** — and the trigger to add RAG, surgically, behind the seam below.

### Scope (v1)
- Time-gated, firewalled, chronological feed of ≤T evidence. No embeddings, no ranking, no clustering.
- `query` param accepted, ignored (retrieval seam).

### Non-goals (v1)
- No semantic/lexical retrieval, no embeddings (local or API), no topic clustering, no recency/salience weighting. All deferred behind the seam until the eval shows a measured need.

### Success criteria
The firewall is provably correct (`max_date ≤ t0` always; an item dated `> t0` never appears), and the formatted view is what the doppelganger (Unit 4) consumes.

## 2. The leakage firewall (the one non-negotiable)

This is the unit's whole reason to exist as a standalone module. The `≤ t0` filter is applied **first, before any other processing**. `MemoryView` carries `max_date` so any consumer (the doppelganger, the eval) can assert `max_date ≤ t0` — leakage is *proven*, not assumed. This is the single auditable home for the time-gate that makes the entire walk-forward honest.

## 3. Interface

```python
@dataclass
class MemoryView:
    items: pd.DataFrame      # the <= t0 evidence, sorted chronologically
    text: str                # formatted block the doppelganger reads
    n_items: int
    max_date: date | None    # leakage guarantee: must be <= t0 (None if empty)


def load_memory(
    slug: str, t0: date, *,
    evidence_path: Path | None = None,
    query: str | None = None,    # accepted, IGNORED in v1 — the retrieval seam
) -> MemoryView:
    ...
```

- Reads `data/doppelganger/<slug>/evidence.parquet` (Unit 1 output) unless `evidence_path` given.
- **Firewall:** `ev[ev["timestamp"].dt.date <= t0]`, sorted ascending.
- **Formatting:** chronological lines, each `[<YYYY-MM-DD>] (<source_type>) <text>`, with `(context: …)` appended where present — tuned for the doppelganger (its own formatter, decoupled from the soul extractor's).
- `query` is documented as the seam: when retrieval lands, it filters/ranks `items` before formatting; the signature and every call site stay unchanged.

## 4. Module layout

```
doppelganger/
  memory.py        # MemoryView, load_memory
  run.py           # add `memory` subcommand (inspection: prints n_items, max_date, sample)
tests/
  test_doppelganger_memory.py
```
Consumes Unit 1 (`evidence.parquet`, `config.py`). Independent of the soul module.

## 5. Testing strategy (TDD)

- **Firewall (the core):** an item dated `> t0` is excluded; `max_date ≤ t0`; all `items.timestamp ≤ t0`.
- **Chronological order:** `items` sorted ascending; `text` lines in date order.
- **Formatting:** `text` contains each item's date, source_type, and text; context appended when present.
- **Empty / edge:** subject with no ≤t0 evidence → `n_items=0`, `max_date=None`, `text=""`.
- **Seam:** passing `query="anything"` returns the same full ≤t0 view (ignored in v1).
- **CLI:** `memory` subcommand parses `--subject` / `--t0`.

## 6. Open questions for planning

1. **Formatter sharing** — the soul extractor (`soul.build_extraction_prompt`) already formats an evidence block; decide at plan time whether to extract a shared `format_evidence(df)` helper or keep memory's formatter independent (lean: independent for now; they may diverge as the doppelganger's needs differ).

---

_Next: on approval, `writing-plans` for the memory unit. Then Unit ❹ (Doppelganger) consumes `soul.md` + `MemoryView`._
