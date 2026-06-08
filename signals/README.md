# signals — A1 signal panel

**A1 baseline only.** This module implements Sprint 1a of the corpus→signal→strategy pipeline:
blended corpus assembly → consensus extraction → canonicalization → leakage audit → lifecycle
panel. No per-member extraction (A2a/A2b), no dispersion/consensus analysis, no market-data
panel (1b), no strategy/backtest (1c). See the full design:
[`docs/superpowers/specs/2026-06-07-corpus-signal-strategy-design.md`](../docs/superpowers/specs/2026-06-07-corpus-signal-strategy-design.md).

> **Package named `signals` (plural)** to avoid collision with the Python stdlib `signal` module.

---

## Pipeline stages

| Stage | File | What it does |
|---|---|---|
| Config & taxonomy | `config.py` | Paths, `DEFAULT_WINDOW_MONTHS` (18), seed sector list, `STANCE_SIGN` map |
| Schema | `schema.py` | Immutable dataclasses: `Citation`, `SignalItem`, `RiskRegime`, `PeriodSignal` |
| Registry | `registry.py` | Canonical sector/token vocabulary — bookkeeping only; seeded from `SEED_SECTORS` |
| Transcript distillation | `distill.py` | One-time extractive pass: raw transcripts → verbatim, dated, stance-bearing passages → `transcript_distillates.jsonl`. Extractive (never paraphrased) so the leakage audit can substring-check quotes. Cached and resumable. |
| Corpus assembly | `corpus.py` | Assembles the as-of-T blended corpus over the trailing window: tweets (retweets excluded), research articles, distilled transcript passages — sorted chronologically, source-tagged. |
| A1 extraction | `extract.py` | LLM reads the assembled corpus and emits the team's consensus market view in free-form names (no taxonomy injected). Recency-privileging: when statements conflict, the most recent wins. |
| Canonicalization | `canonicalize.py` | Agentic fit-or-mint: maps free-form names to the registry by semantic judgment. Items are never silently dropped — unmapped names fall back to a deterministic slug. |
| Leakage audit | `audit.py` | Verifies every citation quote is a verbatim substring of the in-window corpus and dated ≤ T. Reports `checked`, `matched`, `hallucinated`, `leaked` per period. |
| Panel derivation | `panel.py` | Deterministic lifecycle/delta panel across periods. No LLM. Computes `lifecycle_state` (NEW/SUSTAINED/FLIPPED/EXITED), `delta_stance`, `delta_conviction`, `age`. Emits synthetic EXITED rows when an item drops out. |
| Orchestration | `run.py` | Wires all stages together; exposes the CLI. |

---

## CLI workflow

Run these in order:

```bash
# 1. One-time: distill transcripts → verbatim passage cache
python -m signals.run distill

# 2. Build the signal timeseries panel
python -m signals.run panel --start 2022-12-31 --end 2026-03-31 --interval quarterly

# 3. Validation gate: compare full-text vs distilled extraction on one date
python -m signals.run validate --t 2023-03-31
```

`distill` only needs to run once (it is resumable). `panel` accepts `--window-months`
(default 18) and `--interval quarterly|monthly`. `validate` reports item-set Jaccard between
the full-text and distilled extraction paths.

---

## Design rationale

**Trailing holding-period window (18mo default, not all-history)**
Feeding all corpus ≤ T is infeasible: transcripts alone are ~8.38M chars (~2M tokens) blended.
More importantly, the window is semantically the *longest anticipated holding period* — originating
evidence must stay in view for the life of the trade, but stale evidence from years prior should
fall out. The 18-month default is the starting point; the spec notes 24mo as an alternative to test.

**Extractive transcript distillation**
Transcripts are large and largely filler. The distillation pass extracts only verbatim,
stance-bearing passages, caching them permanently. "Extractive" (no paraphrasing) is not a
style choice — it is required so the leakage audit can confirm quotes as verbatim corpus
substrings. Distillation is separated from the per-period extraction loop so it runs once.

**Recency-privileging extraction**
The A1 prompt instructs the LLM to weight the most recent view when statements conflict.
A stance held once long ago and never restated is flagged as `persisted` (still cited and
tracked, but surfaced with lower confidence than `grounded`).

**Free-form extraction → agentic canonicalization**
The extraction prompt does NOT inject the sector taxonomy. Forcing the model into a fixed
vocabulary during extraction causes false fits and suppresses emerging themes. Instead, the
model names things in its own words; canonicalization (`canonicalize.py`) reconciles those
free-form names to the registry by semantic judgment in a separate LLM call.

---

## Panel schema (`signal_panel.parquet`)

One row per (period, item). Produced by `panel.py`.

| Column | Type | Notes |
|---|---|---|
| `as_of` | str (ISO date) | Rebalance date T |
| `item` | str | Canonical sector or token id |
| `item_type` | str | `sector` or `token` |
| `parent_sector` | str or null | For tokens: their parent sector id |
| `stance` | str | `bullish`, `neutral`, or `bearish` |
| `conviction` | int (0–100) | Intensity, not a calibrated probability |
| `horizon` | str | `tactical` or `structural` |
| `lifecycle_state` | str | `NEW`, `SUSTAINED`, `FLIPPED`, or `EXITED` |
| `delta_stance` | int | Change in stance sign vs prior period (−2 to +2) |
| `delta_conviction` | int | Change in conviction vs prior period |
| `age` | int | Periods item has been continuously present (0 on EXITED) |

---

## Outputs under `data/signal/`

| File | Description |
|---|---|
| `transcript_distillates.jsonl` | One JSON line per transcript: `{object_id, passages: [{date, passage}]}`. Cache; append-only. |
| `registry.json` | Canonical sector/token vocabulary accumulated across all panel runs. |
| `signal_panel.parquet` | The signal timeseries panel (schema above). |
| `periods/<YYYY-MM-DD>.json` | Raw `PeriodSignal` for each rebalance date (pre-panel derivation). |
| `audit.json` | Per-period leakage audit results: `checked`, `matched`, `hallucinated`, `leaked`. |

---

## Open follow-ups

- **Validation gate full-text arm (Task 10b stub):** `validate_distillation` currently passes
  `distillates={}` for the full-text path (transcripts excluded), not a true full-text extraction.
  This is a known stand-in — the full-text arm needs proper corpus loading when transcripts are
  large enough that the comparison is meaningful.
- **Sprint 1b — Coinglass market-data panel + BTC beta:** deferred separate sprint.
- **Sprint 1c — strategy, backtest, eval:** deferred separate sprint.
- **A2a (per-member extraction) and A2b (dispersion/consensus analysis):** unbuilt, deferred.
