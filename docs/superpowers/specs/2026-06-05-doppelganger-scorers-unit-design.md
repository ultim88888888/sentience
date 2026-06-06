# Corpus Doppelganger — Unit ❺b Scorers + Findings — Design Spec

_Design spec · 2026-06-05 · status: draft-pending-review_

## 0. One-paragraph summary

The **scorers** turn the walk-forward trajectory into an honest answer to *"is the doppelganger right, and is the corpus doing the work — or is it Opus's hindsight?"* It (1) fixes the ablation to a **clean soul-less baseline**, (2) runs a **held-out prediction judge** (`claude -p`) comparing each step's view@T against the subject's *actual* statements in `[T, T+6mo]`, scoring each claim confirmed/contradicted/absent, (3) computes the **headline metric — corpus-lift = full confirm-rate − soul-less confirm-rate** — plus change-detection, (4) measures **discrimination** (Eddy vs Ali) and **coverage** deterministically, and (5) emits a `metrics.json` + a templated `findings.md`. This is Unit ❺b — the final piece. Building the scorers is this spec; the full quarterly walk-forward *run* (both subjects, ~2 hr) is a separate execution afterward.

## 1. Purpose & scope

### Goal
Score the walk-forward trajectory into trustworthy metrics: does the corpus-grounded doppelganger predict the subject's real later stance **above the soul-less parametric floor**, and are the two subjects distinct minds?

### Scope (5b code)
- The soul-less ablation fix, the held-out judge, discrimination, coverage, the scoring orchestrator, the memo-generator. All `claude -p` (judge only) is mockable for TDD.
- Validate on the existing **3-step Eddy subset**; the full run is separate.

### Non-goals (v1)
- No third subject. No citation-stripped middle rung (deferred). No live trading. The interpretive narrative is written by hand, not auto-generated.

### Success criteria
A `metrics.json` + `findings.md` for the Eddy subset showing per-step confirm-rates for both arms, the corpus-lift, the missed-changes list, coverage trajectory, and Eddy-vs-Ali discrimination — with `0 leaked` holding across all views.

## 2. The clean (soul-less) ablation — `respond` fix

Change the ablation arm from soul-only (confounded — embeds 66 quotes) to **soul-less / pure-parametric**. In `respond`, the ablation arm uses **no soul card and no memory** — only a one-line identity stub:
```
You are {name}, {headline}.   # e.g. "Eddy Lazzarin, Investing in crypto with a16z."
```
built from `data/doppelganger/<slug>/identity.json` (`name`, `headline`). Memory stays empty. Output still goes to `views_ablation/<t0>.json`. The walk-forward's `ablation` variant now means **soul-less**; the existing (confounded) `views_ablation` are regenerated.

Rationale: the only arm with **zero corpus-derived content**, so its score is the true model-hindsight floor. `full − soul-less` is everything our pipeline adds over what Opus already knew. Cost-neutral (still 2 arms/step).

## 3. The held-out prediction judge (`judge.py`)

**Post-T evidence slice.** `post_t_evidence(slug, t0, horizon_months=6) -> str` — the subject's real evidence in `(t0, t0+horizon]`, formatted chronologically (`[date] (source) text`). This is the held-out future the doppelganger never saw.

**The judge.** `judge_step(view: dict, post_t_text: str, subject_name: str, t0: date) -> dict` — one `claude -p` (Opus, isolated, mockable via `run_claude`) that reads the doppelganger's view + the post-T statements and returns a structured verdict:
```json
{
  "claims": [ { "claim": "excited about ZK rollups", "axis": "sectors_excited",
                "label": "confirmed|contradicted|absent" } ],
  "n_confirmed": 5, "n_contradicted": 1, "n_absent": 3,
  "confirm_rate": 0.83,                       // confirmed / (confirmed + contradicted); absent excluded
  "missed_changes": [ "Started emphasizing restaking after T; the view didn't anticipate it" ],
  "notes": "..."
}
```
- **Per-claim, stance-level** — confirmed/contradicted is about the *stance*, not wording.
- **`confirm_rate = confirmed / (confirmed + contradicted)`** — absent claims excluded (he simply didn't address them; not a hit or miss).
- **`missed_changes`** — stances he *newly took or reversed* in the window that the view didn't anticipate. The change-detection / foresight signal.
- **Honesty guard (in the judge's system prompt):** score **only** from the provided post-T statements; do NOT use anything you know about what happened after the window. The judge is reading-comprehension over given text, not prediction.

**Caching/resumability:** each verdict is written to `data/doppelganger/<slug>/judge/<t0>_<variant>.json`; a step whose verdict exists is not re-judged (judge calls are `claude -p`).

## 4. Deterministic scorers

**Discrimination** (`score.py`). `discrimination(view_a, view_b) -> dict` — Jaccard overlap of the **named** sectors/tokens between two subjects' views at the same step (case-normalized names). Low overlap = distinct minds. Returns per-axis overlap + the shared/unique name sets. No `claude -p`.

**Coverage** (`score.py`). `coverage_trajectory(walkforward_rows) -> list[dict]` — reshapes the full-arm rows from `walkforward.json` into the grounded/persisted/extrapolated trajectory over time. Pure data; no `claude -p`.

## 5. Scoring orchestrator + memo (`score.py`)

**`score_subject(slug, *, horizon_months=6) -> dict`** — for each step in `walkforward.json`: load the full and ablation views, run `judge_step` on each (cached), record per-step `{date, full_confirm_rate, ablation_confirm_rate, lift, missed_changes, ...}`. Aggregate: mean lift, per-step lift trajectory, coverage trajectory. Write `data/doppelganger/<slug>/metrics.json`.

**`write_memo(metrics, discrimination=None) -> Path`** — **deterministic template** → `data/doppelganger/<slug>/findings.md`: the corpus-lift headline (with the persistence + leakage caveats stated inline), the per-step confirm-rate table (full vs ablation), the missed-changes list, the coverage trajectory, and (if a second subject's metrics are supplied) the discrimination overlap. No LLM writes the conclusions — the numbers populate a fixed template; interpretation is added by hand.

**CLI:** `score` subcommand — `python -m doppelganger.run score --subject eddy-lazzarin [--horizon-months 6]` → runs the judge over the trajectory, writes `metrics.json` + `findings.md`.

## 6. Module layout

```
doppelganger/
  respond.py     # MODIFY: ablation arm -> soul-less (identity stub, no soul, no memory)
  judge.py       # NEW: post_t_evidence, judge_step (claude -p)
  score.py       # NEW: discrimination, coverage_trajectory, score_subject, write_memo
  run.py         # MODIFY: add `score` subcommand
tests/
  test_doppelganger_judge.py
  test_doppelganger_score.py
```

## 7. Testing strategy

- **soul-less ablation** (mocked `run_claude`): ablation arm's prompt contains the identity stub, NOT the soul prose, and empty memory; writes to `views_ablation/`.
- **judge** (mocked `run_claude`): `post_t_evidence` slices the right window; `judge_step` parses a canned verdict JSON (incl. fenced), caches it, skips re-judging when the verdict file exists.
- **discrimination / coverage** (pure): overlap math on hand-built views; coverage reshape on hand-built rows.
- **score_subject** (mocked judge): aggregates per-step into lift + writes `metrics.json`; resumable.
- **memo** (pure): `write_memo` renders the headline, table, missed-changes, coverage from a metrics dict deterministically.
- **CLI:** `score` subcommand parses.
- **Real-data validation (not TDD):** regenerate the Eddy subset's soul-less ablation (5 `claude -p`), then `score_subject("eddy-lazzarin")` over the 3 steps (judge calls, `claude -p`); read `findings.md` — is there positive corpus-lift? are there missed changes? Honest read either way.

## 8. Open questions for planning

1. **Horizon** — default `horizon_months=6` (next ~2 quarters). Confirm; the last step(s) of the walk-forward will have a truncated/empty future window — handle by skipping steps with < N post-T items.
2. **Discrimination name-normalization** — exact matching of sector/token names across subjects (case + light synonym folding) — keep simple (case-fold) at plan time.

---

_After 5b: the full quarterly walk-forward run (both subjects, ~2 hr), scored → the first honest findings on whether the corpus-doppelganger predicts above the parametric floor._
