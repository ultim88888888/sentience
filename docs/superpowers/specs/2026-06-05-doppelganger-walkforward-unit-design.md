# Corpus Doppelganger — Unit ❺a Walk-forward Runner — Design Spec

_Design spec · 2026-06-05 · status: draft-pending-review_

## 0. One-paragraph summary

The **walk-forward runner** drives the doppelganger across time: for each date on a quarterly schedule (T0 = 2022-12-31 → present), it produces the subject's **full-corpus** market view *and* an **ablation** (soul-only, no-memory) view, audits each for leakage, and records a **trajectory + coverage map**. It is **resumable** — each `respond` step is a ~4–5 min `claude -p` call, so a completed step is never re-run. This is Unit ❺a: it produces the raw material; the scorers (held-out prediction, corpus-lift, discrimination) are Unit ❺b, a separate cycle.

## 1. Purpose & scope

### Goal
`run_walkforward(slug, dates) -> trajectory` — ensure a full and an ablation view exist for every date, audited, and emit `data/doppelganger/<slug>/walkforward.json` (per-step, per-variant: provenance mix, counts, risk stance, leakage).

### Why full + ablation per step
The eval's honest headline metric (settled in the engine design) is **corpus-lift over the ablation baseline** — the full-corpus score *minus* the soul-only score — which simultaneously nets out persistence-triviality and bounds model-hindsight leakage. So the runner must produce **matched (full, ablation) pairs** at every step; 5b computes the lift from them.

### Scope (5a only)
- The runner, the `respond` ablation arm, and the trajectory/coverage artifact.
- **Quarterly** schedule (~14 steps), not monthly — halves cost; a slow-moving VC's views don't shift monthly. Monthly is a later refinement (the schedule is a parameter).
- Validate on a **3-step subset** for one subject; the full quarterly run (both subjects, ~2 hr) is a separate long execution, not part of the build.

### Non-goals (5a)
- No scoring (held-out judge, lift computation, discrimination) — that's 5b.
- No monthly run, no third subject. The artifact is the deliverable; interpreting it is 5b.

### Success criteria
For the 3-step subset: full + ablation views exist for each date, every view passes the `0 leaked` audit, and `walkforward.json` captures the per-step coverage/trajectory correctly.

## 2. The `respond` ablation arm (small extension to Unit 4)

Add one parameter to `respond` (`doppelganger/respond.py`):
```python
def respond(slug, t0, *, query=None, soul_path=None, evidence_path=None,
            out_dir=None, ablate_memory: bool = False) -> dict:
```
- When `ablate_memory=True`: build the prompt with **empty memory text** (soul only — "be the subject at T with no record of specific statements") and write to `<out_dir>/<slug>/views_ablation/<t0>.json` instead of `views/<t0>.json`.
- The build_query_prompt call passes `""` as `memory_text`; everything else identical. This isolates "what the model produces as the subject-at-T from the soul + its own parametric knowledge, without his actual statements" — the leakage/persistence floor.

## 3. The runner

```python
def quarter_ends(start: date, end: date) -> list[date]:
    """Quarter-end dates (Mar 31, Jun 30, Sep 30, Dec 31) in [start, end], inclusive of start if it is one."""

def run_walkforward(
    slug: str, dates: list[date], *,
    ablate: bool = True,
    out_dir: Path | None = None,
    evidence_path: Path | None = None,
    soul_path: Path | None = None,
) -> list[dict]:
    """For each date: ensure full (and, if ablate, ablation) view exists (cached respond),
    audit it, collect a trajectory row. Write <out_dir>/<slug>/walkforward.json. Return the rows."""
```

- **Caching / resumability:** before calling `respond` for a (date, variant), check whether its `view.json` exists; if so, load it and skip the `claude -p` call. This makes the multi-hour run restartable and lets the subset validation reuse Unit 4's already-generated `views/2022-12-31.json`.
- **Per-step audit:** run `audit_answer(view, evidence_path, t0)` on each produced view; record `leaked`, `hallucinated`, `matched`, `checked`. **`leaked > 0` on any step is flagged loudly** (it should be 0).
- **Trajectory row** (one per date × variant) captures: `date`, `variant` (`full`|`ablation`), `abstained`, `risk` (`risk_regime.stance`), counts `n_sectors_excited/concerned`, `n_tokens_excited/concerned`, provenance counts `grounded/persisted/extrapolated`, and audit `leaked/hallucinated/matched/checked`.
- **Output:** `data/doppelganger/<slug>/walkforward.json` — `{subject, dates, rows: [...]}`.

## 4. Module layout

```
doppelganger/
  respond.py       # MODIFY: add ablate_memory param + views_ablation output path
  walkforward.py   # NEW: quarter_ends, run_walkforward
  run.py           # MODIFY: add `walkforward` subcommand (--subject [--start --end])
tests/
  test_doppelganger_walkforward.py
```

## 5. Testing strategy (TDD)

- **`quarter_ends`:** correct quarter-end dates across a span (e.g. 2022-12-31 → 2023-09-30 → [2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30]); handles a start that is/ isn't a quarter-end.
- **`respond` ablation arm** (mocked `run_claude`): `ablate_memory=True` builds the prompt with empty memory (assert the memory feed is absent from the user payload) and writes to `views_ablation/<t0>.json`.
- **`run_walkforward`** (mocked `respond` / `run_claude`): caching (a date whose view.json already exists is NOT re-run — assert `respond`/`run_claude` not called for it); produces full + ablation rows per date; trajectory rows carry the right coverage/risk/audit fields; `walkforward.json` written.
- **CLI:** `walkforward` subcommand parses `--subject` (+ optional `--start`/`--end`).
- **Subset validation (not TDD):** `run_walkforward("eddy-lazzarin", quarter_ends(2022-12-31, 2023-06-30))` — 3 dates; reuses the existing 2022-12-31 full view, generates the rest (full + ablation) via real `claude -p` (~5 steps × ~4 min). Confirm: all `leaked == 0`; `walkforward.json` shows a sane coverage trajectory; eyeball that the ablation views are visibly thinner/more-extrapolated than the full views (a first read on whether the corpus is load-bearing).

## 6. Open questions for planning

1. **Schedule bounds** — `end` defaults to the latest date present in the subject's evidence (don't walk past where there's data to be "as of"). Confirm at plan time.
2. **`walkforward.json` exact shape** — list-of-rows vs nested; lean flat list-of-rows for easy 5b consumption + pandas.

---

_Next: on approval, `writing-plans` for 5a. Then Unit ❺b (scorers + findings) — held-out prediction judge, corpus-lift, discrimination — consumes these trajectories._
