# Corpus Doppelganger — Unit ❺a Walk-forward Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the doppelganger across a quarterly schedule, producing matched full + ablation (soul-only) market views per step, auditing each, and recording a resumable trajectory + coverage map.

**Architecture:** Extend `respond` with an `ablate_memory` arm (empty-memory, soul-only, written to `views_ablation/`). Add `doppelganger/walkforward.py` (`quarter_ends`, `run_walkforward`) that caches by view-file existence (never re-runs a completed `claude -p` step), audits each view, and writes `walkforward.json`.

**Tech Stack:** Python 3.13, pandas, `claude` CLI. pytest. Builds on Units 1-4.

**Spec:** `docs/superpowers/specs/2026-06-05-doppelganger-walkforward-unit-design.md`.

---

## File structure

```
doppelganger/
  respond.py       # MODIFY: add ablate_memory param + views_ablation output
  walkforward.py   # NEW: quarter_ends, run_walkforward, _row
  run.py           # MODIFY: add `walkforward` subcommand
tests/
  test_doppelganger_walkforward.py
```

---

## Task 1: `respond` ablation arm

**Files:** Modify `doppelganger/respond.py`; Test `tests/test_doppelganger_respond.py`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_doppelganger_respond.py`)

```python
def test_respond_ablate_memory_uses_empty_memory_and_separate_dir(tmp_path):
    captured = {}

    def fake_run(system, user):
        captured["user"] = user
        return '{"sectors_excited":[{"name":"X","provenance":"extrapolated","citations":[]}]}'

    with patch("doppelganger.respond.run_claude", side_effect=fake_run):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                       soul_path=_soul(tmp_path), ablate_memory=True)
    # written to the ablation dir, NOT the normal views dir
    assert (tmp_path / "testy-mctest" / "views_ablation" / "2022-12-31.json").exists()
    assert not (tmp_path / "testy-mctest" / "views" / "2022-12-31.json").exists()
    # the record section is present but EMPTY (no memory fed)
    assert "YOUR RECORD" in captured["user"]
    between = captured["user"].split("YOUR RECORD", 1)[1].split("# QUESTION", 1)[0]
    assert between.strip().splitlines()[1:] == []   # nothing between the header line and the question
    assert view["sectors_excited"][0]["provenance"] == "extrapolated"
```

(Uses the existing `_soul` / `patch` / `date` helpers already in this test file from Unit 4.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -k ablate -v`
Expected: FAIL — `respond() got an unexpected keyword argument 'ablate_memory'`.

- [ ] **Step 3: Modify `respond` in `doppelganger/respond.py`** — replace the existing `respond` function with:
```python
def respond(slug: str, t0: date, *, query: str | None = None,
            soul_path: Path | None = None, evidence_path: Path | None = None,
            out_dir: Path | None = None, ablate_memory: bool = False) -> dict:
    sp = soul_path or (config.OUT_DIR / slug / "soul.md")
    soul_md = Path(sp).read_text()
    memory_text = "" if ablate_memory else load_memory(slug, t0, evidence_path=evidence_path).text
    system, user = build_query_prompt(soul_md, memory_text, slug, t0, query)
    raw = run_claude(system, user)
    view = _parse_view(raw, slug, t0)

    subdir = "views_ablation" if ablate_memory else "views"
    base = Path(out_dir or config.OUT_DIR) / slug / subdir
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{t0.isoformat()}.json").write_text(json.dumps(view, indent=2))
    return view
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -v`
Expected: PASS (new ablation test + all existing respond tests — the default `ablate_memory=False` keeps prior behavior).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/respond.py tests/test_doppelganger_respond.py
git commit -m "feat(doppelganger): respond ablation arm (soul-only, views_ablation/)"
```

---

## Task 2: `quarter_ends` (pure date logic)

**Files:** Create `doppelganger/walkforward.py`; Test `tests/test_doppelganger_walkforward.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_doppelganger_walkforward.py`:
```python
"""TDD tests for doppelganger.walkforward."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.walkforward import quarter_ends


def test_quarter_ends_inclusive_span():
    out = quarter_ends(date(2022, 12, 31), date(2023, 9, 30))
    assert out == [date(2022, 12, 31), date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30)]


def test_quarter_ends_skips_partial_quarters():
    out = quarter_ends(date(2023, 1, 15), date(2023, 7, 1))
    assert out == [date(2023, 3, 31), date(2023, 6, 30)]   # Dec-31-2022 excluded, Sep-30 excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_walkforward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.walkforward'`.

- [ ] **Step 3: Write minimal implementation** — `doppelganger/walkforward.py`:
```python
"""doppelganger.walkforward — drive respond() across a quarterly schedule.

Produces matched full + ablation views per step, audits each, and records a
resumable trajectory + coverage map. The raw material Unit 5b scores.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.respond import respond
from doppelganger.soul_audit import audit_answer

_QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def quarter_ends(start: date, end: date) -> list[date]:
    """Quarter-end dates (Mar 31 / Jun 30 / Sep 30 / Dec 31) within [start, end], inclusive."""
    out: list[date] = []
    for yr in range(start.year, end.year + 1):
        for m, d in _QUARTER_ENDS:
            qd = date(yr, m, d)
            if start <= qd <= end:
                out.append(qd)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_walkforward.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/walkforward.py tests/test_doppelganger_walkforward.py
git commit -m "feat(doppelganger): quarter_ends schedule generator"
```

---

## Task 3: `run_walkforward` (caching + audit + trajectory)

**Files:** Modify `doppelganger/walkforward.py`; Test `tests/test_doppelganger_walkforward.py`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from doppelganger.walkforward import run_walkforward


def _ev(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([{"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                   "source_type": "x_original", "text": "Tokens align incentives."}]).to_parquet(p)
    return p


def _canned_view():
    # all-extrapolated (no citations) -> audit finds 0 citations -> leaked 0, checked 0
    return {"as_of": "x", "subject": "s", "abstained": False,
            "sectors_excited": [{"name": "ZK", "provenance": "extrapolated", "citations": []}],
            "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
            "risk_regime": {"stance": "risk_on"}, "notes": ""}


def test_run_walkforward_builds_rows_for_both_variants(tmp_path):
    def fake_respond(slug, t0, *, ablate_memory=False, out_dir=None, **kw):
        sub = "views_ablation" if ablate_memory else "views"
        d = Path(out_dir) / slug / sub
        d.mkdir(parents=True, exist_ok=True)
        v = _canned_view()
        (d / f"{t0.isoformat()}.json").write_text(json.dumps(v))
        return v

    with patch("doppelganger.walkforward.respond", side_effect=fake_respond):
        rows = run_walkforward("s", [date(2022, 12, 31)], out_dir=tmp_path, evidence_path=_ev(tmp_path))
    # one full + one ablation row
    variants = sorted(r["variant"] for r in rows)
    assert variants == ["ablation", "full"]
    full = next(r for r in rows if r["variant"] == "full")
    assert full["risk"] == "risk_on" and full["n_sectors_excited"] == 1
    assert full["extrapolated"] == 1 and full["leaked"] == 0
    # walkforward.json written
    wf = json.loads((tmp_path / "s" / "walkforward.json").read_text())
    assert wf["subject"] == "s" and len(wf["rows"]) == 2


def test_run_walkforward_caches_existing_views(tmp_path):
    # pre-create the full view for this date -> respond must NOT be called for the full variant
    vd = tmp_path / "s" / "views"
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "2022-12-31.json").write_text(json.dumps(_canned_view()))

    m = MagicMock(side_effect=lambda slug, t0, *, ablate_memory=False, out_dir=None, **kw: (
        (Path(out_dir) / slug / "views_ablation").mkdir(parents=True, exist_ok=True),
        (Path(out_dir) / slug / "views_ablation" / f"{t0.isoformat()}.json").write_text(json.dumps(_canned_view())),
        _canned_view())[-1])
    with patch("doppelganger.walkforward.respond", m):
        rows = run_walkforward("s", [date(2022, 12, 31)], out_dir=tmp_path, evidence_path=_ev(tmp_path))
    # respond called ONLY for ablation (full was cached)
    assert m.call_count == 1
    assert m.call_args.kwargs["ablate_memory"] is True
    assert len(rows) == 2   # full (from cache) + ablation (generated)


def test_run_walkforward_no_ablate(tmp_path):
    def fake_respond(slug, t0, *, ablate_memory=False, out_dir=None, **kw):
        d = Path(out_dir) / slug / "views"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{t0.isoformat()}.json").write_text(json.dumps(_canned_view()))
        return _canned_view()
    with patch("doppelganger.walkforward.respond", side_effect=fake_respond):
        rows = run_walkforward("s", [date(2022, 12, 31)], ablate=False, out_dir=tmp_path, evidence_path=_ev(tmp_path))
    assert len(rows) == 1 and rows[0]["variant"] == "full"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_walkforward.py -k walkforward -v`
Expected: FAIL — `ImportError: cannot import name 'run_walkforward'`.

- [ ] **Step 3: Write minimal implementation** (append to `doppelganger/walkforward.py`)

```python
_ARRAYS = ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned"]


def _row(d: date, variant: str, view: dict, rep) -> dict:
    prov: Counter = Counter()
    for k in _ARRAYS:
        for it in view.get(k, []) or []:
            prov[it.get("provenance")] += 1
    return {
        "date": d.isoformat(), "variant": variant,
        "abstained": bool(view.get("abstained", False)),
        "risk": (view.get("risk_regime") or {}).get("stance"),
        "n_sectors_excited": len(view.get("sectors_excited", []) or []),
        "n_sectors_concerned": len(view.get("sectors_concerned", []) or []),
        "n_tokens_excited": len(view.get("tokens_excited", []) or []),
        "n_tokens_concerned": len(view.get("tokens_concerned", []) or []),
        "grounded": prov.get("grounded", 0),
        "persisted": prov.get("persisted", 0),
        "extrapolated": prov.get("extrapolated", 0),
        "leaked": len(rep.leaked), "hallucinated": len(rep.hallucinated),
        "matched": rep.matched, "checked": rep.checked,
    }


def run_walkforward(slug: str, dates: list[date], *, ablate: bool = True,
                    out_dir: Path | None = None, evidence_path: Path | None = None,
                    soul_path: Path | None = None) -> list[dict]:
    base_dir = Path(out_dir or config.OUT_DIR)
    ev_path = evidence_path or (base_dir / slug / "evidence.parquet")
    variants = [("full", False, "views")] + ([("ablation", True, "views_ablation")] if ablate else [])

    rows: list[dict] = []
    for d in dates:
        for variant, ablate_mem, subdir in variants:
            vpath = base_dir / slug / subdir / f"{d.isoformat()}.json"
            if vpath.exists():
                view = json.loads(vpath.read_text())          # cached — no claude -p
            else:
                view = respond(slug, d, ablate_memory=ablate_mem, out_dir=out_dir,
                               evidence_path=evidence_path, soul_path=soul_path)
            rep = audit_answer(view, ev_path, d)
            rows.append(_row(d, variant, view, rep))

    (base_dir / slug).mkdir(parents=True, exist_ok=True)
    (base_dir / slug / "walkforward.json").write_text(
        json.dumps({"subject": slug, "dates": [d.isoformat() for d in dates], "rows": rows}, indent=2))
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_walkforward.py -v`
Expected: PASS (all walkforward tests). Then full suite `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q` → all PASS.

- [ ] **Step 5: Commit**
```bash
git add doppelganger/walkforward.py tests/test_doppelganger_walkforward.py
git commit -m "feat(doppelganger): run_walkforward — cached trajectory + per-step audit"
```

---

## Task 4: CLI `walkforward` subcommand

**Files:** Modify `doppelganger/run.py`; Test `tests/test_doppelganger_walkforward.py`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_run_cli_has_walkforward_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["walkforward", "--subject", "eddy-lazzarin"])
    assert ns.cmd == "walkforward" and ns.subject == "eddy-lazzarin"
    assert ns.start == "2022-12-31" and ns.end is None and ns.no_ablate is False
    ns2 = r.build_parser().parse_args(["walkforward", "--subject", "x", "--start", "2023-01-01",
                                       "--end", "2023-12-31", "--no-ablate"])
    assert ns2.start == "2023-01-01" and ns2.end == "2023-12-31" and ns2.no_ablate is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_walkforward.py -k cli -v`
Expected: FAIL — argparse rejects `walkforward`.

- [ ] **Step 3: Modify `doppelganger/run.py`**:
  (i) Add imports at the top: `import pandas as pd` and `from doppelganger.walkforward import quarter_ends, run_walkforward`.
  (ii) In `build_parser()`, after the `respond` subparser and before `return parser`, add:
```python
    wf = sub.add_parser("walkforward", help="run respond() across a quarterly schedule (full + ablation)")
    wf.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    wf.add_argument("--start", default="2022-12-31", help="schedule start YYYY-MM-DD")
    wf.add_argument("--end", default=None, help="schedule end YYYY-MM-DD (default: latest evidence date)")
    wf.add_argument("--no-ablate", action="store_true", help="skip the soul-only ablation arm")
```
  (iii) In `main()`, after the `respond` branch, add:
```python
    elif args.cmd == "walkforward":
        start = date.fromisoformat(args.start)
        if args.end:
            end = date.fromisoformat(args.end)
        else:
            ev = pd.read_parquet(config.OUT_DIR / args.subject / "evidence.parquet")
            end = pd.to_datetime(ev["timestamp"], utc=True).max().date()
        dates = quarter_ends(start, end)
        rows = run_walkforward(args.subject, dates, ablate=not args.no_ablate)
        print(f"{len(rows)} rows over {len(dates)} dates -> {config.OUT_DIR / args.subject / 'walkforward.json'}")
```
  (iv) Ensure `from doppelganger import config` and `from datetime import date` are imported in run.py (they are, via the existing soul/memory handlers — confirm; if `config` isn't imported, add `from doppelganger import config`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_walkforward.py -v`
Expected: PASS. Then full suite: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q` → all PASS.

- [ ] **Step 5: Commit**
```bash
git add doppelganger/run.py tests/test_doppelganger_walkforward.py
git commit -m "feat(doppelganger): walkforward CLI subcommand"
```

---

## Task 5: Real-data subset validation (Eddy, 3 steps) — not TDD

Requires the `claude` CLI on the Max subscription and Eddy's `soul.md` + `evidence.parquet` present. Reuses the existing `views/2022-12-31.json` from Unit 4 (cache); generates 2 more full views + 3 ablation views via real `claude -p` (~5 calls × ~4 min — note the wall-clock).

- [ ] **Step 1: Confirm full suite green**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q`
Expected: all PASS.

- [ ] **Step 2: Run the 3-step subset walk-forward for Eddy**

Run:
```bash
.venv/bin/python - <<'PY'
from datetime import date
from doppelganger.walkforward import run_walkforward, quarter_ends
dates = quarter_ends(date(2022, 12, 31), date(2023, 6, 30))   # [2022-12-31, 2023-03-31, 2023-06-30]
print("dates:", [d.isoformat() for d in dates])
rows = run_walkforward("eddy-lazzarin", dates)
import json
print(json.dumps(rows, indent=2))
PY
```
Expected: 6 rows (3 dates × {full, ablation}). The 2022-12-31 full row comes from cache; the rest are fresh `claude -p`.

- [ ] **Step 3: Check the gate + read the trajectory**

From the printed rows, confirm: **every row has `leaked == 0`** (the hard gate). Then eyeball the coverage contrast: across the dates, the **full** rows should be richer and more `grounded`; the **ablation** rows should be thinner and more `extrapolated` (no memory to ground in). That contrast is the first real read on whether the corpus is load-bearing — a strong signal in favor means model-leakage is bounded. Write a 3-5 sentence read of: (a) are full views more grounded than ablation? (b) does the risk stance / sector count evolve sensibly across the 3 quarters? Report `walkforward.json` location.

- [ ] **Step 4: Commit any fixes**

If the subset run revealed a bug (e.g. a row field wrong on real data), fix it, re-run the affected test, and:
```bash
git add -A && git commit -m "fix(doppelganger): walkforward adjustment per subset run"
```
Otherwise report the trajectory read to the controller. `views*/` and `walkforward.json` are NOT committed (regenerate via CLI), consistent with prior units.

---

## Self-review

**Spec coverage (walkforward-unit-design.md):**
- §2 respond ablation arm (empty memory, views_ablation/) → Task 1. ✓
- §3 `quarter_ends` → Task 2. ✓
- §3 `run_walkforward` (cached full+ablation, per-step audit, trajectory rows, walkforward.json) → Task 3. ✓
- §3 caching/resumability (skip existing view json) → Task 3 (`test_run_walkforward_caches_existing_views`). ✓
- §3 per-step audit + leaked flag, coverage counts in rows → Task 3 (`_row`). ✓
- §4 `walkforward` CLI (--subject/--start/--end/--no-ablate, end defaults to latest evidence date) → Task 4. ✓
- §5 subset validation (3 steps, reuse cache, leaked==0, coverage contrast) → Task 5. ✓

**Placeholder scan:** none — every code step has complete code; the only real-LLM step is Task 5, explicitly non-TDD.

**Type consistency:** `respond(..., ablate_memory=False)` (Task 1) matches the `run_walkforward` call (Task 3) and the CLI. `run_walkforward(slug, dates, *, ablate, out_dir, evidence_path, soul_path)` and `quarter_ends(start, end)` consistent across Tasks 2-5 + CLI. `_row` field names match the test assertions. `audit_answer(view, evidence_path, t0)` used as defined in Unit 4.

---

## Execution handoff

After 5a: Unit ❺b (scorers + findings) — held-out prediction judge (corpus-lift over ablation, change-detection), scaled discrimination, coverage analysis — consumes `walkforward.json` + the `views*/` trajectories. Then the full quarterly walk-forward run (both subjects) as a separate long execution.
