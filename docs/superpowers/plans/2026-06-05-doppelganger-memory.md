# Corpus Doppelganger — Unit ❸ Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A time-gated, leakage-firewalled, chronological feed of a subject's ≤T evidence for the doppelganger to reason over — feed-all, no retrieval, with the RAG seam pre-cut.

**Architecture:** One small module `doppelganger/memory.py`: a `MemoryView` dataclass and `load_memory(slug, t0)`. The leakage firewall (`timestamp ≤ t0` applied first; `max_date` exposed for assertion) is the unit's core. A `query` param is accepted but ignored — the seam where retrieval lands later.

**Tech Stack:** Python 3.13, pandas/pyarrow. pytest. Builds on Units 1-2 (`doppelganger/config.py`, `data/doppelganger/<slug>/evidence.parquet`).

**Spec:** `docs/superpowers/specs/2026-06-05-doppelganger-memory-unit-design.md`.

---

## File structure

```
doppelganger/
  memory.py        # MemoryView, _format, load_memory
  run.py           # add `memory` subcommand (MODIFY)
tests/
  test_doppelganger_memory.py
```

---

## Task 1: `memory.py` — firewall, formatting, seam

**Files:**
- Create: `doppelganger/memory.py`
- Test: `tests/test_doppelganger_memory.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_doppelganger_memory.py`:
```python
"""TDD tests for doppelganger.memory."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger.memory import load_memory, MemoryView


def _ev(tmp_path) -> Path:
    p = tmp_path / "evidence.parquet"
    pd.DataFrame([
        {"id": "a", "timestamp": pd.Timestamp("2022-07-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives.", "context": None},
        {"id": "b", "timestamp": pd.Timestamp("2021-01-01", tz="UTC"),
         "source_type": "podcast", "text": "Points are a balance.", "context": "What about points?"},
        {"id": "c", "timestamp": pd.Timestamp("2023-03-01", tz="UTC"),
         "source_type": "x_original", "text": "Future take after t0.", "context": None},
    ]).to_parquet(p)
    return p


def test_firewall_excludes_future_and_sets_max_date(tmp_path):
    mv = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    assert isinstance(mv, MemoryView)
    assert set(mv.items["id"]) == {"a", "b"}          # 2023 item "c" excluded
    assert (mv.items["timestamp"].dt.date <= date(2022, 12, 31)).all()
    assert mv.max_date == date(2022, 7, 1)            # latest <= t0
    assert mv.n_items == 2


def test_items_sorted_chronologically(tmp_path):
    mv = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    assert list(mv.items["id"]) == ["b", "a"]         # 2021 before 2022
    assert mv.text.index("Points are a balance") < mv.text.index("Tokens align incentives")


def test_formatting_includes_date_type_text_context(tmp_path):
    mv = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    assert "[2021-01-01] (podcast)" in mv.text
    assert "(context: What about points?)" in mv.text
    assert "Tokens align incentives." in mv.text
    assert "[2022-07-01] (x_original)" in mv.text


def test_empty_when_t0_before_all(tmp_path):
    mv = load_memory("x", date(2018, 1, 1), evidence_path=_ev(tmp_path))
    assert mv.n_items == 0 and mv.max_date is None and mv.text == ""


def test_query_is_ignored_seam(tmp_path):
    full = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    queried = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path), query="rollups")
    assert list(queried.items["id"]) == list(full.items["id"])   # query ignored in v1
    assert queried.text == full.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.memory'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/memory.py`:
```python
"""doppelganger.memory — time-gated view of a subject's evidence for the doppelganger.

Feed-all: the full <=T corpus fits in context, so v1 does NO retrieval — it filters
to <= t0 behind a leakage firewall and hands over everything chronologically. The
`query` parameter is accepted but ignored: the seam where retrieval drops in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config


@dataclass
class MemoryView:
    items: pd.DataFrame       # the <= t0 evidence, sorted chronologically
    text: str                 # formatted block the doppelganger reads
    n_items: int
    max_date: date | None     # leakage guarantee: must be <= t0 (None if empty)


def _format(items: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, r in items.iterrows():
        d = pd.Timestamp(r["timestamp"]).date().isoformat()
        ctx = r.get("context")
        ctx_s = f" (context: {ctx})" if isinstance(ctx, str) and ctx else ""
        lines.append(f"[{d}] ({r['source_type']}){ctx_s} {r['text']}")
    return "\n".join(lines)


def load_memory(
    slug: str, t0: date, *,
    evidence_path: Path | None = None,
    query: str | None = None,    # accepted, IGNORED in v1 — the retrieval seam
) -> MemoryView:
    path = evidence_path or (config.OUT_DIR / slug / "evidence.parquet")
    ev = pd.read_parquet(path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    # FIREWALL: <= t0, applied first, before anything else.
    ev = ev[ev["timestamp"].dt.date <= t0].sort_values("timestamp").reset_index(drop=True)
    max_date = pd.Timestamp(ev["timestamp"].max()).date() if len(ev) else None
    return MemoryView(items=ev, text=_format(ev), n_items=len(ev), max_date=max_date)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_memory.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/memory.py tests/test_doppelganger_memory.py
git commit -m "feat(doppelganger): memory — time-gated feed-all view with leakage firewall"
```

---

## Task 2: CLI `memory` subcommand

**Files:**
- Modify: `doppelganger/run.py`
- Test: `tests/test_doppelganger_memory.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_run_cli_has_memory_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["memory", "--subject", "eddy-lazzarin", "--t0", "2022-12-31"])
    assert ns.cmd == "memory" and ns.subject == "eddy-lazzarin" and ns.t0 == "2022-12-31"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_memory.py -k cli -v`
Expected: FAIL — `argparse` error / `AttributeError` (no `memory` subparser).

- [ ] **Step 3: Modify `doppelganger/run.py`** — replace its entire contents with (adds the `memory` subcommand + handler; keeps `ingest` and `soul`):
```python
"""doppelganger.run — CLI entrypoint.

Usage:
    python -m doppelganger.run ingest --subject eddy-lazzarin
    python -m doppelganger.run soul   --subject eddy-lazzarin --t0 2022-12-31
    python -m doppelganger.run memory --subject eddy-lazzarin --t0 2022-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

from doppelganger.ingest import ingest
from doppelganger.memory import load_memory
from doppelganger.soul import extract_soul


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doppelganger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="build identity + evidence stream for a subject")
    ing.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")

    soul = sub.add_parser("soul", help="build the frozen-at-T0 soul card for a subject")
    soul.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    soul.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")

    mem = sub.add_parser("memory", help="inspect the time-gated <=T memory view for a subject")
    mem.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    mem.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "ingest":
        out = ingest(args.subject)
        print(f"wrote {out['evidence']} and {out['identity']}")
    elif args.cmd == "soul":
        path = extract_soul(args.subject, date.fromisoformat(args.t0))
        print(f"wrote {path}")
    elif args.cmd == "memory":
        mv = load_memory(args.subject, date.fromisoformat(args.t0))
        print(f"n_items={mv.n_items} max_date={mv.max_date}")
        print(mv.text[:1000])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_memory.py -v`
Expected: PASS (6 tests). Then full suite: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q` → all PASS.

- [ ] **Step 5: Commit**
```bash
git add doppelganger/run.py tests/test_doppelganger_memory.py
git commit -m "feat(doppelganger): memory CLI subcommand"
```

---

## Task 3: Real-data sanity check (Eddy) — not TDD

- [ ] **Step 1: Confirm full suite green**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q`
Expected: all PASS.

- [ ] **Step 2: Load Eddy's ≤T0 memory and eyeball**

Run:
```bash
.venv/bin/python - <<'PY'
from datetime import date
from doppelganger.memory import load_memory
mv = load_memory("eddy-lazzarin", date(2022, 12, 31))
print("n_items:", mv.n_items, "| max_date:", mv.max_date)
assert mv.max_date <= date(2022, 12, 31), "LEAKAGE: max_date past t0"
print("first 600 chars of formatted view:\n", mv.text[:600])
PY
```
Expected: `n_items` ≈ 329, `max_date` ≤ 2022-12-31 (the assert must hold — that's the firewall proven on real data), formatted lines starting `[YYYY-MM-DD] (source_type) ...` in chronological order. Report the numbers.

---

## Self-review

**Spec coverage:**
- §2 leakage firewall (≤t0 first, `max_date` exposed) → Task 1 (`test_firewall_*`), Task 3 (real-data assert). ✓
- §3 `MemoryView` shape + `load_memory(slug, t0, *, evidence_path, query)` → Task 1. ✓
- §3 feed-all, `query` accepted/ignored seam → Task 1 (`test_query_is_ignored_seam`). ✓
- §3 chronological `[date] (source_type) text` + context formatting → Task 1 (`test_items_sorted_*`, `test_formatting_*`). ✓
- §3 reads `evidence.parquet` via `config.OUT_DIR` unless `evidence_path` → Task 1 impl. ✓
- §4 `memory` CLI subcommand → Task 2. ✓
- §5 empty/edge → Task 1 (`test_empty_when_t0_before_all`). ✓

**Placeholder scan:** none — all steps have runnable code/commands.

**Type consistency:** `MemoryView(items, text, n_items, max_date)` and `load_memory(slug, t0, *, evidence_path, query)` identical across Tasks 1-3 and the CLI handler. `build_parser`/`main` match the Unit 2 CLI structure (extends, doesn't break `ingest`/`soul`).

---

## Execution handoff

After this unit: Unit ❹ (Doppelganger) consumes `soul.md` + `MemoryView` to answer time-gated market-view queries; then Unit ❺ (Eval) runs the walk-forward.
