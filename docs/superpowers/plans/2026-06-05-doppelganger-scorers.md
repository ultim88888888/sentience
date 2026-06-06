# Corpus Doppelganger — Unit ❺b Scorers + Findings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the walk-forward trajectory into honest metrics — a held-out prediction judge scoring each step's view against the subject's real later statements, headlined by **corpus-lift over a clean soul-less baseline**, plus discrimination + coverage, into a `metrics.json` and a templated `findings.md`.

**Architecture:** Fix the ablation arm to soul-less (identity stub, no soul/no memory). Add `judge.py` (the one `claude -p` LLM-judge, mockable) and `score.py` (deterministic discrimination/coverage + the orchestrator + memo template). Judge verdicts cache to disk (resumable).

**Tech Stack:** Python 3.13, pandas, `claude` CLI. pytest. Builds on Units 1-5a.

**Spec:** `docs/superpowers/specs/2026-06-05-doppelganger-scorers-unit-design.md`.

---

## File structure

```
doppelganger/
  respond.py     # MODIFY: ablation arm -> soul-less (identity stub)
  judge.py       # NEW: post_t_evidence, judge_step (claude -p)
  score.py       # NEW: discrimination, coverage_trajectory, score_subject, write_memo
  run.py         # MODIFY: add `score` subcommand
tests/
  test_doppelganger_judge.py
  test_doppelganger_score.py
```

**Decision:** keep the `respond(..., ablate_memory=...)` param name (avoids rippling into walkforward + its tests); only its *behavior* changes to soul-less. Document it as "ablation arm = soul-less."

---

## Task 1: Soul-less ablation (`respond` behavior change)

**Files:** Modify `doppelganger/respond.py`; Test `tests/test_doppelganger_respond.py`.

- [ ] **Step 1: Replace the existing ablation test** in `tests/test_doppelganger_respond.py`. Find `test_respond_ablate_memory_uses_empty_memory_and_separate_dir` and REPLACE it entirely with:
```python
def test_respond_ablate_memory_is_soulless(tmp_path):
    # identity.json provides the stub; soul.md must NOT be read in ablation
    (tmp_path / "testy-mctest").mkdir(parents=True, exist_ok=True)
    (tmp_path / "testy-mctest" / "identity.json").write_text(
        '{"name": "Testy McTest", "headline": "Investing in tokens."}')
    captured = {}

    def fake_run(system, user):
        captured["system"] = system
        captured["user"] = user
        return '{"sectors_excited":[{"name":"X","provenance":"extrapolated","citations":[]}]}'

    with patch("doppelganger.respond.run_claude", side_effect=fake_run):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path, ablate_memory=True)
    # written to the ablation dir
    assert (tmp_path / "testy-mctest" / "views_ablation" / "2022-12-31.json").exists()
    # the stub (name + headline) is present; the soul prose is NOT
    assert "Testy McTest" in captured["system"] and "Investing in tokens." in captured["system"]
    assert "How He Thinks" not in captured["system"]      # no soul card
    # empty memory record
    between = captured["user"].split("YOUR RECORD", 1)[1].split("# QUESTION", 1)[0]
    assert between.strip().splitlines()[1:] == []
    assert view["sectors_excited"][0]["provenance"] == "extrapolated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -k soulless -v`
Expected: FAIL (the current ablation keeps the soul; assertion `"How He Thinks" not in system` or the identity-read fails).

- [ ] **Step 3: Replace the `respond` function** in `doppelganger/respond.py` with:
```python
def respond(slug: str, t0: date, *, query: str | None = None,
            soul_path: Path | None = None, evidence_path: Path | None = None,
            out_dir: Path | None = None, ablate_memory: bool = False) -> dict:
    base = Path(out_dir or config.OUT_DIR)
    if ablate_memory:
        # Ablation arm = SOUL-LESS: no soul card, no memory — just an identity stub.
        # The clean parametric floor for the corpus-lift metric.
        ident = json.loads((base / slug / "identity.json").read_text())
        soul_md = f"# {ident.get('name') or slug}\n{ident.get('headline') or ident.get('bio') or ''}"
        memory_text = ""
    else:
        sp = soul_path or (base / slug / "soul.md")
        soul_md = Path(sp).read_text()
        memory_text = load_memory(slug, t0, evidence_path=evidence_path).text

    system, user = build_query_prompt(soul_md, memory_text, slug, t0, query)
    raw = run_claude(system, user)
    view = _parse_view(raw, slug, t0)

    subdir = "views_ablation" if ablate_memory else "views"
    d = base / slug / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{t0.isoformat()}.json").write_text(json.dumps(view, indent=2))
    return view
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -v`
Expected: PASS (the new soul-less test + all other respond tests — `ablate_memory=False` path unchanged). Then `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q` → all PASS (walkforward tests mock `respond`, so they're unaffected).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/respond.py tests/test_doppelganger_respond.py
git commit -m "feat(doppelganger): soul-less ablation arm (clean parametric floor)"
```

---

## Task 2: `post_t_evidence` (held-out future slice)

**Files:** Create `doppelganger/judge.py`; Test `tests/test_doppelganger_judge.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_doppelganger_judge.py`:
```python
"""TDD tests for doppelganger.judge."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger.judge import post_t_evidence


def _ev(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([
        {"id": "a", "timestamp": pd.Timestamp("2022-11-01", tz="UTC"), "source_type": "x_original",
         "text": "before T", "context": None},
        {"id": "b", "timestamp": pd.Timestamp("2023-02-01", tz="UTC"), "source_type": "x_original",
         "text": "inside window", "context": None},
        {"id": "c", "timestamp": pd.Timestamp("2023-09-01", tz="UTC"), "source_type": "x_original",
         "text": "after window", "context": None},
    ]).to_parquet(p)
    return p


def test_post_t_evidence_slices_window(tmp_path):
    out = post_t_evidence("s", date(2022, 12, 31), horizon_months=6, evidence_path=_ev(tmp_path))
    assert "inside window" in out          # 2023-02-01 in (2022-12-31, 2023-06-30]
    assert "before T" not in out           # 2022-11-01 excluded
    assert "after window" not in out       # 2023-09-01 excluded
    assert "[2023-02-01]" in out


def test_post_t_evidence_empty(tmp_path):
    out = post_t_evidence("s", date(2024, 1, 1), horizon_months=6, evidence_path=_ev(tmp_path))
    assert out == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.judge'`.

- [ ] **Step 3: Write minimal implementation** — `doppelganger/judge.py`:
```python
"""doppelganger.judge — held-out prediction judge.

Compares a view@T against what the subject ACTUALLY said in (T, T+horizon],
scoring each claim confirmed/contradicted/absent. One claude -p call, cached.
"""

from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.llm import run_claude


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def post_t_evidence(slug: str, t0: date, horizon_months: int = 6, *,
                    evidence_path: Path | None = None) -> str:
    path = evidence_path or (config.OUT_DIR / slug / "evidence.parquet")
    ev = pd.read_parquet(path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    end = _add_months(t0, horizon_months)
    d = ev["timestamp"].dt.date
    win = ev[(d > t0) & (d <= end)].sort_values("timestamp")
    return "\n".join(
        f"[{pd.Timestamp(r['timestamp']).date().isoformat()}] ({r['source_type']}) {r['text']}"
        for _, r in win.iterrows()
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_judge.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/judge.py tests/test_doppelganger_judge.py
git commit -m "feat(doppelganger): post_t_evidence held-out window slice"
```

---

## Task 3: `judge_step` (the LLM-judge, cached)

**Files:** Modify `doppelganger/judge.py`; Test `tests/test_doppelganger_judge.py`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import json
from unittest.mock import patch, MagicMock
from doppelganger.judge import judge_step

_VIEW = {"sectors_excited": [{"name": "ZK", "why": "w", "provenance": "grounded"}],
         "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
         "risk_regime": {"stance": "risk_on"}}

_VERDICT = ('{"claims":[{"claim":"excited about ZK","axis":"sectors_excited","label":"confirmed"},'
            '{"claim":"risk on","axis":"risk_regime","label":"contradicted"}],'
            '"n_confirmed":1,"n_contradicted":1,"n_absent":0,'
            '"missed_changes":["picked up restaking"],"notes":"ok"}')


def test_judge_step_parses_and_computes_confirm_rate(tmp_path):
    with patch("doppelganger.judge.run_claude", return_value=f"```json\n{_VERDICT}\n```"):
        v = judge_step(_VIEW, "he kept tweeting about ZK", "Eddy", date(2022, 12, 31),
                       judge_path=tmp_path / "j.json")
    assert v["n_confirmed"] == 1 and v["n_contradicted"] == 1
    assert v["confirm_rate"] == 0.5                       # 1 / (1+1)
    assert v["missed_changes"] == ["picked up restaking"]
    assert (tmp_path / "j.json").exists()                 # cached


def test_judge_step_confirm_rate_none_when_no_scored(tmp_path):
    verdict = '{"claims":[],"n_confirmed":0,"n_contradicted":0,"n_absent":3,"missed_changes":[]}'
    with patch("doppelganger.judge.run_claude", return_value=verdict):
        v = judge_step(_VIEW, "unrelated", "Eddy", date(2022, 12, 31), judge_path=tmp_path / "j.json")
    assert v["confirm_rate"] is None                      # no confirmed+contradicted -> undefined


def test_judge_step_uses_cache(tmp_path):
    jp = tmp_path / "j.json"
    jp.write_text(json.dumps({"n_confirmed": 9, "n_contradicted": 0, "confirm_rate": 1.0,
                              "missed_changes": [], "claims": []}))
    m = MagicMock()
    with patch("doppelganger.judge.run_claude", m):
        v = judge_step(_VIEW, "x", "Eddy", date(2022, 12, 31), judge_path=jp)
    m.assert_not_called()                                 # cached -> no claude -p
    assert v["n_confirmed"] == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_judge.py -k judge_step -v`
Expected: FAIL — `ImportError: cannot import name 'judge_step'`.

- [ ] **Step 3: Write minimal implementation** (append to `doppelganger/judge.py`)

```python
_JUDGE_INSTRUCTIONS = """You are evaluating a market-view PREDICTION against what a person \
ACTUALLY said afterward. You are given (1) the person's predicted market view as of a date, and \
(2) their REAL statements in the months that followed.

For each distinct claim in the prediction (each sector/token they were excited or concerned about, \
and their risk stance), label it against ONLY the provided later statements:
- "confirmed": they expressed or acted on this view in the window.
- "contradicted": they expressed the opposite.
- "absent": they did not address it in the window.

Judge at the level of STANCE, not wording. Use ONLY the provided later statements — do NOT use \
anything you know about what happened after the window. Also list "missed_changes": stances the \
person NEWLY took or REVERSED in the window that the prediction did not anticipate.

Output ONLY a JSON object:
{"claims":[{"claim":"...","axis":"sectors_excited|sectors_concerned|tokens_excited|tokens_concerned|risk_regime","label":"confirmed|contradicted|absent"}],
 "n_confirmed":<int>,"n_contradicted":<int>,"n_absent":<int>,"missed_changes":["..."],"notes":"..."}"""


def _parse_json(raw: str) -> dict:
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"no JSON object in judge output: {raw[:200]!r}")
    return json.loads(raw[i:j + 1])


def judge_step(view: dict, post_t_text: str, subject_name: str, t0: date, *,
               judge_path: Path | None = None) -> dict:
    if judge_path is not None and Path(judge_path).exists():
        return json.loads(Path(judge_path).read_text())          # cached

    user = (f"# {subject_name}'s PREDICTED market view as of {t0.isoformat()}\n\n"
            f"{json.dumps(view, indent=2)}\n\n"
            f"# What {subject_name} ACTUALLY said afterward\n\n"
            f"{post_t_text or '(no statements in this window)'}")
    raw = run_claude(_JUDGE_INSTRUCTIONS, user)
    v = _parse_json(raw)

    c, k = int(v.get("n_confirmed", 0)), int(v.get("n_contradicted", 0))
    v["confirm_rate"] = (c / (c + k)) if (c + k) > 0 else None
    v.setdefault("missed_changes", [])
    v.setdefault("claims", [])

    if judge_path is not None:
        Path(judge_path).parent.mkdir(parents=True, exist_ok=True)
        Path(judge_path).write_text(json.dumps(v, indent=2))
    return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_judge.py -v`
Expected: PASS (all judge tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/judge.py tests/test_doppelganger_judge.py
git commit -m "feat(doppelganger): judge_step — held-out prediction verdict (cached)"
```

---

## Task 4: Deterministic scorers (`discrimination`, `coverage_trajectory`)

**Files:** Create `doppelganger/score.py`; Test `tests/test_doppelganger_score.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_doppelganger_score.py`:
```python
"""TDD tests for doppelganger.score."""
from __future__ import annotations

from doppelganger.score import discrimination, coverage_trajectory


def test_discrimination_overlap():
    a = {"sectors_excited": [{"name": "ZK"}, {"name": "Games"}], "sectors_concerned": [],
         "tokens_excited": [{"name": "ETH"}], "tokens_concerned": []}
    b = {"sectors_excited": [{"name": "zk"}, {"name": "DAOs"}], "sectors_concerned": [],
         "tokens_excited": [{"name": "SOL"}], "tokens_concerned": []}
    d = discrimination(a, b)
    # sectors: {zk,games} vs {zk,daos} -> intersection {zk}=1, union 3 -> 1/3
    assert round(d["sector_overlap"], 2) == 0.33
    assert d["token_overlap"] == 0.0          # eth vs sol, no overlap
    assert "zk" in d["shared_sectors"]


def test_coverage_trajectory():
    rows = [
        {"date": "2022-12-31", "variant": "full", "grounded": 13, "persisted": 0, "extrapolated": 0},
        {"date": "2022-12-31", "variant": "ablation", "grounded": 5, "persisted": 2, "extrapolated": 2},
        {"date": "2023-03-31", "variant": "full", "grounded": 11, "persisted": 0, "extrapolated": 0},
    ]
    cov = coverage_trajectory(rows)
    assert cov == [{"date": "2022-12-31", "grounded": 13, "persisted": 0, "extrapolated": 0},
                   {"date": "2023-03-31", "grounded": 11, "persisted": 0, "extrapolated": 0}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.score'`.

- [ ] **Step 3: Write minimal implementation** — `doppelganger/score.py`:
```python
"""doppelganger.score — deterministic scorers + the held-out scoring orchestrator + memo."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from doppelganger import config

_NAMED = ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned"]


def _names(view: dict, keys: list[str]) -> set[str]:
    out: set[str] = set()
    for k in keys:
        for it in view.get(k, []) or []:
            n = (it.get("name") or "").strip().lower()
            if n:
                out.add(n)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def discrimination(view_a: dict, view_b: dict) -> dict:
    sa, sb = _names(view_a, ["sectors_excited", "sectors_concerned"]), _names(view_b, ["sectors_excited", "sectors_concerned"])
    ta, tb = _names(view_a, ["tokens_excited", "tokens_concerned"]), _names(view_b, ["tokens_excited", "tokens_concerned"])
    return {
        "sector_overlap": _jaccard(sa, sb), "token_overlap": _jaccard(ta, tb),
        "shared_sectors": sorted(sa & sb), "shared_tokens": sorted(ta & tb),
    }


def coverage_trajectory(rows: list[dict]) -> list[dict]:
    return [{"date": r["date"], "grounded": r["grounded"], "persisted": r["persisted"],
             "extrapolated": r["extrapolated"]}
            for r in rows if r.get("variant") == "full"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/score.py tests/test_doppelganger_score.py
git commit -m "feat(doppelganger): deterministic discrimination + coverage scorers"
```

---

## Task 5: `score_subject` (orchestrator)

**Files:** Modify `doppelganger/score.py`; Test `tests/test_doppelganger_score.py`.

- [ ] **Step 1: Write the failing test** (append)

```python
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch
from doppelganger.score import score_subject


def _setup(tmp_path):
    slug = "s"
    base = tmp_path / slug
    (base / "views").mkdir(parents=True, exist_ok=True)
    (base / "views_ablation").mkdir(parents=True, exist_ok=True)
    view = {"sectors_excited": [{"name": "ZK", "provenance": "grounded"}], "sectors_concerned": [],
            "tokens_excited": [], "tokens_concerned": [], "risk_regime": {"stance": "risk_on"}}
    for d in ["2022-12-31", "2023-03-31"]:
        (base / "views" / f"{d}.json").write_text(json.dumps(view))
        (base / "views_ablation" / f"{d}.json").write_text(json.dumps(view))
    (base / "walkforward.json").write_text(json.dumps({"subject": slug,
        "dates": ["2022-12-31", "2023-03-31"],
        "rows": [{"date": "2022-12-31", "variant": "full", "grounded": 1, "persisted": 0, "extrapolated": 0}]}))
    # evidence with a post-T window for both dates
    import pandas as pd
    pd.DataFrame([{"id": "x", "timestamp": pd.Timestamp("2023-05-01", tz="UTC"),
                   "source_type": "x_original", "text": "still ZK", "context": None}]).to_parquet(base / "evidence.parquet")
    return slug


def test_score_subject_computes_lift(tmp_path):
    slug = _setup(tmp_path)
    # full judges high, ablation low
    def fake_judge(view, post_t, name, t0, *, judge_path=None):
        rate = 0.9 if "views_ablation" not in str(judge_path) else 0.4
        return {"confirm_rate": rate, "n_confirmed": 1, "n_contradicted": 0,
                "missed_changes": ["m"] if rate == 0.9 else [], "claims": []}
    with patch("doppelganger.score.judge_step", side_effect=fake_judge):
        m = score_subject(slug, out_dir=tmp_path, evidence_path=tmp_path / slug / "evidence.parquet")
    assert m["subject"] == slug
    # lift = 0.9 - 0.4 = 0.5 each step
    assert round(m["mean_lift"], 2) == 0.5
    assert len(m["steps"]) == 2
    assert (tmp_path / slug / "metrics.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -k score_subject -v`
Expected: FAIL — `ImportError: cannot import name 'score_subject'`.

- [ ] **Step 3: Write minimal implementation** (append to `doppelganger/score.py`; add imports `from doppelganger.judge import judge_step, post_t_evidence`)

```python
def score_subject(slug: str, *, horizon_months: int = 6, out_dir: Path | None = None,
                  evidence_path: Path | None = None) -> dict:
    base = Path(out_dir or config.OUT_DIR) / slug
    wf = json.loads((base / "walkforward.json").read_text())
    ev_path = evidence_path or (base / "evidence.parquet")

    steps = []
    for ds in wf["dates"]:
        t0 = date.fromisoformat(ds)
        post = post_t_evidence(slug, t0, horizon_months, evidence_path=ev_path)
        if not post:
            continue                                   # no held-out future -> unscorable
        row = {"date": ds, "full_confirm_rate": None, "ablation_confirm_rate": None,
               "lift": None, "missed_changes": [], "n_missed_changes": 0}
        for variant, sub in [("full", "views"), ("ablation", "views_ablation")]:
            vpath = base / sub / f"{ds}.json"
            if not vpath.exists():
                continue
            view = json.loads(vpath.read_text())
            jp = base / "judge" / f"{ds}_{variant}.json"
            v = judge_step(view, post, wf["subject"], t0, judge_path=jp)
            row[f"{variant}_confirm_rate"] = v.get("confirm_rate")
            if variant == "full":
                row["missed_changes"] = v.get("missed_changes", [])
                row["n_missed_changes"] = len(v.get("missed_changes", []))
        if row["full_confirm_rate"] is not None and row["ablation_confirm_rate"] is not None:
            row["lift"] = row["full_confirm_rate"] - row["ablation_confirm_rate"]
        steps.append(row)

    lifts = [s["lift"] for s in steps if s["lift"] is not None]
    metrics = {
        "subject": slug, "horizon_months": horizon_months,
        "mean_lift": (sum(lifts) / len(lifts)) if lifts else None,
        "steps": steps, "coverage": coverage_trajectory(wf["rows"]),
    }
    (base / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -v`
Expected: PASS (all score tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/score.py tests/test_doppelganger_score.py
git commit -m "feat(doppelganger): score_subject — corpus-lift orchestrator over the trajectory"
```

---

## Task 6: `write_memo` (findings template)

**Files:** Modify `doppelganger/score.py`; Test `tests/test_doppelganger_score.py`.

- [ ] **Step 1: Write the failing test** (append)

```python
from doppelganger.score import write_memo


def test_write_memo_renders(tmp_path):
    metrics = {"subject": "eddy-lazzarin", "horizon_months": 6, "mean_lift": 0.25,
               "steps": [{"date": "2022-12-31", "full_confirm_rate": 0.9, "ablation_confirm_rate": 0.65,
                          "lift": 0.25, "missed_changes": ["picked up restaking"], "n_missed_changes": 1}],
               "coverage": [{"date": "2022-12-31", "grounded": 13, "persisted": 0, "extrapolated": 0}]}
    p = write_memo(metrics, out_dir=tmp_path)
    txt = Path(p).read_text()
    assert p == tmp_path / "eddy-lazzarin" / "findings.md"
    assert "corpus-lift" in txt.lower() and "0.25" in txt          # headline
    assert "2022-12-31" in txt and "0.9" in txt and "0.65" in txt  # per-step table
    assert "picked up restaking" in txt                            # missed change
    assert "persistence" in txt.lower() and "soul-less" in txt.lower()  # caveats stated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -k memo -v`
Expected: FAIL — `ImportError: cannot import name 'write_memo'`.

- [ ] **Step 3: Write minimal implementation** (append to `doppelganger/score.py`)

```python
def _fmt(x) -> str:
    return "—" if x is None else f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def write_memo(metrics: dict, discrimination: dict | None = None, *, out_dir: Path | None = None) -> Path:
    slug = metrics["subject"]
    lift = metrics.get("mean_lift")
    lines = [
        f"# Doppelganger findings — {slug}",
        "",
        f"**Headline — mean corpus-lift: {_fmt(lift)}** "
        f"(full confirm-rate minus the soul-less ablation, horizon {metrics['horizon_months']} mo).",
        "",
        "_Caveats (read these with the number):_ the lift nets out **persistence** "
        "(both arms get easy credit for stable views, which partly cancels) and bounds "
        "**model-hindsight** (the **soul-less** arm is the parametric floor — what Opus produces "
        "with no corpus). The judge scored only from the subject's provided later statements.",
        "",
        "## Per-step held-out prediction",
        "",
        "| date | full | ablation (soul-less) | lift |",
        "|---|---|---|---|",
    ]
    for s in metrics["steps"]:
        lines.append(f"| {s['date']} | {_fmt(s['full_confirm_rate'])} | "
                     f"{_fmt(s['ablation_confirm_rate'])} | {_fmt(s['lift'])} |")
    lines += ["", "## Missed changes (foresight gaps)", ""]
    any_missed = False
    for s in metrics["steps"]:
        for mc in s.get("missed_changes", []):
            lines.append(f"- ({s['date']}) {mc}")
            any_missed = True
    if not any_missed:
        lines.append("- none flagged")
    lines += ["", "## Coverage trajectory (full arm)", "",
              "| date | grounded | persisted | extrapolated |", "|---|---|---|---|"]
    for c in metrics.get("coverage", []):
        lines.append(f"| {c['date']} | {c['grounded']} | {c['persisted']} | {c['extrapolated']} |")
    if discrimination is not None:
        lines += ["", "## Discrimination", "",
                  f"- sector overlap vs comparator: {_fmt(discrimination.get('sector_overlap'))}",
                  f"- token overlap vs comparator: {_fmt(discrimination.get('token_overlap'))}"]

    out = Path(out_dir or config.OUT_DIR) / slug / "findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -v`
Expected: PASS (all score tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/score.py tests/test_doppelganger_score.py
git commit -m "feat(doppelganger): write_memo — deterministic findings template"
```

---

## Task 7: CLI `score` subcommand

**Files:** Modify `doppelganger/run.py`; Test `tests/test_doppelganger_score.py`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_run_cli_has_score_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["score", "--subject", "eddy-lazzarin"])
    assert ns.cmd == "score" and ns.subject == "eddy-lazzarin" and ns.horizon_months == 6
    ns2 = r.build_parser().parse_args(["score", "--subject", "x", "--horizon-months", "3"])
    assert ns2.horizon_months == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -k cli -v`
Expected: FAIL — argparse rejects `score`.

- [ ] **Step 3: Modify `doppelganger/run.py`**:
  (i) Add import at the top: `from doppelganger.score import score_subject, write_memo`.
  (ii) In `build_parser()`, after the `walkforward` subparser and before `return parser`:
```python
    sc = sub.add_parser("score", help="judge the walk-forward trajectory -> metrics.json + findings.md")
    sc.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    sc.add_argument("--horizon-months", type=int, default=6, help="held-out window length (months)")
```
  (iii) In `main()`, after the `walkforward` branch:
```python
    elif args.cmd == "score":
        metrics = score_subject(args.subject, horizon_months=args.horizon_months)
        memo = write_memo(metrics)
        print(f"mean_lift={metrics['mean_lift']} -> {config.OUT_DIR / args.subject / 'metrics.json'} + {memo}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_score.py -v`
Expected: PASS. Then full suite `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q` → all PASS.

- [ ] **Step 5: Commit**
```bash
git add doppelganger/run.py tests/test_doppelganger_score.py
git commit -m "feat(doppelganger): score CLI subcommand"
```

---

## Task 8: Real-data validation (Eddy subset) — not TDD

Requires the `claude` CLI on the Max subscription. Regenerates Eddy's soul-less ablation views (the old `views_ablation` are the confounded soul-only ones), then scores the 3-step subset.

- [ ] **Step 1: Confirm full suite green**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q`
Expected: all PASS.

- [ ] **Step 2: Regenerate the soul-less ablation views for the subset**

Run:
```bash
rm -f data/doppelganger/eddy-lazzarin/views_ablation/2022-12-31.json \
      data/doppelganger/eddy-lazzarin/views_ablation/2023-03-31.json \
      data/doppelganger/eddy-lazzarin/views_ablation/2023-06-30.json
.venv/bin/python - <<'PY'
from datetime import date
from doppelganger.walkforward import run_walkforward, quarter_ends
dates = quarter_ends(date(2022, 12, 31), date(2023, 6, 30))
run_walkforward("eddy-lazzarin", dates)   # full views cached; regenerates soul-less ablation arm (~3 claude -p)
print("ablation regenerated")
PY
```
Expected: the 3 ablation views are regenerated soul-less (full views come from cache). ~3 `claude -p` (~12 min).

- [ ] **Step 3: Score the subset**

Run: `.venv/bin/python -m doppelganger.run score --subject eddy-lazzarin`
Expected: runs the judge over the scorable steps (those with a post-T window — the last subset date 2023-06-30 may have a full 6-mo window; earlier dates definitely do), writes `metrics.json` + `findings.md`. ~4-6 `claude -p` (~20 min).

- [ ] **Step 4: Read the findings**

Run: `cat data/doppelganger/eddy-lazzarin/findings.md`

Report honestly: **is the mean corpus-lift positive** (full predicts above the soul-less floor)? **Are there missed changes** (foresight gaps)? Is the per-step lift stable or noisy on n=2-3? This is n=3, one subject — a smoke test, not a verdict. If lift ≈ 0 or negative, say so plainly: it would mean the soul-less model already matches the corpus-grounded one, i.e. model-hindsight may dominate — a real and important finding. Report the numbers to the controller.

- [ ] **Step 5: Commit any fixes**

If the run revealed a bug, fix + re-run the affected test + commit. Otherwise report. `metrics.json`/`findings.md`/`judge/` are generated artifacts — NOT committed (regenerate via CLI), consistent with prior units.

---

## Self-review

**Spec coverage (scorers-unit-design.md):**
- §2 soul-less ablation (identity stub, no soul/memory) → Task 1. ✓
- §3 post_t_evidence + judge_step (per-claim, confirm_rate, missed_changes, honesty guard, cached) → Tasks 2, 3. ✓
- §4 discrimination (Jaccard) + coverage_trajectory → Task 4. ✓
- §5 score_subject (lift orchestrator, metrics.json, skip unscorable steps) + write_memo (templated findings.md, caveats inline) → Tasks 5, 6. ✓
- §5 `score` CLI → Task 7. ✓
- §7 real-data validation (regen soul-less ablation, score subset, honest read) → Task 8. ✓

**Placeholder scan:** none — every code step has complete code. Only Task 8 calls the real LLM (explicitly non-TDD).

**Type consistency:** `respond(..., ablate_memory=...)` unchanged name (Task 1). `post_t_evidence(slug, t0, horizon_months, *, evidence_path)` and `judge_step(view, post_t_text, subject_name, t0, *, judge_path)` consistent across Tasks 2/3/5. `discrimination(view_a, view_b)`, `coverage_trajectory(rows)`, `score_subject(slug, *, horizon_months, out_dir, evidence_path)`, `write_memo(metrics, discrimination=None, *, out_dir)` consistent across Tasks 4-7. `confirm_rate` None-semantics identical in judge + orchestrator + memo. Judge cache path `judge/<date>_<variant>.json` matches between Task 5 and Task 3's `judge_path` contract.

---

## Execution handoff

After 5b: the **full quarterly walk-forward run** (Eddy + Ali, all quarters, ~2 hr `claude -p`) then `score` both → the first real findings on whether the corpus-doppelganger predicts above the parametric floor. That run is a separate execution, not code.
