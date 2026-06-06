# Corpus Doppelganger — Unit ❹ Doppelganger (Respond) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The doppelganger reasoner — combine the frozen soul card + the ≤T memory feed in one `claude -p` pass to answer a market-view query as the subject at date T, emitting a structured, provenance-tagged JSON market view.

**Architecture:** A new `doppelganger/respond.py` (`build_query_prompt`, `respond`). Two behavior-preserving DRY refactors first: extract the `claude -p` wrapper into `doppelganger/llm.py` (shared by soul + respond), and extract the citation-audit core in `soul_audit.py` so an `audit_answer` can reuse it. Immersive present-tense framing suppresses model-hindsight; provenance tags mark leakage exposure.

**Tech Stack:** Python 3.13, pandas, `claude` CLI (print mode). pytest. Builds on Units 1-3.

**Spec:** `docs/superpowers/specs/2026-06-05-doppelganger-respond-unit-design.md`.

---

## File structure

```
doppelganger/
  llm.py           # NEW: run_claude + CLAUDE_* constants (moved from soul.py)
  respond.py       # NEW: build_query_prompt, respond, _parse_view
  soul.py          # MODIFY: import run_claude from llm (drop its own copy)
  soul_audit.py    # MODIFY: extract audit_citations core; add audit_answer
  run.py           # MODIFY: add `respond` subcommand
tests/
  test_doppelganger_llm.py       # NEW: the moved run_claude tests
  test_doppelganger_respond.py   # NEW
  test_doppelganger_soul.py      # MODIFY: drop moved tests, repoint extract_soul patch
  test_doppelganger_soul_audit.py# MODIFY: add audit_answer tests
```

---

## Task 1: Extract `run_claude` into `llm.py` (refactor, behavior-preserving)

**Files:** Create `doppelganger/llm.py`, `tests/test_doppelganger_llm.py`; Modify `doppelganger/soul.py`, `tests/test_doppelganger_soul.py`.

- [ ] **Step 1: Create `doppelganger/llm.py`**

```python
"""doppelganger.llm — the shared `claude -p` subprocess wrapper (Max sub, no API cost).

Instructions go via --system-prompt; the large payload via stdin. Runs from an
isolated dir so the call does NOT inherit this repo's CLAUDE.md.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

CLAUDE_MODEL = "opus"
CLAUDE_EFFORT = "max"
CLAUDE_TIMEOUT_S = 900   # 15 min; a single pass over a big corpus can be slow


def run_claude(system: str, user: str, *, workdir: Path | None = None,
               timeout: int = CLAUDE_TIMEOUT_S) -> str:
    """Run `claude -p`; return stdout (stripped). Raise RuntimeError on non-zero exit."""
    wd = workdir or Path(tempfile.mkdtemp(prefix="doppelganger-"))
    Path(wd).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["claude", "-p", "--model", CLAUDE_MODEL, "--effort", CLAUDE_EFFORT,
         "--system-prompt", system, "--no-session-persistence"],
        input=user, cwd=str(wd), capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exited {proc.returncode}. stderr:\n{proc.stderr}")
    return proc.stdout.strip()
```

- [ ] **Step 2: Create `tests/test_doppelganger_llm.py`** (the two run_claude tests, repointed at `doppelganger.llm`)

```python
"""TDD tests for doppelganger.llm."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from doppelganger.llm import run_claude


def test_run_claude_invokes_cli_with_stdin(tmp_path):
    fake = MagicMock(returncode=0, stdout="## Bio Lens\nok\n", stderr="")
    with patch("doppelganger.llm.subprocess.run", return_value=fake) as mrun:
        out = run_claude("SYS", "USERBODY", workdir=tmp_path)
    assert out == "## Bio Lens\nok"
    args, kwargs = mrun.call_args
    cmd = args[0]
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "max"
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert kwargs["input"] == "USERBODY"
    assert str(kwargs["cwd"]) == str(tmp_path)


def test_run_claude_raises_on_nonzero():
    fake = MagicMock(returncode=2, stdout="", stderr="boom")
    with patch("doppelganger.llm.subprocess.run", return_value=fake):
        try:
            run_claude("SYS", "U")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "boom" in str(e)
```

- [ ] **Step 3: Update `doppelganger/soul.py` to use `llm.run_claude`** — three edits:

  (i) Replace the import lines (currently `import subprocess` / `import tempfile` near the top) — remove both and add the llm import. The import block becomes:
```python
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.identity import build_identity
from doppelganger.ingest import build_evidence_stream
from doppelganger.llm import CLAUDE_EFFORT, CLAUDE_MODEL, run_claude
from doppelganger.schema import IdentityProfile
```

  (ii) DELETE the now-duplicated block in `soul.py` — the three constants `CLAUDE_MODEL = "opus"`, `CLAUDE_EFFORT = "max"`, `CLAUDE_TIMEOUT_S = 900`, AND the entire `def _run_claude(...)` function (everything from `CLAUDE_MODEL = "opus"` through the end of `_run_claude`'s `return proc.stdout.strip()`). `_frontmatter` still references `CLAUDE_MODEL`/`CLAUDE_EFFORT`, now imported.

  (iii) In `extract_soul`, change the call `card = _run_claude(system, user)` to `card = run_claude(system, user)`.

- [ ] **Step 4: Update `tests/test_doppelganger_soul.py`** — two edits:

  (i) DELETE the two tests `test_run_claude_invokes_cli_with_stdin` and `test_run_claude_raises_on_nonzero` and their `from doppelganger.soul import _run_claude` import (they now live in `test_doppelganger_llm.py`).

  (ii) In `test_extract_soul_writes_card_with_frontmatter`, change the patch target `patch("doppelganger.soul._run_claude", ...)` to `patch("doppelganger.soul.run_claude", ...)`.

- [ ] **Step 5: Run the full suite to verify the refactor is green**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q`
Expected: all PASS (llm tests pass; soul tests pass with the repointed patch; nothing references the old `_run_claude`).

- [ ] **Step 6: Commit**
```bash
git add doppelganger/llm.py doppelganger/soul.py tests/test_doppelganger_llm.py tests/test_doppelganger_soul.py
git commit -m "refactor(doppelganger): extract claude -p wrapper into llm.run_claude"
```

---

## Task 2: Extract `audit_citations` + add `audit_answer` (`soul_audit.py`)

**Files:** Modify `doppelganger/soul_audit.py`; Test `tests/test_doppelganger_soul_audit.py`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_doppelganger_soul_audit.py`)

```python
import json as _json
from doppelganger.soul_audit import audit_answer


def _ev_for_answer(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([
        {"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives and that is the core thesis."},
        {"id": "2", "timestamp": pd.Timestamp("2023-03-01", tz="UTC"),
         "source_type": "x_original", "text": "ZK rollups are the endgame for scaling."},
    ]).to_parquet(p)
    return p


def _view(citation_date, quote):
    return {"sectors_excited": [{"name": "DeFi", "why": "x", "provenance": "grounded",
            "citations": [{"date": citation_date, "quote": quote}]}],
            "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
            "risk_regime": {"stance": "no_view"}}


def test_audit_answer_passes_clean(tmp_path):
    ev = _ev_for_answer(tmp_path)
    rep = audit_answer(_view("2022-06-01", "Tokens align incentives"), ev, date(2022, 12, 31))
    assert rep.ok and rep.matched == 1


def test_audit_answer_flags_hallucinated(tmp_path):
    ev = _ev_for_answer(tmp_path)
    rep = audit_answer(_view("2022-06-01", "this was never said by anyone at all"), ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.hallucinated) == 1


def test_audit_answer_flags_leaked(tmp_path):
    ev = _ev_for_answer(tmp_path)
    rep = audit_answer(_view("2023-03-01", "ZK rollups are the endgame"), ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.leaked) == 1


def test_audit_answer_reads_path(tmp_path):
    ev = _ev_for_answer(tmp_path)
    f = tmp_path / "view.json"
    f.write_text(_json.dumps(_view("2022-06-01", "Tokens align incentives")))
    rep = audit_answer(f, ev, date(2022, 12, 31))
    assert rep.ok and rep.matched == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul_audit.py -k audit_answer -v`
Expected: FAIL — `ImportError: cannot import name 'audit_answer'`.

- [ ] **Step 3: Refactor `audit_soul` + add `audit_citations` and `audit_answer`** in `doppelganger/soul_audit.py`.

  (i) Add imports at the top (alongside the existing ones): `import json` and `from pathlib import Path` (Path is already imported — keep it; add `import json`).

  (ii) REPLACE the existing `audit_soul` function (the whole `def audit_soul(...)` body that loads evidence, builds `norm_items`, and loops citations) with this trio:
```python
def audit_citations(cites: list[Citation], evidence_path: Path, t0: date) -> AuditReport:
    ev = pd.read_parquet(evidence_path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    norm_items = [(_norm(t), pd.Timestamp(ts).date())
                  for t, ts in zip(ev["text"], ev["timestamp"])]

    matched, hallucinated, leaked = 0, [], []
    for c in cites:
        q = _norm(c.quote)
        hits = [d for text, d in norm_items if _coverage(q, text) >= _MATCH_THRESHOLD]
        if not hits:
            hallucinated.append(c)
        elif min(hits) > t0 or c.date > t0:
            leaked.append(c)
        else:
            matched += 1
    return AuditReport(len(cites), matched, hallucinated, leaked)


def audit_soul(card_path: Path, evidence_path: Path, t0: date) -> AuditReport:
    cites = parse_citations(Path(card_path).read_text())
    return audit_citations(cites, evidence_path, t0)


_ANSWER_ARRAYS = ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned"]


def audit_answer(view, evidence_path: Path, t0: date) -> AuditReport:
    """Audit a market-view JSON (dict or path) — pull {date,quote} citations and check them."""
    if isinstance(view, (str, Path)):
        view = json.loads(Path(view).read_text())
    cites: list[Citation] = []
    for key in _ANSWER_ARRAYS:
        for item in view.get(key, []) or []:
            for c in item.get("citations", []) or []:
                try:
                    y, m, d = (int(x) for x in str(c["date"]).split("-"))
                    cites.append(Citation(date(y, m, d), str(c["quote"])))
                except (KeyError, ValueError, TypeError):
                    continue
    return audit_citations(cites, evidence_path, t0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul_audit.py -v`
Expected: PASS — the new `audit_answer` tests AND all existing soul-audit tests (audit_soul still works via the extracted core).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/soul_audit.py tests/test_doppelganger_soul_audit.py
git commit -m "feat(doppelganger): audit_citations core + audit_answer for market-view JSON"
```

---

## Task 3: `respond.py` — prompt assembly (`build_query_prompt`)

**Files:** Create `doppelganger/respond.py`; Test `tests/test_doppelganger_respond.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_doppelganger_respond.py`:

```python
"""TDD tests for doppelganger.respond."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.respond import build_query_prompt

SOUL = "---\nname: Eddy Lazzarin\nt0: 2022-12-31\n---\n\n## How He Thinks\nMechanism-first."


def test_build_query_prompt_structure():
    system, user = build_query_prompt(SOUL, "[2022-06-01] (x_original) Tokens align incentives.",
                                      "eddy-lazzarin", date(2022, 12, 31))
    # immersive present-tense framing
    assert "You ARE Eddy Lazzarin" in system
    assert "It is 2022-12-31" in system
    assert "future has not happened" in system.lower()
    # schema keys + rules present
    for k in ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned",
              "risk_regime", "provenance"]:
        assert k in system
    assert "never manufacture" in system.lower()
    assert "abstained" in system
    # soul card embedded in system
    assert "Mechanism-first." in system
    # memory + query in the stdin payload
    assert "Tokens align incentives." in user
    assert "market view" in user.lower()


def test_build_query_prompt_custom_query():
    system, user = build_query_prompt(SOUL, "mem", "eddy-lazzarin", date(2022, 12, 31),
                                      query="What about ZK rollups?")
    assert "What about ZK rollups?" in user
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.respond'`.

- [ ] **Step 3: Write minimal implementation** — `doppelganger/respond.py` (this task: header + `build_query_prompt`):

```python
"""doppelganger.respond — answer a market-view query AS the subject at date T.

Combines the frozen soul card (who he is) + the time-gated memory feed (what he's
said) in one claude -p pass, returning a structured, provenance-tagged JSON view.
Immersive present-tense framing suppresses model-hindsight.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from doppelganger import config
from doppelganger.llm import run_claude
from doppelganger.memory import load_memory

SURVEY_QUERY = (
    "What are your current market views right now? Which sectors and which tokens are you "
    "excited about, which are you concerned about, and what is your risk-on / risk-off posture "
    "— and why? Remember: it's fine to have a view on a sector but not a specific token (or vice "
    "versa), and fine to have no view on something."
)

_SCHEMA_HINT = """{
  "as_of": "<today's date>", "subject": "<your slug>", "abstained": false,
  "sectors_excited":   [{"name": "...", "why": "...", "provenance": "grounded|persisted|extrapolated", "age_note": "", "citations": [{"date": "YYYY-MM-DD", "quote": "verbatim"}]}],
  "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
  "risk_regime": {"stance": "risk_on|risk_off|neutral|no_view", "why": "...", "provenance": "..."},
  "notes": "..."
}"""


def _name_from_soul(soul_md: str, fallback: str) -> str:
    m = re.search(r"(?m)^name:\s*(.+)$", soul_md)
    return m.group(1).strip() if m else fallback


def build_query_prompt(soul_md: str, memory_text: str, subject: str, t0: date,
                       query: str | None = None) -> tuple[str, str]:
    name = _name_from_soul(soul_md, subject)
    system = f"""You ARE {name}. It is {t0.isoformat()}. The future has not happened yet — you \
know only what you know as of today. Reason in the present tense, as yourself, in real time. Use \
ONLY your character description below and your own record of what you've said and seen; do NOT use \
anything you might know about events after today.

When asked for your market views, answer as a SINGLE JSON object with exactly this shape:
{_SCHEMA_HINT}

RULES:
- Each axis is INDEPENDENT and OPTIONAL. Excited about a sector with no specific token is a complete \
answer; a token concern with no broad sector view is complete; any array may be empty — that is \
expected, not a gap. NEVER manufacture an item just to fill a bucket. If you genuinely have no view \
at all, set "abstained": true with empty arrays.
- provenance per item: "grounded" = you have actually said this (put a dated verbatim quote in \
citations); "persisted" = a view you still hold but haven't restated recently (cite it, set age_note \
e.g. "stated 2021-06, not revisited"); "extrapolated" = inferred from how you think, no direct quote \
(no citation). Cite verbatim.
- Output ONLY the JSON object — no prose before or after.

--- WHO YOU ARE (your soul) ---
{soul_md}"""
    q = query or SURVEY_QUERY
    user = (f"# YOUR RECORD — everything you've said and seen, through today ({t0.isoformat()})\n\n"
            f"{memory_text}\n\n# QUESTION\n\n{q}")
    return system, user
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/respond.py tests/test_doppelganger_respond.py
git commit -m "feat(doppelganger): respond prompt assembly (present-tense framing + schema)"
```

---

## Task 4: `respond.py` — orchestrator (`respond` + `_parse_view`)

**Files:** Modify `doppelganger/respond.py`; Test `tests/test_doppelganger_respond.py`.

- [ ] **Step 1: Write the failing tests** (append)

```python
from unittest.mock import patch
from doppelganger.respond import respond

FIX = Path("tests/fixtures/doppelganger")


def _canned(fenced=False):
    body = ('{"as_of":"2022-12-31","subject":"testy-mctest","abstained":false,'
            '"sectors_excited":[{"name":"DeFi","why":"w","provenance":"grounded","citations":[{"date":"2022-06-01","quote":"q"}]}],'
            '"sectors_concerned":[],"tokens_excited":[],"tokens_concerned":[],'
            '"risk_regime":{"stance":"risk_on","why":"w","provenance":"grounded"},"notes":""}')
    return f"```json\n{body}\n```" if fenced else body


def _soul(tmp_path):
    p = tmp_path / "soul.md"
    p.write_text("---\nname: Testy McTest\n---\n\n## How He Thinks\nMechanism-first.")
    return p


def test_respond_parses_and_writes(tmp_path):
    with patch("doppelganger.respond.run_claude", return_value=_canned(fenced=True)):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                       soul_path=_soul(tmp_path),
                       evidence_path=FIX / "twitter" / "testy.parquet")  # any parquet w/ timestamp/text/source_type/context
    assert view["subject"] == "testy-mctest" and view["abstained"] is False
    assert view["sectors_excited"][0]["name"] == "DeFi"
    assert view["risk_regime"]["stance"] == "risk_on"
    out = tmp_path / "testy-mctest" / "views" / "2022-12-31.json"
    assert out.exists()
    import json as j
    assert j.loads(out.read_text())["subject"] == "testy-mctest"


def test_respond_normalizes_missing_keys(tmp_path):
    minimal = '{"sectors_excited":[]}'   # everything else missing
    with patch("doppelganger.respond.run_claude", return_value=minimal):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                       soul_path=_soul(tmp_path),
                       evidence_path=FIX / "twitter" / "testy.parquet")
    assert view["sectors_concerned"] == [] and view["tokens_excited"] == []
    assert view["risk_regime"] == {"stance": "no_view"}
    assert view["abstained"] is False and view["as_of"] == "2022-12-31"
    assert view["subject"] == "testy-mctest"


def test_respond_raises_on_non_json(tmp_path):
    with patch("doppelganger.respond.run_claude", return_value="I cannot help with that."):
        try:
            respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                    soul_path=_soul(tmp_path), evidence_path=FIX / "twitter" / "testy.parquet")
            assert False, "should raise"
        except ValueError:
            pass
```

Note: the `evidence_path` fixture `twitter/testy.parquet` has the columns `load_memory` needs (`timestamp`, `source_type`, `text`, `context`? — it has `created_at` not `timestamp`). USE a tiny built parquet instead — replace the `evidence_path` in these tests with a local `_ev` helper:
```python
def _ev(tmp_path):
    import pandas as pd
    p = tmp_path / "ev.parquet"
    pd.DataFrame([{"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                   "source_type": "x_original", "text": "Tokens align incentives.", "context": None}]).to_parquet(p)
    return p
```
and call `respond(..., evidence_path=_ev(tmp_path))` in all three tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -k respond -v`
Expected: FAIL — `ImportError: cannot import name 'respond'`.

- [ ] **Step 3: Write minimal implementation** (append to `doppelganger/respond.py`)

```python
_ARRAYS = ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned"]


def _parse_view(raw: str, subject: str, t0: date) -> dict:
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"no JSON object in claude output: {raw[:200]!r}")
    data = json.loads(raw[i:j + 1])          # raises json.JSONDecodeError (a ValueError) if malformed
    for k in _ARRAYS:
        if not isinstance(data.get(k), list):
            data[k] = []
    if not isinstance(data.get("risk_regime"), dict):
        data["risk_regime"] = {"stance": "no_view"}
    data.setdefault("as_of", t0.isoformat())
    data.setdefault("subject", subject)
    data.setdefault("abstained", False)
    data.setdefault("notes", "")
    return data


def respond(slug: str, t0: date, *, query: str | None = None,
            soul_path: Path | None = None, evidence_path: Path | None = None,
            out_dir: Path | None = None) -> dict:
    sp = soul_path or (config.OUT_DIR / slug / "soul.md")
    soul_md = Path(sp).read_text()
    mv = load_memory(slug, t0, evidence_path=evidence_path)
    system, user = build_query_prompt(soul_md, mv.text, slug, t0, query)
    raw = run_claude(system, user)
    view = _parse_view(raw, slug, t0)

    base = Path(out_dir or config.OUT_DIR) / slug / "views"
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{t0.isoformat()}.json").write_text(json.dumps(view, indent=2))
    return view
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -v`
Expected: PASS (all respond tests). `json.JSONDecodeError` subclasses `ValueError`, so the non-JSON test's `except ValueError` covers both the "no braces" and "malformed JSON" cases.

- [ ] **Step 5: Commit**
```bash
git add doppelganger/respond.py tests/test_doppelganger_respond.py
git commit -m "feat(doppelganger): respond orchestrator — parse/normalize/write market view"
```

---

## Task 5: CLI `respond` subcommand

**Files:** Modify `doppelganger/run.py`; Test `tests/test_doppelganger_respond.py`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_run_cli_has_respond_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["respond", "--subject", "eddy-lazzarin", "--t0", "2022-12-31"])
    assert ns.cmd == "respond" and ns.subject == "eddy-lazzarin" and ns.t0 == "2022-12-31"
    assert ns.query is None
    ns2 = r.build_parser().parse_args(["respond", "--subject", "x", "--t0", "2022-12-31", "--query", "Q"])
    assert ns2.query == "Q"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -k cli -v`
Expected: FAIL — argparse rejects `respond`.

- [ ] **Step 3: Modify `doppelganger/run.py`** — add the `respond` subparser inside `build_parser()` (after the `memory` subparser, before `return parser`):
```python
    resp = sub.add_parser("respond", help="answer a market-view query as the subject at date T")
    resp.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    resp.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")
    resp.add_argument("--query", default=None, help="optional custom query (default: market-view survey)")
```
  and add the handler branch in `main()` (after the `memory` branch); also add the import `from doppelganger.respond import respond` at the top:
```python
    elif args.cmd == "respond":
        view = respond(args.subject, date.fromisoformat(args.t0), query=args.query)
        import json
        print(json.dumps(view, indent=2)[:2000])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_respond.py -v`
Expected: PASS. Then full suite: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q` → all PASS.

- [ ] **Step 5: Commit**
```bash
git add doppelganger/run.py tests/test_doppelganger_respond.py
git commit -m "feat(doppelganger): respond CLI subcommand"
```

---

## Task 6: Real-data gate (Eddy + Ali) — not TDD

Requires the `claude` CLI logged into the Max subscription, and Eddy's + Ali's `soul.md` + `evidence.parquet` present (built in Unit 2's gate). Each `respond` call is one `claude -p` pass (~few minutes).

- [ ] **Step 1: Confirm full suite green**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q`
Expected: all PASS.

- [ ] **Step 2: Generate Eddy's T0 market view**

Run: `.venv/bin/python -m doppelganger.run respond --subject eddy-lazzarin --t0 2022-12-31`
Expected: prints a JSON market view; writes `data/doppelganger/eddy-lazzarin/views/2022-12-31.json`.

- [ ] **Step 3: Audit Eddy's answer (grounding + leakage)**

Run:
```bash
.venv/bin/python - <<'PY'
from datetime import date
from doppelganger.soul_audit import audit_answer
rep = audit_answer("data/doppelganger/eddy-lazzarin/views/2022-12-31.json",
                   "data/doppelganger/eddy-lazzarin/evidence.parquet", date(2022, 12, 31))
print(f"ok={rep.ok} checked={rep.checked} matched={rep.matched} "
      f"hallucinated={len(rep.hallucinated)} leaked={len(rep.leaked)}")
for c in rep.leaked[:6]: print("  LEAK:", c.date, repr(c.quote[:70]))
PY
```
Acceptance: **`leaked == 0`** is the hard gate. A few `hallucinated` may be model paraphrase (same as the soul gate) — investigate but don't block on it. Report the numbers.

- [ ] **Step 4: Generate Ali's answer + discrimination eyeball**

Run:
```bash
.venv/bin/python -m doppelganger.run respond --subject ali-yahya --t0 2022-12-31
echo "=== EDDY ==="; cat data/doppelganger/eddy-lazzarin/views/2022-12-31.json
echo "=== ALI ==="; cat data/doppelganger/ali-yahya/views/2022-12-31.json
```
Judge (analytical, not stylistic): do Eddy and Ali hold **recognizably different market views** — different sectors/tokens excited/concerned, different risk posture, different reasoning — or near-identical generic crypto takes? Also sanity-check the **provenance mix** (a healthy answer is mostly `grounded`/`persisted`, with `extrapolated` clearly marked — the leakage-exposed set). Write a 3-5 sentence verdict.

- [ ] **Step 5: Commit any prompt tuning**

If the gate prompted a change in `respond.py` (e.g. the model ignored the partial-is-OK rule, or over-used `extrapolated`), fix it, re-run the affected tests, and:
```bash
git add -A && git commit -m "fix(doppelganger): tune respond prompt per real-data gate"
```
Otherwise report the audit numbers + discrimination verdict to the controller. Generated `views/*.json` are NOT committed (regenerate via CLI), consistent with Units 1-2.

---

## Self-review

**Spec coverage (respond-unit-design.md):**
- §2 output schema (sectors/tokens excited+concerned, risk_regime, per-item provenance + citations + age_note, optionality, abstention) → Task 3 (schema in prompt), Task 4 (parse/normalize). ✓
- §1/§2 immersive present-tense framing + never-manufacture rule + provenance defs → Task 3. ✓
- §3 prompt assembly (system=instructions+soul, stdin=memory+query) → Task 3. ✓
- §3 parsing (strip fences via first-`{`/last-`}`, lenient normalize, raise only on non-JSON) → Task 4. ✓
- §3 DRY refactors (run_claude→llm; audit_citations core + audit_answer) → Tasks 1, 2. ✓
- §3 writes views/<t0>.json → Task 4. ✓
- §4 modules (llm.py, respond.py, soul/soul_audit modify, run respond cmd) → Tasks 1-5. ✓
- §5 testing (plumbing mocked; audit_answer; refactors keep soul tests green; real-data gate) → all tasks + Task 6. ✓

**Placeholder scan:** none — every code step has complete code. Task 4 Step 1 explicitly supplies the `_ev` helper to avoid a column-mismatch footgun.

**Type consistency:** `run_claude(system, user, *, workdir, timeout)` defined in Task 1, called in soul (Task 1) + respond (Task 4). `build_query_prompt(soul_md, memory_text, subject, t0, query=None) -> (system, user)` and `respond(slug, t0, *, query, soul_path, evidence_path, out_dir) -> dict` consistent across Tasks 3-5 + CLI. `audit_citations(cites, evidence_path, t0)` / `audit_answer(view, evidence_path, t0)` consistent in Task 2 + Task 6. `_ARRAYS`/`_ANSWER_ARRAYS` are the same four keys.

---

## Execution handoff

After this unit: Unit ❺ (Eval) runs `respond()` across the walk-forward (T0=2022-12-31 monthly), the with/without-corpus ablation (model-leakage bound), cross-author discrimination, and held-out prediction scoring against the subjects' actual later statements.
