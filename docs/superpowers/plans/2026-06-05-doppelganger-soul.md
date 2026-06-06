# Corpus Doppelganger — Unit ❷ Soul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the soul unit — a single Opus pass (`claude -p`) over a subject's bio + ≤T0 evidence that produces a frozen, evidence-grounded Markdown "soul card", plus an automated audit that proves every cited quote is real and not time-leaked.

**Architecture:** A `doppelganger/soul.py` module that gates inputs (`evidence ≤ t0`, `identity.as_of(t0)`), assembles a prompt, runs `claude -p` in an isolated temp dir (Max subscription, no API cost), and writes `soul.md`. A separate `doppelganger/soul_audit.py` parses the card's inline citations and verifies each against the evidence. The LLM call sits behind one thin function (`_run_claude`) that tests monkeypatch — so all plumbing + audit logic is TDD'd without ever calling the real model.

**Tech Stack:** Python 3.13, pandas/pyarrow, `claude` CLI (print mode). pytest. Builds on Unit 1 (`doppelganger/{config,schema,identity}.py`, `data/doppelganger/<slug>/evidence.parquet`).

**Spec:** `docs/superpowers/specs/2026-06-05-doppelganger-soul-unit-design.md`. First subject `eddy-lazzarin`, T0 = `2022-12-31`.

---

## File structure

```
doppelganger/
  soul.py          # load_soul_inputs, build_extraction_prompt, _run_claude, extract_soul
  soul_audit.py    # Citation, AuditReport, parse_citations, audit_soul
  run.py           # add `soul` subcommand (MODIFY existing)
tests/
  test_doppelganger_soul.py
  test_doppelganger_soul_audit.py
```

**Locked conventions (use exactly):**

- **Citation format in the card** — every factual claim is immediately followed by ≥1 inline citation: an ISO date in square brackets then the verbatim quote in straight double quotes, e.g. `[2022-07-01] "rollups inherit security"`. The audit parses these with regex `\[(\d{4}-\d{2}-\d{2})\]\s+"([^"]{3,})"`.
- **`claude -p` invocation** (from the original doppelganger runner): `claude -p --model opus --effort max --system-prompt <INSTRUCTIONS> --no-session-persistence`, `cwd=<isolated temp dir>`, **user content (bio + evidence) piped via stdin** (not a CLI arg — avoids ARG_MAX on the ~40–200k-token payload), stdout = the card.
- **t0** is a `datetime.date` (CLI parses `YYYY-MM-DD`). Evidence gating: `evidence[evidence["timestamp"].dt.date <= t0]`.
- **Match normalization** (audit): lowercase, collapse all whitespace to single spaces, strip. A quote matches an evidence item if `quote_norm` is a substring of `item_text_norm`.

---

## Task 1: Input loading + time-gating (`load_soul_inputs`)

**Files:**
- Create: `doppelganger/soul.py`
- Test: `tests/test_doppelganger_soul.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_soul.py`:
```python
"""TDD tests for doppelganger.soul."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.soul import load_soul_inputs

FIX = Path("tests/fixtures/doppelganger")


def test_load_soul_inputs_gates_by_t0():
    identity, evidence = load_soul_inputs(
        "testy-mctest", date(2022, 8, 31),
        evidence_path=None,  # use the built fixture stream below
        identity_path=FIX / "linkedin" / "testy-1.json",
        team_path=FIX / "team.parquet",
        tracked_people_path=FIX / "tracked_people.yaml",
        twitter_path=FIX / "twitter" / "testy.parquet",
        articles_path=FIX / "articles.parquet",
        podcast_path=FIX / "attributed_transcripts.jsonl",
    )
    # identity is truncated to <= 2022-08-31 (Engineer only; GP/CTO are 2023+/2026)
    assert [e.title for e in identity.experience] == ["Engineer"]
    # evidence is filtered to <= 2022-08-31: the 2022-09-01 quote (id "6") is excluded
    assert (evidence["timestamp"].dt.date <= date(2022, 8, 31)).all()
    assert "6" not in set(evidence["id"])
    # sorted ascending
    assert list(evidence["timestamp"]) == sorted(evidence["timestamp"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.soul'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/soul.py` (this task adds the header + `load_soul_inputs` only):
```python
"""doppelganger.soul — build the frozen-at-T0 soul card via a single claude -p pass.

The LLM call is isolated in `_run_claude` so the rest is deterministically testable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.identity import build_identity
from doppelganger.ingest import build_evidence_stream
from doppelganger.schema import IdentityProfile


def load_soul_inputs(
    slug: str, t0: date, *,
    evidence_path: Path | None = None,
    identity_path: Path | None = None,
    team_path: Path | None = None,
    tracked_people_path: Path | None = None,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
) -> tuple[IdentityProfile, pd.DataFrame]:
    """Return (identity truncated to <=t0, evidence DataFrame filtered to <=t0, sorted)."""
    identity = build_identity(
        slug, linkedin_path=identity_path, team_path=team_path,
        tracked_people_path=tracked_people_path,
    ).as_of(t0)

    if evidence_path is not None and Path(evidence_path).exists():
        ev = pd.read_parquet(evidence_path)
    else:
        items = build_evidence_stream(
            slug, twitter_path=twitter_path, articles_path=articles_path,
            podcast_path=podcast_path, tracked_people_path=tracked_people_path,
        )
        from dataclasses import asdict
        ev = pd.DataFrame([asdict(e) for e in items])

    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    ev = ev[ev["timestamp"].dt.date <= t0].sort_values("timestamp").reset_index(drop=True)
    return identity, ev
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/soul.py tests/test_doppelganger_soul.py
git commit -m "feat(doppelganger): soul input loading + t0 time-gating"
```

---

## Task 2: Extraction prompt assembly (`build_extraction_prompt`)

**Files:**
- Modify: `doppelganger/soul.py`
- Test: `tests/test_doppelganger_soul.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_doppelganger_soul.py`)

```python
from doppelganger.soul import build_extraction_prompt
from doppelganger.schema import IdentityProfile, Experience


def test_build_extraction_prompt_structure():
    import pandas as pd
    identity = IdentityProfile(
        slug="testy-mctest", name="Testy McTest", headline="Investing.", bio="A GP.",
        current_role="Engineer", experience=[Experience("Engineer", "Beta", None, None, None)],
        education=[], socials={},
    )
    ev = pd.DataFrame([
        {"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives.",
         "attribution_confidence": 1.0, "context": None},
    ])
    system, user = build_extraction_prompt(identity, ev)
    # system instructions name the required sections and the citation format
    for section in ["How He Thinks", "What He Believes", "What He Attends To",
                    "Open Contradictions", "How He Talks", "Bio Lens"]:
        assert section in system
    assert "[2022-06-01]" not in system          # the date format is described, not pre-filled
    assert '[<YYYY-MM-DD>]' in system or "YYYY-MM-DD" in system
    # user content carries the identity and every evidence item with its date + text
    assert "Testy McTest" in user and "A GP." in user
    assert "2022-06-01" in user and "Tokens align incentives." in user
    assert "x_original" in user
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py::test_build_extraction_prompt_structure -v`
Expected: FAIL — `ImportError: cannot import name 'build_extraction_prompt'`.

- [ ] **Step 3: Write minimal implementation** (append to `doppelganger/soul.py`)

```python
SECTIONS = [
    "Bio Lens", "How He Thinks", "What He Believes",
    "What He Attends To", "Open Contradictions", "How He Talks",
]

_INSTRUCTIONS = f"""You are an expert analyst building a CHARACTERIZATION of a person \
from their own words, to be used as a lens for reconstructing their market views. \
You are given the subject's bio and the complete record of what they said up to a cutoff date.

Write a Markdown "soul card" with EXACTLY these H2 sections, in this order \
(views first; voice last):

1. ## Bio Lens — how their background shapes their analytical lens. Note where the \
corpus CONFIRMS vs OVERRIDES what the bio alone would predict.
2. ## How He Thinks — their reasoning *moves*, epistemic style (how they update, hedge, \
calibrate certainty), and named frameworks / mental models. This is the priority section.
3. ## What He Believes — durable, RECURRING convictions only (stable across the whole span). \
Exclude one-off or volatile positions.
4. ## What He Attends To — what they fixate on vs. dismiss.
5. ## Open Contradictions — genuine tensions in their thinking. Preserve them; never average away.
6. ## How He Talks — brief; lowest priority. Only register/voice that reveals how they think.

RULES:
- Ground EVERY factual claim in the evidence. Immediately after each claim, cite it inline \
in EXACTLY this format: a bracketed ISO date then the verbatim quote in straight double quotes — \
[<YYYY-MM-DD>] "exact quote from the evidence". Keep quotes <= 25 words, verbatim, no ellipsis.
- Use ONLY the provided evidence. Do not use anything you know about this person from outside it. \
Do not reference events after the cutoff.
- Be specific and concrete. Generic statements that could describe any investor are failures.
- Weight solo first-person evidence over co-authored/firm material.

Output ONLY the Markdown soul card (starting with the first ## heading). No preamble."""


def build_extraction_prompt(identity: IdentityProfile, evidence: pd.DataFrame) -> tuple[str, str]:
    """Return (system_instructions, user_content). user_content is piped to claude via stdin."""
    lines = [f"# SUBJECT: {identity.name} ({identity.slug})", ""]
    lines.append("## BIO")
    lines.append(f"Headline: {identity.headline or ''}")
    lines.append(f"Current role (as of cutoff): {identity.current_role or ''}")
    lines.append(f"Bio: {identity.bio or ''}")
    if identity.experience:
        lines.append("Experience: " + "; ".join(
            f"{e.title} @ {e.company}" for e in identity.experience))
    if identity.education:
        lines.append("Education: " + "; ".join(
            f"{e.degree or ''} {e.field or ''} @ {e.school}".strip() for e in identity.education))
    lines += ["", f"## EVIDENCE ({len(evidence)} items, chronological)", ""]
    for _, r in evidence.iterrows():
        d = pd.Timestamp(r["timestamp"]).date().isoformat()
        ctx = r.get("context")
        ctx_s = f" (context: {ctx})" if isinstance(ctx, str) and ctx else ""
        lines.append(f"[{d}] ({r['source_type']}){ctx_s} {r['text']}")
    return _INSTRUCTIONS, "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/soul.py tests/test_doppelganger_soul.py
git commit -m "feat(doppelganger): soul extraction prompt assembly (sections + citation format)"
```

---

## Task 3: `claude -p` subprocess wrapper (`_run_claude`)

**Files:**
- Modify: `doppelganger/soul.py`
- Test: `tests/test_doppelganger_soul.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from unittest.mock import patch, MagicMock
from doppelganger.soul import _run_claude


def test_run_claude_invokes_cli_with_stdin(tmp_path):
    fake = MagicMock(returncode=0, stdout="## Bio Lens\nok\n", stderr="")
    with patch("doppelganger.soul.subprocess.run", return_value=fake) as mrun:
        out = _run_claude("SYS", "USERBODY", workdir=tmp_path)
    assert out == "## Bio Lens\nok"          # stripped
    args, kwargs = mrun.call_args
    cmd = args[0]
    assert cmd[0] == "claude" and "-p" in cmd
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "opus"
    assert "--effort" in cmd and cmd[cmd.index("--effort") + 1] == "max"
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert kwargs["input"] == "USERBODY"      # big payload via stdin, not argv
    assert str(kwargs["cwd"]) == str(tmp_path)


def test_run_claude_raises_on_nonzero():
    fake = MagicMock(returncode=2, stdout="", stderr="boom")
    with patch("doppelganger.soul.subprocess.run", return_value=fake):
        try:
            _run_claude("SYS", "U")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "boom" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -k run_claude -v`
Expected: FAIL — `ImportError: cannot import name '_run_claude'`.

- [ ] **Step 3: Write minimal implementation** (append; add `import subprocess`, `import tempfile` to the top imports of `soul.py`)

```python
import subprocess
import tempfile

CLAUDE_MODEL = "opus"
CLAUDE_EFFORT = "max"
CLAUDE_TIMEOUT_S = 900   # 15 min; soul extraction over a big corpus can be slow


def _run_claude(system: str, user: str, *, workdir: Path | None = None,
                timeout: int = CLAUDE_TIMEOUT_S) -> str:
    """Run `claude -p` (Max subscription, no API cost) from an isolated dir.

    Instructions go via --system-prompt; the large bio+evidence payload via stdin.
    Returns stdout (stripped). Raises RuntimeError on non-zero exit.
    """
    wd = workdir or Path(tempfile.mkdtemp(prefix="doppelganger-soul-"))
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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -k run_claude -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/soul.py tests/test_doppelganger_soul.py
git commit -m "feat(doppelganger): isolated claude -p subprocess wrapper (stdin payload)"
```

---

## Task 4: Orchestrator (`extract_soul`) + frontmatter

**Files:**
- Modify: `doppelganger/soul.py`
- Test: `tests/test_doppelganger_soul.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from datetime import date as _date
from doppelganger.soul import extract_soul


def test_extract_soul_writes_card_with_frontmatter(tmp_path):
    canned = "## Bio Lens\nQuant lens. [2022-06-01] \"Tokens align incentives.\"\n"
    with patch("doppelganger.soul._run_claude", return_value=canned):
        out = extract_soul(
            "testy-mctest", _date(2022, 8, 31), out_dir=tmp_path,
            identity_path=FIX / "linkedin" / "testy-1.json", team_path=FIX / "team.parquet",
            tracked_people_path=FIX / "tracked_people.yaml",
            twitter_path=FIX / "twitter" / "testy.parquet",
            articles_path=FIX / "articles.parquet",
            podcast_path=FIX / "attributed_transcripts.jsonl",
        )
    text = Path(out).read_text()
    assert out == tmp_path / "testy-mctest" / "soul.md"
    assert text.startswith("---\n")                 # YAML frontmatter
    assert "subject: testy-mctest" in text
    assert "t0: 2022-08-31" in text
    assert "evidence_items:" in text
    assert "## Bio Lens" in text and "Tokens align incentives." in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -k extract_soul -v`
Expected: FAIL — `ImportError: cannot import name 'extract_soul'`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
def _frontmatter(slug: str, name: str, t0: date, evidence: pd.DataFrame) -> str:
    if len(evidence):
        span = (f"{pd.Timestamp(evidence['timestamp'].min()).date()}.."
                f"{pd.Timestamp(evidence['timestamp'].max()).date()}")
    else:
        span = ""
    return (
        "---\n"
        f"subject: {slug}\n"
        f"name: {name}\n"
        f"t0: {t0.isoformat()}\n"
        f"built_from:\n"
        f"  evidence_items: {len(evidence)}\n"
        f"  span: \"{span}\"\n"
        f"  model: claude-opus ({CLAUDE_MODEL}/{CLAUDE_EFFORT})\n"
        "---\n\n"
    )


def extract_soul(
    slug: str, t0: date, *,
    out_dir: Path | None = None,
    evidence_path: Path | None = None,
    identity_path: Path | None = None,
    team_path: Path | None = None,
    tracked_people_path: Path | None = None,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
) -> Path:
    identity, evidence = load_soul_inputs(
        slug, t0, evidence_path=evidence_path, identity_path=identity_path,
        team_path=team_path, tracked_people_path=tracked_people_path,
        twitter_path=twitter_path, articles_path=articles_path, podcast_path=podcast_path,
    )
    system, user = build_extraction_prompt(identity, evidence)
    card = _run_claude(system, user)

    base = Path(out_dir or config.OUT_DIR) / slug
    base.mkdir(parents=True, exist_ok=True)
    path = base / "soul.md"
    path.write_text(_frontmatter(slug, identity.name, t0, evidence) + card.strip() + "\n")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -v`
Expected: PASS (all soul tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/soul.py tests/test_doppelganger_soul.py
git commit -m "feat(doppelganger): extract_soul orchestrator writes soul.md with frontmatter"
```

---

## Task 5: Citation parsing (`parse_citations`)

**Files:**
- Create: `doppelganger/soul_audit.py`
- Test: `tests/test_doppelganger_soul_audit.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_soul_audit.py`:
```python
"""TDD tests for doppelganger.soul_audit."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.soul_audit import parse_citations, Citation


def test_parse_citations_extracts_date_and_quote():
    card = (
        "## Bio Lens\n"
        'Quant lens. [2022-06-01] "Tokens align incentives." and more.\n'
        '## How He Thinks\n'
        'Mechanism-first. [2022-07-01] "rollups inherit security"\n'
        "No citation on this sentence.\n"
    )
    cites = parse_citations(card)
    assert cites == [
        Citation(date(2022, 6, 1), "Tokens align incentives."),
        Citation(date(2022, 7, 1), "rollups inherit security"),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.soul_audit'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/soul_audit.py` (this task: header + `Citation` + `parse_citations`):
```python
"""doppelganger.soul_audit — verify a soul card's inline citations against the corpus.

Two failure modes the audit catches:
  - hallucinated: a cited quote matches NO evidence item.
  - leaked: a cited quote's real (or claimed) date is AFTER the soul's t0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

_CITE = re.compile(r'\[(\d{4}-\d{2}-\d{2})\]\s+"([^"]{3,})"')


@dataclass(frozen=True)
class Citation:
    date: date
    quote: str


def parse_citations(card: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _CITE.finditer(card):
        y, mo, d = (int(x) for x in m.group(1).split("-"))
        out.append(Citation(date(y, mo, d), m.group(2).strip()))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul_audit.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/soul_audit.py tests/test_doppelganger_soul_audit.py
git commit -m "feat(doppelganger): soul-card citation parsing"
```

---

## Task 6: The audit (`audit_soul`)

**Files:**
- Modify: `doppelganger/soul_audit.py`
- Test: `tests/test_doppelganger_soul_audit.py`

- [ ] **Step 1: Write the failing test** (append)

```python
import pandas as pd
from doppelganger.soul_audit import audit_soul, AuditReport


def _evidence(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([
        {"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives and that is the core thesis."},
        {"id": "2", "timestamp": pd.Timestamp("2023-03-01", tz="UTC"),
         "source_type": "x_original", "text": "ZK rollups are the endgame for scaling."},
    ]).to_parquet(p)
    return p


def test_audit_passes_clean_card(tmp_path):
    ev = _evidence(tmp_path)
    card = tmp_path / "soul.md"
    card.write_text('## X\nClaim. [2022-06-01] "Tokens align incentives"\n')
    rep = audit_soul(card, ev, date(2022, 12, 31))
    assert rep.ok and rep.checked == 1 and rep.matched == 1
    assert not rep.hallucinated and not rep.leaked


def test_audit_flags_hallucinated(tmp_path):
    ev = _evidence(tmp_path)
    card = tmp_path / "soul.md"
    card.write_text('## X\nClaim. [2022-06-01] "this quote does not exist anywhere"\n')
    rep = audit_soul(card, ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.hallucinated) == 1


def test_audit_flags_leaked_future_quote(tmp_path):
    ev = _evidence(tmp_path)
    card = tmp_path / "soul.md"
    # quote is real but from a 2023 item — leakage past a 2022 t0
    card.write_text('## X\nClaim. [2023-03-01] "ZK rollups are the endgame"\n')
    rep = audit_soul(card, ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.leaked) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul_audit.py -k audit -v`
Expected: FAIL — `ImportError: cannot import name 'audit_soul'`.

- [ ] **Step 3: Write minimal implementation** (append to `doppelganger/soul_audit.py`; add `from pathlib import Path` to imports)

```python
from pathlib import Path


@dataclass
class AuditReport:
    checked: int
    matched: int
    hallucinated: list[Citation]
    leaked: list[Citation]

    @property
    def ok(self) -> bool:
        return not self.hallucinated and not self.leaked


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def audit_soul(card_path: Path, evidence_path: Path, t0: date) -> AuditReport:
    card = Path(card_path).read_text()
    cites = parse_citations(card)

    ev = pd.read_parquet(evidence_path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    norm_items = [(_norm(t), pd.Timestamp(ts).date())
                  for t, ts in zip(ev["text"], ev["timestamp"])]

    matched, hallucinated, leaked = 0, [], []
    for c in cites:
        q = _norm(c.quote)
        hits = [d for text, d in norm_items if q in text]
        if not hits:
            hallucinated.append(c)            # quote matches no evidence item
        elif min(hits) > t0 or c.date > t0:
            leaked.append(c)                  # only matches post-t0 items, or cites a post-t0 date
        else:
            matched += 1
    return AuditReport(len(cites), matched, hallucinated, leaked)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul_audit.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/soul_audit.py tests/test_doppelganger_soul_audit.py
git commit -m "feat(doppelganger): soul audit — grounding + leakage checks"
```

---

## Task 7: CLI `soul` subcommand + README

**Files:**
- Modify: `doppelganger/run.py`, `doppelganger/README.md`
- Test: `tests/test_doppelganger_soul.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_doppelganger_soul.py`)

```python
def test_run_cli_has_soul_subcommand():
    import doppelganger.run as r
    parser = r.build_parser()
    ns = parser.parse_args(["soul", "--subject", "eddy-lazzarin", "--t0", "2022-12-31"])
    assert ns.cmd == "soul" and ns.subject == "eddy-lazzarin" and ns.t0 == "2022-12-31"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -k cli -v`
Expected: FAIL — `AttributeError: module 'doppelganger.run' has no attribute 'build_parser'`.

- [ ] **Step 3: Write minimal implementation**

Replace the body of `doppelganger/run.py` with (refactors `main` to expose `build_parser`, adds `soul`):
```python
"""doppelganger.run — CLI entrypoint.

Usage:
    python -m doppelganger.run ingest --subject eddy-lazzarin
    python -m doppelganger.run soul   --subject eddy-lazzarin --t0 2022-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

from doppelganger.ingest import ingest
from doppelganger.soul import extract_soul


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doppelganger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="build identity + evidence stream for a subject")
    ing.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")

    soul = sub.add_parser("soul", help="build the frozen-at-T0 soul card for a subject")
    soul.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    soul.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "ingest":
        out = ingest(args.subject)
        print(f"wrote {out['evidence']} and {out['identity']}")
    elif args.cmd == "soul":
        path = extract_soul(args.subject, date.fromisoformat(args.t0))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_soul.py -k cli -v`
Expected: PASS.

- [ ] **Step 5: Update `doppelganger/README.md`** — under "Unit 1 — Ingestion", add a new section:

```markdown
## Unit 2 — Soul

Builds the frozen-at-T0 "soul card" — the view-generating characterization the doppelganger
reasons through — via a single `claude -p` pass (Max subscription, no API cost) over the
subject's bio + evidence dated <= T0.

```bash
python -m doppelganger.run soul --subject eddy-lazzarin --t0 2022-12-31
```

Output: `data/doppelganger/<slug>/soul.md` (YAML frontmatter + views-first Markdown sections;
every claim cites a dated quote as `[YYYY-MM-DD] "verbatim"`).

Audit a card (every cited quote must exist in <= T0 evidence and not be time-leaked):

```python
from datetime import date
from doppelganger.soul_audit import audit_soul
rep = audit_soul("data/doppelganger/eddy-lazzarin/soul.md",
                 "data/doppelganger/eddy-lazzarin/evidence.parquet", date(2022, 12, 31))
print(rep.ok, rep.checked, rep.matched, len(rep.hallucinated), len(rep.leaked))
```
```

- [ ] **Step 6: Commit**
```bash
git add doppelganger/run.py doppelganger/README.md tests/test_doppelganger_soul.py
git commit -m "feat(doppelganger): soul CLI subcommand + README"
```

---

## Task 8: Real-data gate — generate + audit + discriminate (not TDD)

Verification against the real corpus. Requires the `claude` CLI logged in to the Max subscription and the Eddy + Ali evidence present (`data/doppelganger/eddy-lazzarin/evidence.parquet` exists from Unit 1; build Ali's first).

- [ ] **Step 1: Confirm full suite is green**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -q`
Expected: all PASS.

- [ ] **Step 2: Ensure Ali Yahya's evidence exists (foil for discrimination)**

Run:
```bash
.venv/bin/python -m doppelganger.run ingest --subject ali-yahya
```
Expected: `wrote data/doppelganger/ali-yahya/evidence.parquet and .../identity.json`. (Ali's X handle `alive_` + research are in the corpus; podcast may be absent — adapter handles it.)

- [ ] **Step 3: Generate both T0 soul cards**

Run:
```bash
.venv/bin/python -m doppelganger.run soul --subject eddy-lazzarin --t0 2022-12-31
.venv/bin/python -m doppelganger.run soul --subject ali-yahya    --t0 2022-12-31
```
Expected: each prints `wrote data/doppelganger/<slug>/soul.md`. (Each is one `claude -p` call; may take a few minutes.)

- [ ] **Step 4: Run the grounding + leakage audit on Eddy's card**

Run:
```bash
.venv/bin/python - <<'PY'
from datetime import date
from doppelganger.soul_audit import audit_soul
rep = audit_soul("data/doppelganger/eddy-lazzarin/soul.md",
                 "data/doppelganger/eddy-lazzarin/evidence.parquet", date(2022, 12, 31))
print(f"ok={rep.ok} checked={rep.checked} matched={rep.matched} "
      f"hallucinated={len(rep.hallucinated)} leaked={len(rep.leaked)}")
for c in rep.hallucinated[:5]: print("  HALLUC:", c)
for c in rep.leaked[:5]: print("  LEAK:", c)
PY
```
Expected/acceptance: `checked` is a healthy number (dozens of cited claims), `matched` is the large majority, **`leaked == 0`**. A few `hallucinated` may indicate the model lightly paraphrased a quote (tune the prompt to quote verbatim) — investigate, but `leaked == 0` is the hard gate. Report the numbers.

- [ ] **Step 5: Discrimination eyeball — read both cards side by side**

Run: `echo "=== EDDY ==="; cat data/doppelganger/eddy-lazzarin/soul.md; echo "=== ALI ==="; cat data/doppelganger/ali-yahya/soul.md`

Judge (the make-or-break, analytical not stylistic): do Eddy and Ali present as **two distinct analytical minds** — different frameworks, priors, things they fixate on — or two generic crypto-VC templates? Write a 3-5 sentence verdict. If they read as templates, the extraction prompt needs sharpening (more pressure toward specific, contrastive, evidence-grounded claims) before this unit is "done".

- [ ] **Step 6: Commit the verdict + any prompt tuning**

If the audit/discrimination prompted a prompt change in `soul.py`, re-run the affected tests and commit:
```bash
git add -A && git commit -m "fix(doppelganger): sharpen soul extraction prompt per real-data gate"
```
Otherwise record the audit numbers + discrimination verdict in the commit body of a notes commit, or report them back to the controller. Note: generated `soul.md` cards under `data/doppelganger/` are NOT committed (regenerate via CLI), consistent with Unit 1's artifact handling.

---

## Self-review

**Spec coverage (soul-unit-design.md):**
- §2 inputs (identity.json as_of, evidence ≤t0) → Task 1. ✓
- §3 soul card: Markdown, YAML frontmatter, fixed views-first sections, inline dated-quote citations → Tasks 2 (sections+citation format in prompt), 4 (frontmatter). ✓
- §4 extraction: single holistic `claude -p --model opus --effort max --no-session-persistence`, isolated dir, stdin payload, bio→corpus instruction, behavior-wins → Tasks 2, 3, 4. ✓
- §5 Gate 1 grounding+leakage audit → Tasks 5, 6, 8.4. ✓ ; Gate 2 discrimination eyeball → Task 8.5. ✓
- §6 modules (soul.py, soul_audit.py, soul CLI) → Tasks 1-7. ✓
- §7 testing: plumbing + audit TDD with mocked LLM; real-data gate → all tasks + Task 8. ✓

**Placeholder scan:** none — every step has runnable code/commands. `claude -p` is the only step needing the live CLI, isolated to Task 8 (real-data gate), explicitly marked non-TDD.

**Type consistency:** `extract_soul`/`load_soul_inputs`/`build_extraction_prompt`/`_run_claude` signatures match across tasks and call sites. `Citation(date, quote)` and `AuditReport(checked, matched, hallucinated, leaked)` used identically in Tasks 5, 6, 8. Citation regex in `parse_citations` matches the format described in the Task 2 prompt instructions (`[YYYY-MM-DD] "quote"`). `build_parser`/`main` in Task 7 match the Task 7 CLI test.

**Note on mocking discipline:** every test monkeypatches `doppelganger.soul.subprocess.run` (Task 3) or `doppelganger.soul._run_claude` (Task 4) — the real model is NEVER called in the test suite. Only Task 8 invokes `claude` for real.

---

## Execution handoff

After this unit: Unit ❸ (Memory/retrieval) is next, then ❹ (Doppelganger), ❺ (Eval), plus the docs/visual explainer.
