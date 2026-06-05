# Corpus Doppelganger — Unit ❶ Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ingestion unit that turns one subject's raw public sources (X, research, speaker-attributed podcasts, bio/LinkedIn) into two clean artifacts: a static, time-truncatable **identity profile** and a normalized, dated **evidence stream**.

**Architecture:** A `doppelganger/` Python package in the `sentience` repo. Each source is an isolated **adapter** that normalizes into a shared `EvidenceItem`. An identity module merges LinkedIn + the a16z bio into an `IdentityProfile` with an `as_of(T)` time-truncation. An orchestrator runs all adapters for a subject and writes `data/doppelganger/<slug>/{evidence.parquet, identity.json}`. Pure data transformation — no network, no LLM (keeps it deterministic and fast to test).

**Tech Stack:** Python 3.13, pandas, pyarrow, pyyaml. pytest (`asyncio_mode=auto` already set). Follows existing repo conventions (top-level package like `market_data/`, `tests/test_<module>_*.py`, `python -m pkg.run` CLI).

**Spec:** `docs/superpowers/specs/2026-06-05-corpus-doppelganger-engine-design.md` (§4 ❶ Ingestion). First subject: `eddy-lazzarin`.

---

## File structure

```
doppelganger/
  __init__.py
  config.py              # paths + tuning constants
  registry.py            # resolve a subject slug → its source locations (from tracked_people.yaml)
  schema.py              # EvidenceItem, Experience, Education, IdentityProfile dataclasses
  identity.py            # parse LinkedIn dates, merge LinkedIn + a16z bio, IdentityProfile.as_of(T)
  adapters/
    __init__.py
    twitter.py           # X parquet → EvidenceItems (reply filter, quote context, self-thread reassembly)
    research.py          # research articles → EvidenceItems (solo vs firm attribution)
    podcast.py           # attributed_transcripts.jsonl → EvidenceItems (subject turns + question context)
  ingest.py              # orchestrate adapters + identity → write artifacts
  run.py                 # CLI: python -m doppelganger.run ingest --subject eddy-lazzarin
  requirements.txt
  README.md

tests/
  fixtures/doppelganger/ # tiny fixture files (built in Task 0)
  test_doppelganger_schema.py
  test_doppelganger_identity.py
  test_doppelganger_twitter.py
  test_doppelganger_research.py
  test_doppelganger_podcast.py
  test_doppelganger_ingest.py
```

**Canonical types (locked — use these exact names/fields in every task):**

```python
EvidenceItem(
    id: str,
    subject: str,                  # subject slug, e.g. "eddy-lazzarin"
    timestamp: datetime,           # tz-aware UTC
    source_type: str,              # x_original | x_quote | x_reply | research | research_firm | podcast
    text: str,
    speaker_slug: str,             # the subject's slug (the utterance's author)
    attribution_confidence: float, # 1.0 solo voice; 0.5 firm voice; segment conf for podcast
    thread_id: str | None = None,
    context: str | None = None,    # quoted tweet / preceding question
    context_missing: bool = False,
    engagement: int | None = None,
)
```

**Evidence parquet column order (locked):** `id, subject, timestamp, source_type, text, speaker_slug, attribution_confidence, thread_id, context, context_missing, engagement`.

---

## Task 0: Package scaffolding, config, and test fixtures

**Files:**
- Create: `doppelganger/__init__.py`, `doppelganger/config.py`, `doppelganger/adapters/__init__.py`
- Create: `doppelganger/requirements.txt`
- Create: `tests/fixtures/doppelganger/` fixtures (below)

- [ ] **Step 1: Create the package files**

`doppelganger/__init__.py`:
```python
"""doppelganger — build a corpus-only digital doppelganger of a person.

Unit 1 (ingestion): raw sources -> identity profile + normalized evidence stream.
"""
```

`doppelganger/adapters/__init__.py`:
```python
"""Source adapters: each normalizes one raw source into EvidenceItem lists."""
```

`doppelganger/config.py`:
```python
"""doppelganger.config — paths and tuning constants for ingestion.

All paths are relative to the sentience repo root (the process CWD when run via
`python -m doppelganger.run`). Override DATA_DIR in tests by passing explicit paths.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")

# --- Source locations ---
TWITTER_DIR = DATA_DIR / "twitter"                              # <x_handle>.parquet
RESEARCH_ARTICLES = DATA_DIR / "a16z_research" / "articles.parquet"
TEAM_PARQUET = DATA_DIR / "a16z_team" / "team.parquet"
LINKEDIN_DIR = DATA_DIR / "linkedin" / "parsed"                 # <linkedin-slug>.json
# Produced by the corpus-attribution session (branch doppelganger/corpus-attribution).
# Must be merged/copied to this path before the podcast adapter runs on real data.
ATTRIBUTED_TRANSCRIPTS = DATA_DIR / "a16z_research" / "attributed_transcripts.jsonl"
TRACKED_PEOPLE = DATA_DIR / "tracked_people.yaml"

# --- Output ---
OUT_DIR = DATA_DIR / "doppelganger"                            # <slug>/evidence.parquet, identity.json

# --- Tuning (eval-tuned later; documented defaults, not unexamined) ---
MIN_REPLY_CONTENT_CHARS = 50    # X replies with less substantive content than this are dropped
PODCAST_MIN_CONFIDENCE = 0.8    # keep podcast segments at/above this attribution confidence

SOURCE_TYPES = {
    "x_original", "x_quote", "x_reply", "research", "research_firm", "podcast",
}
```

`doppelganger/requirements.txt`:
```
pandas>=2.0
pyarrow>=14
pyyaml>=6
```

- [ ] **Step 2: Build the test fixtures**

Run (creates the fixture dir and tiny, hand-written source files the adapter tests read):
```bash
cd /Users/jax/workspaces/ultim8/projects/sentience
mkdir -p tests/fixtures/doppelganger/linkedin tests/fixtures/doppelganger/twitter
.venv/bin/python - <<'PY'
import json, pandas as pd
from pathlib import Path
base = Path("tests/fixtures/doppelganger")

# --- LinkedIn (subject: testy-mctest) ---
(base/"linkedin"/"testy-1.json").write_text(json.dumps({
    "slug": "testy-1", "name": "Testy McTest", "headline": "Investing in things.",
    "location": "United States", "bio": "Investor.",
    "experience": [
        {"title": "General Partner", "company": "Acme", "start": "May 2026", "end": None, "description": None},
        {"title": "CTO", "company": "Acme", "start": "Feb 2023", "end": "May 2026", "description": None},
        {"title": "Engineer", "company": "Beta", "start": "2016", "end": "2018", "description": "Lead."},
    ],
    "education": [
        {"school": "State U", "degree": "BA", "field": "Philosophy", "start": "2006", "end": "2010"},
    ],
}, indent=2))

# --- a16z team bio parquet (one row) ---
pd.DataFrame([{
    "slug": "testy-mctest", "name": "Testy McTest", "title": "General Partner",
    "listing_title": "General Partner", "bio": "Testy is a GP at Acme focused on tokens.",
    "x_url": "https://twitter.com/@testy", "linkedin_url": "https://www.linkedin.com/in/testy-1/",
    "farcaster_url": None, "socials_count": 2,
}]).to_parquet(base/"team.parquet")

# --- Twitter parquet (subject author_id = "999") ---
def tw(id, type, text, created, reply_to_id=None, reply_to_uid=None, quoted=None, rt=None, conv=None, likes=0):
    return dict(id=id, created_at=pd.Timestamp(created, tz="UTC"), type=type, text=text, lang="en",
                author_id="999", author_username="testy", author_name="Testy",
                reply_count=0, retweet_count=0, like_count=likes, quote_count=0, view_count=0, bookmark_count=0,
                conversation_id=conv or id, in_reply_to_id=reply_to_id, in_reply_to_user_id=reply_to_uid,
                in_reply_to_username=None, quoted_id=quoted, retweeted_id=rt, url=f"http://x/{id}", raw_json="{}")
rows = [
    tw("1", "original", "Tokens align incentives. This is the core thesis.", "2022-06-01"),
    # self-thread: root original "2" + self-reply "3"
    tw("2", "original", "Thread on L2s.", "2022-07-01", conv="2"),
    tw("3", "reply", "Continued: rollups inherit security.", "2022-07-01 00:05", reply_to_id="2", reply_to_uid="999", conv="2"),
    # substantive reply to someone else (kept)
    tw("4", "reply", "I disagree — the marginal buyer matters more than the narrative here.", "2022-08-01", reply_to_id="x", reply_to_uid="111"),
    # noise reply (dropped: short, just a mention)
    tw("5", "reply", "@someone lol", "2022-08-02", reply_to_id="y", reply_to_uid="222"),
    # quote with missing quoted tweet (not in corpus)
    tw("6", "quote", "This is exactly right.", "2022-09-01", quoted="55555"),
    # retweet (dropped)
    tw("7", "retweet", "RT @a16z: ...", "2022-09-02", rt="88888"),
]
pd.DataFrame(rows).to_parquet(base/"twitter"/"testy.parquet")

# --- Research articles parquet ---
pd.DataFrame([
    {"object_id": "100-0", "title": "Solo essay on tokens", "post_date": "2022-05-06T00:00:00+00:00",
     "author_slugs": ["testy-mctest"], "formats": ["articles"], "acf_content": "Solo body about token design.",
     "extracted_text": "Solo body about token design."},
    {"object_id": "101-0", "title": "Firm: things we're excited about", "post_date": "2022-12-04T00:00:00+00:00",
     "author_slugs": ["testy-mctest", "other-person"], "formats": ["articles"],
     "acf_content": "Firm body co-signed.", "extracted_text": "Firm body co-signed."},
    {"object_id": "102-0", "title": "Not the subject", "post_date": "2022-01-01T00:00:00+00:00",
     "author_slugs": ["someone-else"], "formats": ["articles"], "acf_content": "x", "extracted_text": "x"},
]).to_parquet(base/"articles.parquet")

# --- Attributed transcripts jsonl ---
(base/"attributed_transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in [
    {"object_id": "200-0", "title": "Tokencraft", "post_date": "2022-08-08T05:55:23+00:00", "format": "videos",
     "permalink": "http://x", "categories": [], "tags": [], "credited_authors": ["testy-mctest"],
     "a16z_participants": ["testy-mctest"], "all_speakers": ["Testy McTest", "AUDIENCE"],
     "segments": [
        {"idx": 0, "speaker": "Testy McTest", "slug": "testy-mctest", "is_a16z": True, "confidence": 0.99, "kept": True, "text": "Who here may issue a token?"},
        {"idx": 1, "speaker": "AUDIENCE", "slug": None, "is_a16z": False, "confidence": 0.6, "kept": False, "text": "What about points?"},
        {"idx": 2, "speaker": "Testy McTest", "slug": "testy-mctest", "is_a16z": True, "confidence": 0.98, "kept": True, "text": "Points are a balance in a database until transferable."},
        {"idx": 3, "speaker": "Testy McTest", "slug": "testy-mctest", "is_a16z": True, "confidence": 0.5, "kept": True, "text": "low-confidence turn that should be dropped"},
     ]},
    {"object_id": "201-0", "title": "Not the subject's podcast", "post_date": "2022-09-09T00:00:00+00:00", "format": "podcasts",
     "permalink": "http://y", "categories": [], "tags": [], "credited_authors": ["other"],
     "a16z_participants": ["other-person"], "all_speakers": ["Other"],
     "segments": [{"idx": 0, "speaker": "Other", "slug": "other-person", "is_a16z": True, "confidence": 0.99, "kept": True, "text": "not testy"}]},
]) + "\n")

# --- tracked_people.yaml (fixture) ---
(base/"tracked_people.yaml").write_text(
"""people:
- slug: testy-mctest
  name: Testy McTest
  x_handle: testy
  linkedin_url: https://www.linkedin.com/in/testy-1/
""")
print("fixtures written:", sorted(p.name for p in base.rglob("*") if p.is_file()))
PY
```
Expected: prints the fixture filenames (articles.parquet, attributed_transcripts.jsonl, team.parquet, testy-1.json, testy.parquet, tracked_people.yaml).

- [ ] **Step 3: Commit**
```bash
git add doppelganger/ tests/fixtures/doppelganger/
git commit -m "feat(doppelganger): ingestion package scaffold + test fixtures"
```

---

## Task 1: Shared schema (`schema.py`)

**Files:**
- Create: `doppelganger/schema.py`
- Test: `tests/test_doppelganger_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_schema.py`:
```python
"""TDD tests for doppelganger.schema."""
from __future__ import annotations

from datetime import date, datetime, timezone

from doppelganger.schema import EvidenceItem, Experience, Education, IdentityProfile


def test_evidence_item_defaults():
    e = EvidenceItem(
        id="1", subject="testy-mctest", timestamp=datetime(2022, 6, 1, tzinfo=timezone.utc),
        source_type="x_original", text="hi", speaker_slug="testy-mctest", attribution_confidence=1.0,
    )
    assert e.thread_id is None and e.context is None and e.context_missing is False and e.engagement is None


def test_identity_as_of_truncates_experience_and_education():
    prof = IdentityProfile(
        slug="testy-mctest", name="Testy", headline=None, bio=None, current_role=None,
        experience=[
            Experience("GP", "Acme", date(2026, 5, 1), None, None),
            Experience("CTO", "Acme", date(2023, 2, 1), date(2026, 5, 1), None),
            Experience("Engineer", "Beta", date(2016, 1, 1), date(2018, 1, 1), "Lead."),
        ],
        education=[Education("State U", "BA", "Philosophy", date(2006, 1, 1), date(2010, 1, 1))],
        socials={},
    )
    at = prof.as_of(date(2022, 12, 31))
    # GP (2026) and CTO (2023) are in the future relative to 2022 -> dropped
    assert [x.title for x in at.experience] == ["Engineer"]
    # current role at 2022-12-31: most recent experience started on/before then = Engineer ended 2018,
    # but nothing active; current_role is the latest-started role with start<=T -> "Engineer"
    assert at.current_role == "Engineer"
    assert [x.school for x in at.education] == ["State U"]


def test_identity_as_of_picks_active_role():
    prof = IdentityProfile(
        slug="s", name="N", headline=None, bio=None, current_role=None,
        experience=[
            Experience("CTO", "Acme", date(2023, 2, 1), date(2026, 5, 1), None),
            Experience("Engineer", "Beta", date(2016, 1, 1), date(2018, 1, 1), None),
        ],
        education=[],
        socials={},
    )
    at = prof.as_of(date(2024, 6, 1))
    # CTO active 2023-2026 covers 2024 -> current role
    assert at.current_role == "CTO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.schema'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/schema.py`:
```python
"""doppelganger.schema — shared data types for ingestion artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    subject: str
    timestamp: datetime           # tz-aware UTC
    source_type: str
    text: str
    speaker_slug: str
    attribution_confidence: float
    thread_id: str | None = None
    context: str | None = None
    context_missing: bool = False
    engagement: int | None = None


@dataclass
class Experience:
    title: str
    company: str
    start: date | None
    end: date | None
    description: str | None


@dataclass
class Education:
    school: str
    degree: str | None
    field: str | None
    start: date | None
    end: date | None


@dataclass
class IdentityProfile:
    slug: str
    name: str
    headline: str | None
    bio: str | None
    current_role: str | None
    experience: list[Experience] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    socials: dict[str, str] = field(default_factory=dict)

    def as_of(self, t: date) -> "IdentityProfile":
        """Return a copy truncated to what was true on/before date t."""
        exp = [e for e in self.experience if e.start is None or e.start <= t]
        edu = [e for e in self.education if e.start is None or e.start <= t]
        # current role: prefer a role active at t (start<=t and (end is None or end>t));
        # else the latest-started role with start<=t.
        active = [e for e in exp if (e.end is None or e.end > t)]
        pick = active or exp
        current = max(pick, key=lambda e: e.start or date.min).title if pick else None
        return replace(self, experience=exp, education=edu, current_role=current)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/schema.py tests/test_doppelganger_schema.py
git commit -m "feat(doppelganger): EvidenceItem + IdentityProfile schema with as_of() truncation"
```

---

## Task 2: Identity merge (`identity.py`)

**Files:**
- Create: `doppelganger/registry.py`, `doppelganger/identity.py`
- Test: `tests/test_doppelganger_identity.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_identity.py`:
```python
"""TDD tests for doppelganger.identity + registry."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.identity import parse_li_date, build_identity
from doppelganger.registry import resolve_subject

FIX = Path("tests/fixtures/doppelganger")


def test_parse_li_date():
    assert parse_li_date("May 2026") == date(2026, 5, 1)
    assert parse_li_date("2008") == date(2008, 1, 1)
    assert parse_li_date(None) is None
    assert parse_li_date("") is None


def test_resolve_subject_reads_registry():
    s = resolve_subject("testy-mctest", tracked_people_path=FIX / "tracked_people.yaml")
    assert s.slug == "testy-mctest"
    assert s.x_handle == "testy"
    assert s.linkedin_file == "testy-1.json"   # derived from linkedin_url trailing segment


def test_build_identity_merges_linkedin_and_bio():
    prof = build_identity(
        "testy-mctest",
        linkedin_path=FIX / "linkedin" / "testy-1.json",
        team_path=FIX / "team.parquet",
    )
    assert prof.name == "Testy McTest"
    assert prof.headline == "Investing in things."          # from LinkedIn
    assert "GP at Acme" in (prof.bio or "")                  # a16z bio merged in
    assert len(prof.experience) == 3 and prof.experience[0].title == "General Partner"
    assert prof.experience[0].start == date(2026, 5, 1)     # "May 2026" parsed
    assert prof.socials.get("x_url") == "https://twitter.com/@testy"
    # time-gate: as-of EOY 2022 drops the 2023/2026 roles
    at = prof.as_of(date(2022, 12, 31))
    assert [e.title for e in at.experience] == ["Engineer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.identity'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/registry.py`:
```python
"""doppelganger.registry — resolve a subject slug to its source locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from doppelganger import config


@dataclass
class SubjectRef:
    slug: str
    name: str
    x_handle: str | None
    linkedin_url: str | None
    linkedin_file: str | None     # "<segment>.json" or None


def _linkedin_file(url: str | None) -> str | None:
    if not url:
        return None
    seg = url.rstrip("/").split("/")[-1]
    return f"{seg}.json" if seg else None


def resolve_subject(slug: str, *, tracked_people_path: Path | None = None) -> SubjectRef:
    path = tracked_people_path or config.TRACKED_PEOPLE
    people = yaml.safe_load(Path(path).read_text())["people"]
    for p in people:
        if p["slug"] == slug:
            return SubjectRef(
                slug=slug, name=p.get("name", slug), x_handle=p.get("x_handle"),
                linkedin_url=p.get("linkedin_url"), linkedin_file=_linkedin_file(p.get("linkedin_url")),
            )
    raise KeyError(f"subject {slug!r} not found in {path}")
```

`doppelganger/identity.py`:
```python
"""doppelganger.identity — merge LinkedIn + a16z bio into an IdentityProfile."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.registry import resolve_subject
from doppelganger.schema import Education, Experience, IdentityProfile

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_li_date(s: str | None) -> date | None:
    """Parse LinkedIn date strings: 'May 2026', '2008', None/'' -> date|None (day=1)."""
    if not s:
        return None
    parts = s.strip().split()
    if len(parts) == 2 and parts[0][:3].lower() in _MONTHS:
        return date(int(parts[1]), _MONTHS[parts[0][:3].lower()], 1)
    if len(parts) == 1 and parts[0].isdigit():
        return date(int(parts[0]), 1, 1)
    return None


def build_identity(
    slug: str,
    *,
    linkedin_path: Path | None = None,
    team_path: Path | None = None,
    tracked_people_path: Path | None = None,
) -> IdentityProfile:
    ref = resolve_subject(slug, tracked_people_path=tracked_people_path)

    li_path = linkedin_path or (config.LINKEDIN_DIR / (ref.linkedin_file or ""))
    li = json.loads(Path(li_path).read_text()) if Path(li_path).exists() else {}

    team = pd.read_parquet(team_path or config.TEAM_PARQUET)
    row = team[team["slug"] == slug]
    bio_a16z = str(row.iloc[0]["bio"]) if len(row) else None
    socials = {}
    if len(row):
        for col in ("x_url", "linkedin_url", "farcaster_url"):
            v = row.iloc[0].get(col)
            if isinstance(v, str) and v:
                socials[col] = v

    # bio: prefer the richer a16z bio, fall back to LinkedIn bio
    bio = bio_a16z or li.get("bio")

    experience = [
        Experience(e.get("title", ""), e.get("company", ""),
                   parse_li_date(e.get("start")), parse_li_date(e.get("end")), e.get("description"))
        for e in li.get("experience", [])
    ]
    education = [
        Education(e.get("school", ""), e.get("degree"), e.get("field"),
                  parse_li_date(e.get("start")), parse_li_date(e.get("end")))
        for e in li.get("education", [])
    ]

    return IdentityProfile(
        slug=slug, name=li.get("name") or ref.name, headline=li.get("headline"),
        bio=bio, current_role=None, experience=experience, education=education, socials=socials,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_identity.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/registry.py doppelganger/identity.py tests/test_doppelganger_identity.py
git commit -m "feat(doppelganger): identity merge (LinkedIn + a16z bio) with subject registry"
```

---

## Task 3: Twitter adapter (`adapters/twitter.py`)

Handles: drop retweets; filter low-substance replies; attach quote context (flag missing); reassemble self-threads into one item.

**Files:**
- Create: `doppelganger/adapters/twitter.py`
- Test: `tests/test_doppelganger_twitter.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_twitter.py`:
```python
"""TDD tests for doppelganger.adapters.twitter."""
from __future__ import annotations

from pathlib import Path

from doppelganger.adapters.twitter import load_twitter

FIX = Path("tests/fixtures/doppelganger/twitter/testy.parquet")


def _by_id(items):
    return {e.id: e for e in items}


def test_drops_retweets_and_noise_replies():
    items = load_twitter(FIX, "testy-mctest")
    ids = {e.id for e in items}
    assert "7" not in ids        # retweet dropped
    assert "5" not in ids        # "@someone lol" noise reply dropped


def test_keeps_substantive_reply_to_other():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    assert "4" in items
    assert items["4"].source_type == "x_reply"


def test_self_thread_reassembled():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    # root "2" absorbs self-reply "3"; "3" is not its own item
    assert "3" not in items
    assert items["2"].source_type == "x_original"
    assert "rollups inherit security" in items["2"].text
    assert "Thread on L2s" in items["2"].text


def test_quote_context_missing_flag():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    assert items["6"].source_type == "x_quote"
    assert items["6"].context_missing is True   # quoted tweet 55555 not in corpus


def test_fields_populated():
    items = _by_id(load_twitter(FIX, "testy-mctest"))
    e = items["1"]
    assert e.subject == "testy-mctest" and e.speaker_slug == "testy-mctest"
    assert e.attribution_confidence == 1.0
    assert e.timestamp.tzinfo is not None        # tz-aware
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_twitter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.adapters.twitter'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/adapters/twitter.py`:
```python
"""doppelganger.adapters.twitter — X parquet -> EvidenceItems.

Rules (spec §4 ❶): drop retweets; filter low-substance replies; attach quoted-tweet
text as context (flag when the quoted tweet is not in the subject's corpus); reassemble
self-threads (the subject replying to himself) into a single opinion unit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.schema import EvidenceItem

_URL = re.compile(r"https?://\S+")
_LEADING_MENTIONS = re.compile(r"^(?:@\w+\s+)+")
_TYPE_TO_SOURCE = {"original": "x_original", "quote": "x_quote", "reply": "x_reply"}


def _substance(text: str) -> str:
    """Reply text with leading @mentions and URLs stripped, for the noise filter."""
    return _URL.sub("", _LEADING_MENTIONS.sub("", text or "")).strip()


def load_twitter(parquet_path: Path, subject_slug: str) -> list[EvidenceItem]:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return []
    author_id = df["author_id"].iloc[0]
    ids = set(df["id"])
    by_id = {r["id"]: r for _, r in df.iterrows()}

    # children: subject tweet replying to another subject tweet (self-continuation)
    children: dict[str, list[str]] = {}
    is_child: set[str] = set()
    for _, r in df.iterrows():
        if r["type"] == "retweet":
            continue
        parent = r["in_reply_to_id"]
        if r["in_reply_to_user_id"] == author_id and parent in ids:
            children.setdefault(parent, []).append(r["id"])
            is_child.add(r["id"])

    items: list[EvidenceItem] = []
    for _, r in df.iterrows():
        if r["type"] == "retweet" or r["id"] in is_child:
            continue
        # roots that are replies-to-others with no self-continuation get the noise filter
        is_reply_to_other = r["type"] == "reply" and r["in_reply_to_user_id"] != author_id
        has_thread = r["id"] in children
        if is_reply_to_other and not has_thread and len(_substance(r["text"])) < config.MIN_REPLY_CONTENT_CHARS:
            continue

        # assemble text: root + self-reply descendants in chronological order
        chain = [r["id"]]
        stack = list(children.get(r["id"], []))
        while stack:
            cid = stack.pop(0)
            chain.append(cid)
            stack = list(children.get(cid, [])) + stack
        chain_rows = sorted((by_id[c] for c in chain), key=lambda x: x["created_at"])
        text = "\n\n".join(str(cr["text"]) for cr in chain_rows)

        context, context_missing = None, False
        if r["type"] == "quote":
            q = r["quoted_id"]
            if q in by_id:
                context = str(by_id[q]["text"])
            else:
                context_missing = True

        items.append(EvidenceItem(
            id=str(r["id"]), subject=subject_slug,
            timestamp=r["created_at"].to_pydatetime(),
            source_type=_TYPE_TO_SOURCE[r["type"]], text=text, speaker_slug=subject_slug,
            attribution_confidence=1.0, thread_id=str(r["conversation_id"]),
            context=context, context_missing=context_missing,
            engagement=int(r["like_count"]) if pd.notna(r["like_count"]) else None,
        ))
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_twitter.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/adapters/twitter.py tests/test_doppelganger_twitter.py
git commit -m "feat(doppelganger): X adapter — reply filter, quote context, self-thread reassembly"
```

---

## Task 4: Research adapter (`adapters/research.py`)

**Files:**
- Create: `doppelganger/adapters/research.py`
- Test: `tests/test_doppelganger_research.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_research.py`:
```python
"""TDD tests for doppelganger.adapters.research."""
from __future__ import annotations

from pathlib import Path

from doppelganger.adapters.research import load_research

FIX = Path("tests/fixtures/doppelganger/articles.parquet")


def _by_id(items):
    return {e.id: e for e in items}


def test_only_subject_posts():
    items = load_research(FIX, "testy-mctest")
    ids = {e.id for e in items}
    assert ids == {"100-0", "101-0"}     # 102-0 is someone else's


def test_solo_vs_firm_attribution():
    items = _by_id(load_research(FIX, "testy-mctest"))
    assert items["100-0"].source_type == "research" and items["100-0"].attribution_confidence == 1.0
    assert items["101-0"].source_type == "research_firm" and items["101-0"].attribution_confidence == 0.5


def test_timestamp_and_text():
    items = _by_id(load_research(FIX, "testy-mctest"))
    assert items["100-0"].timestamp.year == 2022 and items["100-0"].timestamp.tzinfo is not None
    assert "token design" in items["100-0"].text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_research.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.adapters.research'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/adapters/research.py`:
```python
"""doppelganger.adapters.research — a16z research articles -> EvidenceItems.

Solo-authored posts are high-confidence the subject's voice; co-authored "firm"
posts are co-signed, lower confidence. One item per post (chunking is the memory
unit's concern, not ingestion's).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from doppelganger.schema import EvidenceItem


def _authors(val) -> list[str]:
    if isinstance(val, (list, np.ndarray)):
        return [str(s) for s in val]
    return [] if val is None else [str(val)]


def load_research(articles_path: Path, subject_slug: str) -> list[EvidenceItem]:
    df = pd.read_parquet(articles_path)
    items: list[EvidenceItem] = []
    for _, r in df.iterrows():
        authors = _authors(r["author_slugs"])
        if subject_slug not in authors:
            continue
        text = r.get("acf_content") or r.get("extracted_text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        solo = len(authors) == 1
        ts = pd.to_datetime(r["post_date"], utc=True).to_pydatetime()
        items.append(EvidenceItem(
            id=str(r["object_id"]), subject=subject_slug, timestamp=ts,
            source_type="research" if solo else "research_firm", text=text,
            speaker_slug=subject_slug, attribution_confidence=1.0 if solo else 0.5,
        ))
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_research.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/adapters/research.py tests/test_doppelganger_research.py
git commit -m "feat(doppelganger): research adapter — solo vs firm attribution tagging"
```

---

## Task 5: Podcast adapter (`adapters/podcast.py`)

**Files:**
- Create: `doppelganger/adapters/podcast.py`
- Test: `tests/test_doppelganger_podcast.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_podcast.py`:
```python
"""TDD tests for doppelganger.adapters.podcast."""
from __future__ import annotations

from pathlib import Path

from doppelganger.adapters.podcast import load_podcast

FIX = Path("tests/fixtures/doppelganger/attributed_transcripts.jsonl")


def test_only_subject_turns_above_confidence():
    items = load_podcast(FIX, "testy-mctest")
    texts = [e.text for e in items]
    # both high-confidence subject turns kept; AUDIENCE turn and other-podcast excluded
    assert any("Who here may issue a token" in t for t in texts)
    assert any("balance in a database" in t for t in texts)
    assert all("not testy" not in t for t in texts)
    # low-confidence subject turn (0.5 < 0.8) dropped
    assert all("should be dropped" not in t for t in texts)


def test_preceding_question_attached_as_context():
    items = load_podcast(FIX, "testy-mctest")
    answer = next(e for e in items if "balance in a database" in e.text)
    assert answer.context == "What about points?"   # the AUDIENCE turn just before


def test_fields():
    items = load_podcast(FIX, "testy-mctest")
    e = items[0]
    assert e.source_type == "podcast" and e.speaker_slug == "testy-mctest"
    assert e.timestamp.year == 2022 and e.timestamp.tzinfo is not None
    assert e.id.startswith("200-0:")                # <object_id>:<idx>
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_podcast.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.adapters.podcast'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/adapters/podcast.py`:
```python
"""doppelganger.adapters.podcast — attributed_transcripts.jsonl -> EvidenceItems.

One item per subject turn (slug == subject, confidence >= threshold). The immediately
preceding non-subject turn is attached as context (the question he's answering).
Interlocutor turns are never attributed to the subject.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.schema import EvidenceItem


def load_podcast(jsonl_path: Path, subject_slug: str,
                 min_confidence: float = config.PODCAST_MIN_CONFIDENCE) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for line in Path(jsonl_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if subject_slug not in rec.get("a16z_participants", []):
            continue
        ts = pd.to_datetime(rec["post_date"], utc=True).to_pydatetime()
        segments = rec.get("segments", [])
        for i, seg in enumerate(segments):
            if seg.get("slug") != subject_slug or not seg.get("kept"):
                continue
            if float(seg.get("confidence", 0)) < min_confidence:
                continue
            prev = segments[i - 1] if i > 0 else None
            context = str(prev["text"]) if prev and prev.get("slug") != subject_slug else None
            items.append(EvidenceItem(
                id=f"{rec['object_id']}:{seg['idx']}", subject=subject_slug, timestamp=ts,
                source_type="podcast", text=str(seg["text"]), speaker_slug=subject_slug,
                attribution_confidence=float(seg["confidence"]), thread_id=str(rec["object_id"]),
                context=context,
            ))
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_podcast.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/adapters/podcast.py tests/test_doppelganger_podcast.py
git commit -m "feat(doppelganger): podcast adapter — subject turns + question context"
```

---

## Task 6: Orchestrator + artifacts (`ingest.py`)

**Files:**
- Create: `doppelganger/ingest.py`
- Test: `tests/test_doppelganger_ingest.py`

- [ ] **Step 1: Write the failing test**

`tests/test_doppelganger_ingest.py`:
```python
"""TDD tests for doppelganger.ingest orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from doppelganger.ingest import build_evidence_stream, ingest

FIX = Path("tests/fixtures/doppelganger")
EVIDENCE_COLS = ["id", "subject", "timestamp", "source_type", "text", "speaker_slug",
                 "attribution_confidence", "thread_id", "context", "context_missing", "engagement"]


def _sources():
    return dict(
        twitter_path=FIX / "twitter" / "testy.parquet",
        articles_path=FIX / "articles.parquet",
        podcast_path=FIX / "attributed_transcripts.jsonl",
    )


def test_evidence_stream_merged_and_sorted():
    items = build_evidence_stream("testy-mctest", **_sources())
    # sorted ascending by timestamp
    ts = [e.timestamp for e in items]
    assert ts == sorted(ts)
    # contains items from all three sources
    types = {e.source_type for e in items}
    assert {"x_original", "research", "research_firm", "podcast"} <= types


def test_ingest_writes_artifacts(tmp_path):
    out = ingest(
        "testy-mctest", out_dir=tmp_path,
        linkedin_path=FIX / "linkedin" / "testy-1.json", team_path=FIX / "team.parquet",
        **_sources(),
    )
    ev = pd.read_parquet(out["evidence"])
    assert list(ev.columns) == EVIDENCE_COLS
    assert len(ev) > 0
    ident = json.loads(Path(out["identity"]).read_text())
    assert ident["name"] == "Testy McTest" and "experience" in ident
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'doppelganger.ingest'`.

- [ ] **Step 3: Write minimal implementation**

`doppelganger/ingest.py`:
```python
"""doppelganger.ingest — orchestrate adapters + identity into artifacts.

Outputs (per subject): data/doppelganger/<slug>/evidence.parquet + identity.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.adapters.podcast import load_podcast
from doppelganger.adapters.research import load_research
from doppelganger.adapters.twitter import load_twitter
from doppelganger.identity import build_identity
from doppelganger.registry import resolve_subject
from doppelganger.schema import EvidenceItem

EVIDENCE_COLS = ["id", "subject", "timestamp", "source_type", "text", "speaker_slug",
                 "attribution_confidence", "thread_id", "context", "context_missing", "engagement"]


def build_evidence_stream(
    slug: str, *,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
    tracked_people_path: Path | None = None,
) -> list[EvidenceItem]:
    ref = resolve_subject(slug, tracked_people_path=tracked_people_path)
    items: list[EvidenceItem] = []

    tw = twitter_path or (config.TWITTER_DIR / f"{ref.x_handle}.parquet")
    if Path(tw).exists():
        items += load_twitter(Path(tw), slug)

    art = articles_path or config.RESEARCH_ARTICLES
    if Path(art).exists():
        items += load_research(Path(art), slug)

    pod = podcast_path or config.ATTRIBUTED_TRANSCRIPTS
    if Path(pod).exists():
        items += load_podcast(Path(pod), slug)

    items.sort(key=lambda e: e.timestamp)
    return items


def _evidence_df(items: list[EvidenceItem]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(e) for e in items])
    if df.empty:
        return pd.DataFrame(columns=EVIDENCE_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[EVIDENCE_COLS]


def _identity_json(profile) -> str:
    def _enc(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        raise TypeError(type(o))
    return json.dumps(asdict(profile), default=_enc, indent=2)


def ingest(
    slug: str, *,
    out_dir: Path | None = None,
    linkedin_path: Path | None = None,
    team_path: Path | None = None,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
    tracked_people_path: Path | None = None,
) -> dict[str, Path]:
    items = build_evidence_stream(
        slug, twitter_path=twitter_path, articles_path=articles_path,
        podcast_path=podcast_path, tracked_people_path=tracked_people_path,
    )
    profile = build_identity(
        slug, linkedin_path=linkedin_path, team_path=team_path,
        tracked_people_path=tracked_people_path,
    )

    base = Path(out_dir or config.OUT_DIR) / slug
    base.mkdir(parents=True, exist_ok=True)
    ev_path, id_path = base / "evidence.parquet", base / "identity.json"
    _evidence_df(items).to_parquet(ev_path)
    id_path.write_text(_identity_json(profile))
    return {"evidence": ev_path, "identity": id_path}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_ingest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add doppelganger/ingest.py tests/test_doppelganger_ingest.py
git commit -m "feat(doppelganger): ingestion orchestrator + artifact writers"
```

---

## Task 7: CLI + README

**Files:**
- Create: `doppelganger/run.py`, `doppelganger/README.md`

- [ ] **Step 1: Write `run.py`**

`doppelganger/run.py`:
```python
"""doppelganger.run — CLI entrypoint.

Usage:
    python -m doppelganger.run ingest --subject eddy-lazzarin
"""

from __future__ import annotations

import argparse

from doppelganger.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(prog="doppelganger")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="build identity + evidence stream for a subject")
    ing.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    args = parser.parse_args()

    if args.cmd == "ingest":
        out = ingest(args.subject)
        print(f"wrote {out['evidence']} and {out['identity']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI wiring**

Run: `.venv/bin/python -c "import doppelganger.run as r; print(callable(r.main))"`
Expected: prints `True`.

- [ ] **Step 3: Write `doppelganger/README.md`**

```markdown
# doppelganger

Builds a corpus-only digital doppelganger of a person and uses it as a time-gated
analytical lens. See `docs/superpowers/specs/2026-06-05-corpus-doppelganger-engine-design.md`.

## Unit 1 — Ingestion (this module so far)

Normalizes a subject's raw sources into two artifacts:
- `data/doppelganger/<slug>/identity.json` — merged LinkedIn + a16z bio, time-truncatable.
- `data/doppelganger/<slug>/evidence.parquet` — dated, normalized utterances (X, research, podcast).

```bash
python -m doppelganger.run ingest --subject eddy-lazzarin
```

### Sources (resolved per subject via `data/tracked_people.yaml`)
- **X** — `data/twitter/<x_handle>.parquet` (originals/quotes/substantive replies; retweets dropped; self-threads merged).
- **Research** — `data/a16z_research/articles.parquet` (solo = high confidence; co-authored firm posts flagged).
- **Podcast** — `data/a16z_research/attributed_transcripts.jsonl` (subject's diarized turns ≥ 0.8 confidence; preceding question kept as context). **Produced by the corpus-attribution pipeline — copy that file to this path before ingesting.**
- **Identity** — `data/linkedin/parsed/<slug>.json` + `data/a16z_team/team.parquet`.

### Tuning (`doppelganger/config.py`)
`MIN_REPLY_CONTENT_CHARS=50`, `PODCAST_MIN_CONFIDENCE=0.8` — documented defaults, eval-tuned later.
```

- [ ] **Step 4: Commit**
```bash
git add doppelganger/run.py doppelganger/README.md
git commit -m "feat(doppelganger): ingestion CLI + README"
```

---

## Task 8: Real-data sanity check (Eddy)

Not TDD — a verification gate against the real corpus. Confirms the adapters survive real data and produces numbers to eyeball before building the soul unit on top.

**Prerequisite:** the podcast file must exist at `data/a16z_research/attributed_transcripts.jsonl`. It currently lives on branch `doppelganger/corpus-attribution`. Copy it in (one-off):
```bash
git show doppelganger/corpus-attribution:data/a16z_research/attributed_transcripts.jsonl \
  > data/a16z_research/attributed_transcripts.jsonl 2>/dev/null \
  || echo "attribution file not available yet — podcast source will be skipped (adapter handles absence)"
```

- [ ] **Step 1: Run ingestion on Eddy**

Run:
```bash
.venv/bin/python -m doppelganger.run ingest --subject eddy-lazzarin
```
Expected: `wrote data/doppelganger/eddy-lazzarin/evidence.parquet and data/doppelganger/eddy-lazzarin/identity.json`.

- [ ] **Step 2: Eyeball the artifacts**

Run:
```bash
.venv/bin/python - <<'PY'
import pandas as pd, json
ev = pd.read_parquet("data/doppelganger/eddy-lazzarin/evidence.parquet")
print("evidence rows:", len(ev))
print("by source_type:\n", ev["source_type"].value_counts())
print("date range:", ev["timestamp"].min(), "->", ev["timestamp"].max())
print("nulls in text:", ev["text"].isna().sum(), "| empty text:", (ev["text"].str.len()==0).sum())
ident = json.load(open("data/doppelganger/eddy-lazzarin/identity.json"))
print("identity:", ident["name"], "| experience entries:", len(ident["experience"]))
PY
```
Expected sanity: hundreds–~1k+ X items, ~40 research items (split research/research_firm), podcast items if the attribution file was present; date range ~2019→2026; no empty text; identity with multiple experience entries.

- [ ] **Step 3: Run the full ingestion test suite**

Run: `.venv/bin/python -m pytest tests/test_doppelganger_*.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit any fixture/tuning adjustments discovered**

If real data revealed a needed tweak (e.g. a column name mismatch), fix the adapter, re-run its test, and:
```bash
git add -A && git commit -m "fix(doppelganger): adjust ingestion for real-corpus edge case"
```
If nothing needed adjusting, skip this step.

---

## Self-review

**Spec coverage (§4 ❶ Ingestion):**
- Identity profile, merged LinkedIn + bio, dedup, `identity_as_of(T)` with bio time-gate → Tasks 1, 2. ✓
- Evidence stream uniform schema → Task 1; populated by Tasks 3–5. ✓
- X: originals/quotes kept, replies filtered, self-threads stitched, quotes' context + `context_missing`, retweets dropped, UTC, engagement → Task 3. ✓
- Research: body from acf/extracted, solo vs firm attribution flag/confidence → Task 4. ✓
- Podcast: subject turns by slug, confidence threshold, interlocutor turn as context, episode-dated, `<object_id>:<idx>` ids → Task 5. ✓
- Cross-source: sort by time, monotonic stream → Task 6. ✓ (Near-identical cross-post dedup deferred — noted below.)
- CLI + artifacts at `data/doppelganger/<slug>/` → Tasks 6, 7. ✓

**Deferred within this unit (intentional, not gaps):** near-identical cross-post dedup (a tweet quoting his own essay) — low volume, revisit if the soul/memory units show duplication noise. Paragraph chunking of research bodies belongs to the memory unit, not ingestion (per spec §4 ❸).

**Placeholder scan:** none — every step has runnable code/commands.

**Type consistency:** `EvidenceItem` fields and `EVIDENCE_COLS` order are identical across Tasks 1, 6, and the ingest test. Adapter signatures (`load_twitter`, `load_research`, `load_podcast`) and `ingest()`/`build_evidence_stream()` kwargs match their call sites in Task 6.

---

## Execution handoff

After this plan: the next units get their own plans in dependency order — **❷ Soul**, **❸ Memory/retrieval**, **❹ Doppelganger**, **❺ Eval** — plus the docs/visual explainer deliverable.
