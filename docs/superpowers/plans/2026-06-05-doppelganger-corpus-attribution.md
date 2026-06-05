# Doppelganger Corpus Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the flat a16z transcript corpus into per-person attributed speech corpora (a16z team speakers named, precision-first) ready to feed the doppelganger engine.

**Architecture:** A content-routed dual-engine pipeline. Caption-only talks go through LLM text attribution; podcasts (audio available) additionally go through local pyannote diarization, fused with the LLM pass so method-agreement acts as a confidence gate. Pure-logic units (route, roster, fuse, gate, assemble) are TDD'd fully; model-calling units (LLM attribution, diarization) isolate their testable helpers and validate the model call in the pilot.

**Tech Stack:** Python 3.13, pandas/pyarrow, `claude -p` (Opus 4.8, $0 on Max sub) for LLM attribution, `pyannote.audio` 3.1 (local/MPS, free) + mlx-whisper for diarization, pytest.

**Execution note:** Per the spec's multi-session coordination, run this in an **isolated git worktree** (create via `superpowers:using-git-worktrees` at execution start). Base it on the branch holding the expanded corpus + roster (`scrape/a16z-team` unless merged to `main` by then). The pipeline lives in a new top-level `attribution/` package (sibling to `scrapers/`, `study/`, `market_data/`).

**Scope:** Build the full pipeline and run the **pilot** (seminar host via text-LLM; cached podcasts via both engines). Full-roster scale run is explicitly deferred.

---

## File Structure

```
attribution/
  __init__.py
  config.py          # paths, thresholds, model id, host-prior config
  route.py           # content router: structured | conversational
  roster.py          # load team.parquet; name/alias matching -> slug, is_a16z
  attribute_text.py  # LLM text attribution: chunk, prompt claude -p, parse, merge
  diarize.py         # pyannote diarization + timestamped retranscribe + overlap-assign
  fuse.py            # combine diarization boundaries + LLM names; agreement signal
  gate.py            # confidence/agreement gate: keep | flag | drop
  assemble.py        # per-person corpus builder
  run.py             # orchestrator + report
tests/
  test_route.py
  test_roster.py
  test_attribute_text.py   # pure helpers (chunk/parse/merge) + a golden snippet
  test_diarize.py          # overlap-assign unit + gated pyannote integration
  test_fuse.py
  test_gate.py
  test_assemble.py
```

---

## Task 0: Refresh transcripts against the expanded corpus (prerequisite)

**Files:** none created — runs the existing `scrapers/a16z_transcripts`.

- [ ] **Step 1: Re-run the transcript scraper against the 1,018-post corpus**

Run: `python -m scrapers.a16z_transcripts.run`
Expected: routing line reports ~335 video+podcast posts; resume skips the already-OK 148; new posts fetched. New podcasts get mp3s cached under `data/a16z_research/audio_cache/`.

- [ ] **Step 2: Verify coverage**

Run:
```bash
python -c "import pandas as pd; t=pd.read_parquet('data/a16z_research/transcripts.parquet'); print(len(t),'rows', (t.status=='ok').sum(),'ok'); print(t.groupby('source').size())"
```
Expected: row count ≈ 335, large majority `ok`. If the parallel session already refreshed it, confirm freshness instead and skip the re-run.

- [ ] **Step 3: Commit any data refresh** (only if this session ran it)

```bash
git add data/a16z_research/transcripts.parquet
git commit -m "data: refresh transcripts against expanded 1018-post corpus"
```

---

## Task 1: Package scaffold + config

**Files:**
- Create: `attribution/__init__.py`
- Create: `attribution/config.py`
- Create: `attribution/requirements.txt`

- [ ] **Step 1: Create the package**

`attribution/__init__.py`:
```python
"""Per-person speech attribution for the a16z transcript corpus (doppelganger feed)."""
```

- [ ] **Step 2: Write config**

`attribution/config.py`:
```python
"""Configuration for the attribution pipeline."""
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "a16z_research"
CORPUS = DATA / "articles.parquet"
TRANSCRIPTS = DATA / "transcripts.parquet"
ROSTER = Path(__file__).resolve().parents[1] / "data" / "a16z_team" / "team.parquet"
AUDIO_CACHE = DATA / "audio_cache"

SEGMENTS_OUT = DATA / "attributed_segments.parquet"
PERSONS_DIR = DATA / "persons"
REPORT_OUT = DATA / "attribution_report.md"
TS_CACHE = DATA / "ts_transcripts"  # timestamped whisper re-transcribe (gitignored)

# LLM attribution — claude -p on the Max subscription ($0), Opus for precision.
LLM_MODEL = "opus"          # resolves to claude-opus-4-8
LLM_EFFORT = "max"
CHUNK_CHARS = 24000         # transcript chunk size for the LLM pass
CHUNK_OVERLAP = 1500        # carry-over so a speaker turn is not split blind

# Precision gate
CONF_MIN = 0.7              # min LLM/segment confidence to keep for the clean corpus

# Diarization (pyannote, local)
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_OP = "op://local/huggingface/credential"  # free HF token, read via op
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

# Known-host prior: where the host is unnamed in their own intro, seed resolution.
HOST_PRIOR = {
    "research-seminar": ["tim-roughgarden", "justin-thaler"],
    "podcast": ["sonal-chokshi"],
}
```

- [ ] **Step 3: Write requirements**

`attribution/requirements.txt`:
```
pyannote.audio==3.3.2
mlx-whisper==0.4.3
# claude CLI must be installed and authenticated (Max subscription).
# Free HuggingFace token (accept pyannote/speaker-diarization-3.1 license) in 1Password.
```

- [ ] **Step 4: Commit**

```bash
git add attribution/__init__.py attribution/config.py attribution/requirements.txt
git commit -m "feat(attribution): scaffold package + config"
```

---

## Task 2: Content router (`route.py`)

**Files:**
- Create: `attribution/route.py`
- Test: `tests/test_route.py`

- [ ] **Step 1: Write the failing test**

`tests/test_route.py`:
```python
import pandas as pd
from attribution.route import route_post, STRUCTURED, CONVERSATIONAL

def _row(fmt, n_auth=1, title="t"):
    return pd.Series({"formats": [fmt], "author_slugs": ["a"] * n_auth, "title": title})

def test_podcast_is_conversational():
    r = route_post(_row("podcasts"))
    assert r.mode == CONVERSATIONAL and r.has_audio is True

def test_video_is_structured():
    r = route_post(_row("videos"))
    assert r.mode == STRUCTURED and r.has_audio is False

def test_multi_author_video_flagged_as_panel_candidate():
    r = route_post(_row("videos", n_auth=3))
    assert r.mode == STRUCTURED and r.panel_candidate is True

def test_single_author_video_not_panel():
    assert route_post(_row("videos", n_auth=1)).panel_candidate is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_route.py -q`
Expected: FAIL (`ModuleNotFoundError: attribution.route`).

- [ ] **Step 3: Implement**

`attribution/route.py`:
```python
"""Route each post to an attribution mode."""
from dataclasses import dataclass

STRUCTURED = "structured"        # caption-only talk -> text-LLM only
CONVERSATIONAL = "conversational"  # podcast (audio) -> both engines

@dataclass
class Route:
    mode: str
    has_audio: bool
    panel_candidate: bool  # multi-voice video: future audio-diarization candidate

def _fmt(row):
    try:
        seq = list(row.get("formats"))
        return str(seq[0]) if seq else None
    except TypeError:
        return None

def _n_auth(row):
    try:
        return len([x for x in list(row.get("author_slugs")) if x])
    except TypeError:
        return 0

def route_post(row) -> Route:
    fmt = _fmt(row)
    if fmt == "podcasts":
        return Route(CONVERSATIONAL, True, False)
    # videos (and anything else with a transcript) are caption-only/structured in v1.
    return Route(STRUCTURED, False, _n_auth(row) >= 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_route.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add attribution/route.py tests/test_route.py
git commit -m "feat(attribution): content router with tests"
```

---

## Task 3: Roster resolution (`roster.py`)

**Files:**
- Create: `attribution/roster.py`
- Test: `tests/test_roster.py`

- [ ] **Step 1: Write the failing test**

`tests/test_roster.py`:
```python
import pandas as pd
from attribution.roster import Roster

def _roster():
    df = pd.DataFrame([
        {"slug": "tim-roughgarden", "name": "Tim Roughgarden", "title": "Head of Research"},
        {"slug": "chris-dixon", "name": "Chris Dixon", "title": "Managing Partner"},
        {"slug": "ali-yahya", "name": "Ali Yahya", "title": "General Partner"},
    ])
    return Roster(df)

def test_exact_match_is_a16z():
    m = _roster().resolve("Tim Roughgarden")
    assert m.slug == "tim-roughgarden" and m.is_a16z is True

def test_last_name_match():
    assert _roster().resolve("Roughgarden").slug == "tim-roughgarden"

def test_case_insensitive():
    assert _roster().resolve("chris dixon").slug == "chris-dixon"

def test_unknown_is_external_guest():
    m = _roster().resolve("Ron Rothblum")
    assert m.slug is None and m.is_a16z is False

def test_ambiguous_last_name_returns_none_match():
    df = pd.DataFrame([
        {"slug": "a-zhang", "name": "Jeremy Zhang", "title": "x"},
        {"slug": "b-zhang", "name": "Michael Zhu", "title": "y"},
    ])
    # two different people; "Zhang" matches one, but test a genuinely ambiguous surname:
    df2 = pd.DataFrame([
        {"slug": "j-lee", "name": "Jay Lee", "title": "x"},
        {"slug": "k-lee", "name": "Kim Lee", "title": "y"},
    ])
    m = Roster(df2).resolve("Lee")
    assert m.slug is None and m.ambiguous is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_roster.py -q`
Expected: FAIL (`ModuleNotFoundError: attribution.roster`).

- [ ] **Step 3: Implement**

`attribution/roster.py`:
```python
"""Resolve a spoken/inferred name to an a16z roster entry."""
from dataclasses import dataclass
import pandas as pd
from .config import ROSTER

@dataclass
class Match:
    slug: str | None
    name: str | None
    title: str | None
    is_a16z: bool
    ambiguous: bool = False

class Roster:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._df["name_l"] = self._df["name"].str.lower().str.strip()
        self._df["last"] = self._df["name_l"].str.split().str[-1]

    @classmethod
    def load(cls) -> "Roster":
        return cls(pd.read_parquet(ROSTER, columns=["slug", "name", "title"]))

    def resolve(self, name: str | None) -> Match:
        if not name or not name.strip():
            return Match(None, None, None, False)
        q = name.lower().strip()
        exact = self._df[self._df["name_l"] == q]
        if len(exact) == 1:
            r = exact.iloc[0]
            return Match(r["slug"], r["name"], r["title"], True)
        last = q.split()[-1]
        bylast = self._df[self._df["last"] == last]
        if len(bylast) == 1:
            r = bylast.iloc[0]
            return Match(r["slug"], r["name"], r["title"], True)
        if len(bylast) > 1:
            return Match(None, None, None, False, ambiguous=True)
        return Match(None, None, None, False)  # external guest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_roster.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add attribution/roster.py tests/test_roster.py
git commit -m "feat(attribution): roster resolution with tests"
```

---

## Task 4: LLM text attribution (`attribute_text.py`)

Pure helpers (`chunk_transcript`, `parse_segments`, `merge_chunks`) are TDD'd; the `claude -p`
call is a thin wrapper validated in the pilot.

**Files:**
- Create: `attribution/attribute_text.py`
- Test: `tests/test_attribute_text.py`

- [ ] **Step 1: Write the failing test**

`tests/test_attribute_text.py`:
```python
import json
from attribution.attribute_text import chunk_transcript, parse_segments, merge_chunks

def test_chunk_respects_size_and_overlap():
    text = "x" * 50000
    chunks = chunk_transcript(text, size=20000, overlap=1000)
    assert len(chunks) == 3
    assert chunks[0]["end"] == 20000
    assert chunks[1]["start"] == 19000  # overlap pulled back

def test_parse_segments_reads_llm_json():
    raw = '```json\n{"segments":[{"speaker":"Tim Roughgarden","text":"Welcome.","confidence":0.95}]}\n```'
    segs = parse_segments(raw)
    assert segs[0]["speaker"] == "Tim Roughgarden" and segs[0]["confidence"] == 0.95

def test_parse_segments_raises_on_garbage():
    import pytest
    with pytest.raises(ValueError):
        parse_segments("not json at all")

def test_merge_chunks_dedupes_overlap_by_text():
    a = [{"speaker": "A", "text": "hello world", "confidence": 0.9}]
    b = [{"speaker": "A", "text": "hello world", "confidence": 0.9},
         {"speaker": "B", "text": "goodbye", "confidence": 0.8}]
    merged = merge_chunks([a, b])
    assert [s["text"] for s in merged] == ["hello world", "goodbye"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_attribute_text.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`attribution/attribute_text.py`:
```python
"""LLM text attribution via `claude -p` (Opus, $0 on Max). Pure helpers are unit-tested;
the model call is validated in the pilot."""
import json
import re
import subprocess

from .config import CHUNK_CHARS, CHUNK_OVERLAP, LLM_EFFORT, LLM_MODEL

def chunk_transcript(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    """Split into overlapping char windows so a turn isn't blindly cut."""
    if len(text) <= size:
        return [{"start": 0, "end": len(text), "text": text}]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append({"start": start, "end": end, "text": text[start:end]})
        if end == len(text):
            break
        start = end - overlap
    return out

def parse_segments(raw: str):
    """Extract the JSON object from an LLM reply (tolerates ```json fences)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in LLM output")
    obj = json.loads(m.group(0))
    return obj["segments"]

def merge_chunks(chunk_results):
    """Concatenate per-chunk segments, dropping overlap duplicates (same speaker+text)."""
    merged, seen = [], set()
    for segs in chunk_results:
        for s in segs:
            key = (s["speaker"], s["text"].strip())
            if key in seen:
                continue
            seen.add(key)
            merged.append(s)
    return merged

PROMPT = """You are segmenting a transcript by speaker. Known participants: {participants}.
Return ONLY JSON: {{"segments":[{{"speaker": <name or "HOST"/"GUEST_1">, "text": <verbatim>,
"confidence": <0-1>}}]}}. Use a real name only when the transcript makes it unambiguous
(introductions, thank-yous). Otherwise use stable HOST/GUEST_n labels. Be conservative:
when unsure who is speaking, lower the confidence.

TRANSCRIPT:
{chunk}"""

def attribute(text: str, participants: list[str]) -> list[dict]:
    """Run the LLM over each chunk and merge. Returns segment dicts."""
    results = []
    for ch in chunk_transcript(text):
        prompt = PROMPT.format(participants=", ".join(participants) or "unknown", chunk=ch["text"])
        out = subprocess.run(
            ["claude", "-p", "--model", LLM_MODEL, "--effort", LLM_EFFORT,
             "--no-session-persistence"],
            input=prompt, text=True, capture_output=True, check=True).stdout
        results.append(parse_segments(out))
    return merge_chunks(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_attribute_text.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add attribution/attribute_text.py tests/test_attribute_text.py
git commit -m "feat(attribution): LLM text attribution + pure-helper tests"
```

---

## Task 5: Diarization (`diarize.py`)

The overlap-assignment logic is TDD'd; the pyannote/whisper calls are a gated integration test.

**Files:**
- Create: `attribution/diarize.py`
- Test: `tests/test_diarize.py`

- [ ] **Step 1: Write the failing test**

`tests/test_diarize.py`:
```python
import os
import pytest
from attribution.diarize import assign_segments_to_turns

def test_assign_segment_to_max_overlap_turn():
    turns = [(0.0, 5.0, "S0"), (5.0, 10.0, "S1")]
    segs = [{"start": 0.5, "end": 4.0, "text": "a"}, {"start": 6.0, "end": 9.0, "text": "b"}]
    out = assign_segments_to_turns(segs, turns)
    assert [s["voice"] for s in out] == ["S0", "S1"]

def test_segment_spanning_boundary_takes_larger_overlap():
    turns = [(0.0, 5.0, "S0"), (5.0, 10.0, "S1")]
    segs = [{"start": 4.0, "end": 9.0, "text": "x"}]  # 1s in S0, 4s in S1
    assert assign_segments_to_turns(segs, turns)[0]["voice"] == "S1"

@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF token")
def test_pyannote_runs_on_cached_podcast():
    from attribution.diarize import diarize_audio
    import glob
    mp3 = glob.glob("data/a16z_research/audio_cache/*.mp3")
    assert mp3, "no cached podcast audio"
    turns = diarize_audio(mp3[0])
    assert len(turns) >= 2 and all(len(t) == 3 for t in turns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diarize.py -q`
Expected: FAIL (`ModuleNotFoundError`). The gated test is skipped.

- [ ] **Step 3: Implement**

`attribution/diarize.py`:
```python
"""Local pyannote diarization + map timestamped whisper segments to voice turns."""
import subprocess

from .config import HF_TOKEN_OP, PYANNOTE_MODEL, WHISPER_MODEL

def assign_segments_to_turns(segments, turns):
    """Tag each whisper segment with the diarization turn it overlaps most."""
    out = []
    for s in segments:
        best, best_ov = None, 0.0
        for (t0, t1, label) in turns:
            ov = max(0.0, min(s["end"], t1) - max(s["start"], t0))
            if ov > best_ov:
                best, best_ov = label, ov
        out.append({**s, "voice": best})
    return out

def _hf_token() -> str:
    return subprocess.check_output(["op", "read", HF_TOKEN_OP], text=True).strip()

def diarize_audio(mp3_path: str):
    """Return [(start, end, voice_label)] from pyannote (local)."""
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=_hf_token())
    diar = pipe(mp3_path)
    return [(seg.start, seg.end, label) for seg, _, label in diar.itertracks(yield_label=True)]

def transcribe_timestamped(mp3_path: str):
    """Whisper segments WITH timestamps (we need them to align to diarization)."""
    import mlx_whisper
    res = mlx_whisper.transcribe(mp3_path, path_or_hf_repo=WHISPER_MODEL)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in res.get("segments", [])]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diarize.py -q`
Expected: PASS (2 passed, 1 skipped).

- [ ] **Step 5: Commit**

```bash
git add attribution/diarize.py tests/test_diarize.py
git commit -m "feat(attribution): diarization + overlap-assign (gated integration)"
```

---

## Task 6: Fusion (`fuse.py`)

**Files:**
- Create: `attribution/fuse.py`
- Test: `tests/test_fuse.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fuse.py`:
```python
from attribution.fuse import name_voices, fuse_segments

def test_name_voices_maps_label_to_resolved_name():
    # LLM said the host (S0) is Sonal; mapping derived from majority vote per voice.
    voice_segs = [{"voice": "S0", "text": "welcome"}, {"voice": "S0", "text": "next"},
                  {"voice": "S1", "text": "thanks"}]
    llm_by_text = {"welcome": "Sonal Chokshi", "next": "Sonal Chokshi", "thanks": "GUEST_1"}
    assert name_voices(voice_segs, llm_by_text) == {"S0": "Sonal Chokshi", "S1": "GUEST_1"}

def test_fuse_marks_agreement_high_conf():
    voice_segs = [{"voice": "S0", "text": "welcome", "start": 0, "end": 2}]
    voice_names = {"S0": "Sonal Chokshi"}
    llm_by_text = {"welcome": "Sonal Chokshi"}
    out = fuse_segments(voice_segs, voice_names, llm_by_text)
    assert out[0]["speaker"] == "Sonal Chokshi" and out[0]["agreed"] is True

def test_fuse_marks_disagreement_low_conf():
    voice_segs = [{"voice": "S0", "text": "welcome", "start": 0, "end": 2}]
    voice_names = {"S0": "Sonal Chokshi"}
    llm_by_text = {"welcome": "Chris Dixon"}  # LLM disagrees with diarization grouping
    out = fuse_segments(voice_segs, voice_names, llm_by_text)
    assert out[0]["agreed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fuse.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`attribution/fuse.py`:
```python
"""Fuse diarization (boundaries+voices) with LLM attribution (names) and cross-check."""
from collections import Counter

def name_voices(voice_segs, llm_by_text):
    """Assign each anonymous voice label its majority LLM-resolved name."""
    votes = {}
    for s in voice_segs:
        nm = llm_by_text.get(s["text"])
        if nm:
            votes.setdefault(s["voice"], Counter())[nm] += 1
    return {v: c.most_common(1)[0][0] for v, c in votes.items()}

def fuse_segments(voice_segs, voice_names, llm_by_text):
    """Produce fused segments; `agreed` = diarization-voice name matches the LLM's per-segment
    name (the cross-validation / poison detector)."""
    out = []
    for s in voice_segs:
        diar_name = voice_names.get(s["voice"])
        llm_name = llm_by_text.get(s["text"])
        agreed = bool(diar_name) and diar_name == llm_name
        out.append({**s, "speaker": diar_name, "llm_speaker": llm_name, "agreed": agreed})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fuse.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add attribution/fuse.py tests/test_fuse.py
git commit -m "feat(attribution): fusion + agreement cross-check with tests"
```

---

## Task 7: Confidence gate (`gate.py`)

**Files:**
- Create: `attribution/gate.py`
- Test: `tests/test_gate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_gate.py`:
```python
from attribution.gate import gate_segment

def test_structured_kept_when_confident():
    s = {"method": "text", "confidence": 0.8}
    kept, reason = gate_segment(s)
    assert kept is True and reason is None

def test_structured_dropped_when_unsure():
    kept, reason = gate_segment({"method": "text", "confidence": 0.5})
    assert kept is False and reason == "low_confidence"

def test_conversational_dropped_on_disagreement():
    s = {"method": "fused", "confidence": 0.9, "agreed": False}
    kept, reason = gate_segment(s)
    assert kept is False and reason == "method_disagreement"

def test_conversational_kept_on_agreement():
    s = {"method": "fused", "confidence": 0.9, "agreed": True}
    assert gate_segment(s)[0] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`attribution/gate.py`:
```python
"""Decide which segments reach the clean per-person corpus (precision-first)."""
from .config import CONF_MIN

def gate_segment(seg) -> tuple[bool, str | None]:
    """Return (kept, dropped_reason). Conversational requires method agreement."""
    if seg["method"] == "fused" and not seg.get("agreed", False):
        return False, "method_disagreement"
    if seg.get("confidence", 0.0) < CONF_MIN:
        return False, "low_confidence"
    return True, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gate.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add attribution/gate.py tests/test_gate.py
git commit -m "feat(attribution): precision-first confidence gate with tests"
```

---

## Task 8: Per-person corpus assembler (`assemble.py`)

**Files:**
- Create: `attribution/assemble.py`
- Test: `tests/test_assemble.py`

- [ ] **Step 1: Write the failing test**

`tests/test_assemble.py`:
```python
import pandas as pd
from attribution.assemble import build_person_corpus

def test_groups_kept_a16z_segments_by_slug_with_guest_context():
    segs = pd.DataFrame([
        {"post_id": "p1", "segment_idx": 0, "slug": "sonal-chokshi", "is_a16z": True,
         "kept": True, "speaker": "Sonal Chokshi", "text": "Great question."},
        {"post_id": "p1", "segment_idx": 1, "slug": None, "is_a16z": False,
         "kept": True, "speaker": "GUEST_1", "text": "I think L2s win."},
        {"post_id": "p1", "segment_idx": 2, "slug": "sonal-chokshi", "is_a16z": True,
         "kept": True, "speaker": "Sonal Chokshi", "text": "Why?"},
        {"post_id": "p1", "segment_idx": 3, "slug": "sonal-chokshi", "is_a16z": True,
         "kept": False, "speaker": "Sonal Chokshi", "text": "dropped-uncertain"},
    ])
    corp = build_person_corpus(segs)
    assert "sonal-chokshi" in corp
    text = corp["sonal-chokshi"]
    assert "Great question." in text and "Why?" in text
    assert "dropped-uncertain" not in text          # gate respected
    assert "[GUEST]: I think L2s win." in text       # context retained for responses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_assemble.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`attribution/assemble.py`:
```python
"""Assemble clean per-person corpora from gated segments."""

def build_person_corpus(segs) -> dict[str, str]:
    """For each a16z slug, concatenate their kept utterances in post+segment order, keeping
    adjacent guest turns as [GUEST] context lines so responses stay legible."""
    df = segs.sort_values(["post_id", "segment_idx"])
    a16z_slugs = sorted({s for s in df.loc[df["is_a16z"] == True, "slug"].dropna()})  # noqa: E712
    out = {}
    for slug in a16z_slugs:
        lines = []
        for _, r in df.iterrows():
            if r["is_a16z"] and r["slug"] == slug and r["kept"]:
                lines.append(r["text"].strip())
            elif not r["is_a16z"] and r["kept"]:
                lines.append(f"[GUEST]: {r['text'].strip()}")
        out[slug] = "\n".join(lines)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_assemble.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add attribution/assemble.py tests/test_assemble.py
git commit -m "feat(attribution): per-person corpus assembler with tests"
```

---

## Task 9: Orchestrator (`run.py`)

**Files:**
- Create: `attribution/run.py`

- [ ] **Step 1: Implement the orchestrator**

`attribution/run.py`:
```python
"""Run the attribution pipeline end-to-end -> segments parquet, per-person corpora, report.

  python -m attribution.run --pilot          # seminar host + cached podcasts only
  python -m attribution.run                   # full corpus (deferred until pilot clears)
"""
import argparse
import datetime as dt

import pandas as pd

from .config import (CORPUS, PERSONS_DIR, REPORT_OUT, SEGMENTS_OUT, TRANSCRIPTS)
from .route import route_post, CONVERSATIONAL
from .roster import Roster
from .gate import gate_segment
from .assemble import build_person_corpus

def _log(m): print(m, flush=True)

def _participants(row, roster: Roster) -> list[str]:
    """Known participant names for the LLM prompt: guest author(s) + host prior."""
    names = []
    for slug in (list(row.get("author_slugs") or [])):
        hit = roster._df[roster._df["slug"] == slug]
        if len(hit):
            names.append(hit.iloc[0]["name"])
    return names

def _segments_for_post(row, transcript, roster, mode) -> list[dict]:
    from .attribute_text import attribute
    parts = _participants(row, roster)
    llm = attribute(transcript, parts)  # [{speaker,text,confidence}]
    if mode != CONVERSATIONAL:
        for s in llm:
            s["method"] = "text"
        return llm
    # conversational: diarize + fuse
    from .diarize import diarize_audio, transcribe_timestamped, assign_segments_to_turns
    from .fuse import name_voices, fuse_segments
    import glob
    mp3 = glob.glob(f"data/a16z_research/audio_cache/*.mp3")
    if not mp3:
        for s in llm:
            s["method"] = "text"
        return llm
    turns = diarize_audio(mp3[0])
    tsegs = transcribe_timestamped(mp3[0])
    voiced = assign_segments_to_turns(tsegs, turns)
    llm_by_text = {s["text"]: s["speaker"] for s in llm}
    fused = fuse_segments(voiced, name_voices(voiced, llm_by_text), llm_by_text)
    for s in fused:
        s["method"] = "fused"
        s["confidence"] = 0.9 if s["agreed"] else 0.4
    return fused

def build(pilot: bool) -> pd.DataFrame:
    corpus = pd.read_parquet(CORPUS)
    tx = pd.read_parquet(TRANSCRIPTS)
    tx = tx[tx["status"] == "ok"]
    roster = Roster.load()
    rows = []
    merged = tx.merge(corpus[["object_id", "formats", "author_slugs", "title"]],
                      on="object_id", how="left")
    if pilot:
        merged = pd.concat([
            merged[merged["formats"].astype(str).str.contains("podcast", na=False)],
            merged[merged["formats"].astype(str).str.contains("video", na=False)].head(20),
        ])
    for _, row in merged.iterrows():
        mode = route_post(row).mode
        try:
            segs = _segments_for_post(row, row["transcript"], roster, mode)
        except Exception as e:
            _log(f"  attribution_failed {row['object_id']}: {e!r}")
            continue
        for i, s in enumerate(segs):
            m = roster.resolve(s.get("speaker"))
            kept, reason = gate_segment(s)
            rows.append({
                "post_id": row["object_id"], "segment_idx": i, "method": s["method"],
                "speaker": s.get("speaker"), "slug": m.slug, "is_a16z": m.is_a16z,
                "confidence": s.get("confidence"), "agreed": s.get("agreed"),
                "kept": kept, "dropped_reason": reason, "text": s["text"],
            })
    return pd.DataFrame.from_records(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    df = build(pilot=args.pilot)
    SEGMENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SEGMENTS_OUT, index=False)
    PERSONS_DIR.mkdir(parents=True, exist_ok=True)
    corp = build_person_corpus(df)
    for slug, text in corp.items():
        (PERSONS_DIR / f"{slug}.txt").write_text(text)
    kept = int(df["kept"].sum())
    with open(REPORT_OUT, "w") as f:
        f.write(f"# Attribution report ({dt.datetime.now(dt.timezone.utc).isoformat()})\n\n")
        f.write(f"- segments: {len(df)} | kept: {kept} | dropped: {len(df)-kept}\n")
        f.write(f"- a16z people with corpus: {len(corp)}\n\n")
        f.write("## Dropped reasons\n")
        f.write(df[~df['kept']]['dropped_reason'].value_counts().to_string())
        f.write("\n\n## Per-person kept chars\n")
        for slug, text in sorted(corp.items(), key=lambda x: -len(x[1])):
            f.write(f"- {slug}: {len(text):,}\n")
    _log(f"Wrote {SEGMENTS_OUT} ({len(df)} segs, {kept} kept), {len(corp)} person corpora")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the orchestrator wiring (no LLM) on an empty/edge case**

Run: `python -c "from attribution import run, route, roster, gate, fuse, assemble, attribute_text, diarize; print('imports OK')"`
Expected: `imports OK`.

- [ ] **Step 3: Commit**

```bash
git add attribution/run.py
git commit -m "feat(attribution): orchestrator + report"
```

---

## Task 10: Pilot run & validation (manual checkpoint)

**Files:** none — produces `attributed_segments.parquet`, `persons/*.txt`, `attribution_report.md`.

- [ ] **Step 1: Ensure HF token present for diarization**

Run: `op read op://local/huggingface/credential >/dev/null && echo OK`
Expected: `OK`. If absent: create a free HF token, accept the `pyannote/speaker-diarization-3.1` license, store it in 1Password at that path. (Without it, conversational posts degrade to text-only — acceptable for a partial pilot.)

- [ ] **Step 2: Run the pilot**

Run: `python -m attribution.run --pilot`
Expected: report written; ≥1 a16z slug (tim-roughgarden / justin-thaler / sonal-chokshi) has a non-trivial corpus; dropped-reasons table shows the gate firing.

- [ ] **Step 3: Hand-verify attribution quality (the real gate)**

Open `data/a16z_research/persons/tim-roughgarden.txt` (or justin-thaler) and spot-check against 5 source transcripts: do the lines attributed to the host actually belong to the host (intros/Q&A), with guest content excluded? Open `attribution_report.md` and confirm the podcast both-engines path produced `method_disagreement` drops (proof the cross-check works).

- [ ] **Step 4: Engine-ingestion check**

Hand one `persons/<slug>.txt` to the doppelganger engine (per its input contract — see `vault/docs/specs/2026-04-22-doppelganger-v1.2-design.md` and the repo at `/Users/jax/.../projects/doppelganger/`) and confirm an *observed* corpus drives it, or document the exact adapter needed. This is the riskiest unknown.

- [ ] **Step 5: Commit pilot artifacts + findings**

```bash
git add data/a16z_research/attributed_segments.parquet data/a16z_research/persons/ data/a16z_research/attribution_report.md
git commit -m "data(attribution): pilot run — host + podcast person corpora"
```

- [ ] **Step 6: Write pilot findings memo**

Create `vault/knowledge/2026-06-05-attribution-pilot-findings.md`: attribution accuracy on the host, the both-engines drop rate, whether the doppelganger engine ingests observed corpus, and the go/no-go for scaling to the full roster + the parallel session's expanding podcast corpus.

---

## Self-review notes

- **Spec coverage:** route (§3.1)→T2; attribute_text (§3.2)→T4; diarize (§3.3)→T5; fuse (§3.4)→T6; resolve (§3.5)→T3 + host-prior in config/run; gate (§3.6)→T7; assemble (§3.7)→T8; outputs (§4)→T9; pilot (§7)→T10; prereq (§2.1)→T0. Engine-ingestion (§7.3)→T10S4.
- **Deferred by design:** audio pulls for videos (panel_candidate flag only); full-roster scale run (pilot-first per Jax). Both noted, not silently skipped.
- **Gitignore:** add `data/a16z_research/ts_transcripts/` to `.gitignore` during T5 (timestamped re-transcribe cache, re-derivable).
