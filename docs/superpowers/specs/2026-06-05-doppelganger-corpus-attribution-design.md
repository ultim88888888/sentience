# Doppelganger Corpus Attribution — Per-Person Speech Extraction

_Design spec · 2026-06-05 · status: draft-pending-review_

## 1. Purpose & thesis

Build **per-person attributed speech corpora** for the a16z crypto team from the scraped
transcript corpus, so each person's words can feed a **digital doppelganger** (the
`doppelganger` project — V1.1b at 3.66/5, V1.2 pending). To clone "Eddy," the engine needs
*exactly Eddy's words and responses*, cleanly separated from everyone else's.

The transcripts today are **flat text with no speaker attribution** — caption snippets and
whisper segments concatenated into one string per post. This pipeline turns that into
`speaker → text` segments, names the a16z-affiliated speakers, and assembles a clean
per-person corpus.

> **Governing principle — precision over recall.** A mis-attributed sentence *silently
> poisons* a doppelganger: feed "Eddy" a line Chris actually said and the clone is corrupted
> invisibly. So the pipeline **drops uncertain segments rather than guess.** Less data beats
> wrong data. Every quality knob defaults to conservative.

### Non-goals (v1)
- Not cloning external academic guests — only **a16z team** speakers need naming. Non-a16z
  speakers are segmented and labeled `guest`, not individually resolved.
- No paid diarization/ASR services (AssemblyAI, Deepgram). All compute is **local/free**.
- No change to the doppelganger engine itself — this produces *input corpora*; whether/how
  the engine ingests observed (vs elicited) corpus is validated in the pilot, designed
  separately.
- No live/streaming attribution. Batch over the existing corpus.

## 2. Prerequisites & dependencies (multi-session coordination)

This workstream sits downstream of work the **parallel session** owns. Before/around this:

1. **Transcript corpus must be refreshed against the expanded corpus.** The corpus grew
   235 → **1,018 posts** (commit `3dc798b`); `transcripts.parquet` still covers the old 149
   and its `object_id` join is partially broken. The corpus now has **335 video+podcast
   posts**. **Prereq:** re-run `scrapers.a16z_transcripts.run` against the 1,018-post corpus
   (it resumes, so only new posts are fetched). Owned here unless the parallel session takes
   it — coordinate to avoid double-pull.
2. **Roster exists** — `data/a16z_team/team.parquet` (89 members: `slug, name, title, bio,
   x_url, …`, commit `c623295`). Confirmed to include all named targets. This is the
   name-resolution authority; no roster work needed here.
3. **Branch hygiene.** Two sessions share one working tree. Implementation of this pipeline
   runs in an **isolated git worktree** on its own branch to avoid checkout collisions
   (resolve base branch — `main` vs `scrape/a16z-team` — at plan time, since the expanded
   corpus + roster currently live on `scrape/a16z-team`).

## 3. Architecture

Content-routed **dual-engine** attribution. Each post is routed by whether we have audio and
whether it's conversational; the two engines do complementary jobs and cross-check.

```
transcripts.parquet ─┐
articles.parquet ────┼─> route.py ──> {structured | conversational}
team.parquet ────────┘                   │
                                          ├─ structured (caption-only talks)
                                          │     └─> attribute_text.py (LLM) ──┐
                                          │                                    │
                                          └─ conversational (podcasts, audio)  │
                                                ├─> diarize.py (pyannote) ──┐  │
                                                └─> attribute_text.py (LLM) ┤  │
                                                         fuse.py (boundaries⊕names, agreement) ┘
                                                                  │
                                  resolve.py (roster match, is_a16z) <┘
                                                                  │
                                  gate.py (confidence/agreement → keep|flag|drop)
                                                                  │
                          attributed_segments.parquet  +  persons/<slug>.{parquet,txt}
```

### 3.1 `route.py` — content router
- Input: corpus row (format, has-audio, multi-author hints). Output: `structured` or
  `conversational`.
- **Rule:** `format == podcast` (Simplecast/whisper audio available) → **conversational →
  both engines**. `format == video` (transcribed from YouTube captions, no audio) →
  **structured → text-LLM only**. Audio pulls for videos are deferred (YAGNI) — a flag marks
  panel/fireside videos (≥2 a16z authors or title cues) as future audio-diarization
  candidates without acting on them in v1.
- Pure function; unit-tested on synthetic rows.

### 3.2 `attribute_text.py` — LLM text attribution
- Reuses the doppelganger LLM toolchain: `claude -p --model opus` (or SDK under `op run`) —
  **$0 on the Max subscription**, no API metering.
- Input: a transcript + **participant hints** (guest name from `author_slugs`/title;
  podcast host from episode metadata) + a **roster subset** (candidate a16z names).
- Output (structured/JSON, schema-validated): ordered segments
  `[speaker_label, text, char_span, confidence]`, where `speaker_label` is a resolved name
  where inferable (intros: "please welcome X"; thank-yous: "Thanks, Justin") else a stable
  anonymous label (`HOST`, `GUEST_1`).
- **Long transcripts** (median 56k, max 123k chars) are chunked with overlap; the speaker-
  label map is carried across chunks so labels stay consistent within a post.
- Independent of diarization — for conversational posts it runs anyway, as the cross-check.

### 3.3 `diarize.py` — audio diarization (conversational only)
- `pyannote.audio` `speaker-diarization-3.1`, **local on Apple Silicon (MPS), free.** One-
  time free HF token + license accept (read from 1Password, not committed).
- Input: cached episode mp3 (`audio_cache/`, already downloaded for podcasts). Output:
  voice turns `[start, end, voice_label]` (anonymous Speaker_0/1/…).
- To map words→turns we need **timestamped** transcript segments: re-transcribe podcasts
  with mlx-whisper **keeping segment timestamps** (we discarded them in the flat join) and
  assign each whisper segment to the max-overlap diarization turn.
- Integration-tested on one cached podcast; **gated/skipped without HF token.**

### 3.4 `fuse.py` — combine boundaries ⊕ names (conversational only)
- Diarization gives **boundaries + anonymous voices**; the text-LLM pass gives **names**.
  Fuse: name each voice label (Speaker_0 → "Sonal Chokshi") from the LLM's resolved labels
  + intro cues, producing voice-accurate, name-labeled segments.
- **Cross-validation = the precision gate's core signal.** For each segment compare the
  diarization-derived speaker against the LLM's *independent* text attribution:
  **agree → high confidence; disagree → low confidence (flag).** Two independent methods
  concurring is our poison-detector.

### 3.5 `resolve.py` — roster resolution & a16z tagging
- Maps each resolved speaker name to a roster entry (`data/a16z_team/team.parquet`) via
  name/alias/last-name matching → attaches `slug`, `is_a16z`, `title`. Unmatched named
  speakers (external guests) → `is_a16z=False`, kept as `guest`.
- **Host-name fallback:** the recurring seminar host is often unnamed in their own intro but
  named when a guest thanks them ("Thanks, Justin"). A small **known-host prior** (Tim
  Roughgarden / Justin Thaler for research seminars; Sonal Chokshi for podcasts) seeds
  resolution where textual cues are absent. The prior is auditable config, not hardcoded
  logic.

### 3.6 `gate.py` — confidence gate
- Decides what reaches the **clean per-person corpus** (the doppelganger feed):
  - **Conversational:** keep iff diarization⊕LLM **agree** AND confidence ≥ threshold.
  - **Structured:** keep iff LLM confidence ≥ threshold.
  - Else → retained in `attributed_segments` but **excluded from the person corpus**, with a
    `dropped_reason`. Nothing is silently discarded; drop counts are reported.

### 3.7 `assemble.py` — per-person corpus builder
- Groups kept segments by `slug`, concatenates each a16z person's utterances (preserving
  conversational context — adjacent guest turns kept as `[GUEST]:` context lines so
  *responses* are legible, not just monologue), writes per-person corpus + provenance.

## 4. Data flow & storage

- Inputs (read-only): `transcripts.parquet`, `articles.parquet`, `team.parquet`, `audio_cache/`.
- `data/a16z_research/attributed_segments.parquet` — every segment (kept or dropped):
  `post_id, segment_idx, start, end (nullable for text-only), method (text|diar|fused),
  speaker_label, speaker_name, slug, is_a16z, confidence, kept (bool), dropped_reason, text`.
  **Committed** (the attribution artifact).
- `data/a16z_research/persons/<slug>.parquet` + `.txt` — clean per-person corpus for the
  doppelganger engine. **Committed** (the deliverable).
- `data/a16z_research/attribution_report.md` — per-post method, kept/dropped counts,
  per-person char totals, low-confidence flags. Committed.
- Re-transcription timestamp cache for podcasts is gitignored (re-derivable).

## 5. Error handling
- Missing HF token → conversational posts **fall back to text-LLM only** with a logged
  warning (don't crash; degrade to structured-mode behavior).
- Diarization failure on a post → fall back to text-LLM only for that post, logged.
- LLM JSON schema violation → retry; persistent failure → post marked `attribution_failed`,
  excluded, counted in report (no silent loss).
- Speaker resolvable to **multiple** roster entries → mark ambiguous, low-confidence, drop
  from clean corpus.
- Empty/near-empty transcript → skip, logged.

## 6. Testing
- `route.py`: unit — podcast→conversational, video→structured, panel-video flagged.
- `resolve.py`: unit — exact/last-name/alias match; external guest → `is_a16z=False`;
  ambiguous → flagged; host-prior fills an unnamed-host fixture.
- `fuse.py`: unit — agreement→high-conf, disagreement→flag, on synthetic diar+LLM segments.
- `gate.py`: unit — keep/flag/drop thresholds on both modes; drop accounting sums correctly.
- `attribute_text.py`: golden test on a **hand-labeled** seminar-opening snippet (assert
  host-intro vs guest-body split correct); chunk-boundary label-consistency test.
- `diarize.py`: one integration test on a cached podcast (gated; skipped without HF token).
- `assemble.py`: unit — grouping by slug, guest-context retention, provenance.

## 7. Pilot (validates the whole path before scale)
1. **Text-LLM on the seminar host** — run structured attribution over the research-seminar
   talks, extract **Tim Roughgarden / Justin Thaler** segments → first person corpus. Spot-
   check 10 talks by hand for mis-attribution.
2. **Both-engines on the cached podcasts** — run diarize⊕text-LLM⊕fuse on the 7 (now more)
   cached podcasts hosted by **Sonal Chokshi**; verify the agreement gate behaves and a
   clean Sonal corpus emerges.
3. **Engine ingestion check** — hand one person corpus to the doppelganger engine to confirm
   an *observed* corpus can drive it (or surface exactly what adapter it needs). This is the
   biggest unknown and the pilot's real purpose.

## 8. Deliverable & success criteria
One command (`python -m attribution.run`) produces `attributed_segments.parquet`, per-person
corpora, and `attribution_report.md`. **Success = at least one a16z person (the seminar
host) has a clean, hand-verified, mis-attribution-free corpus large enough to seed a
doppelganger, AND the both-engines agreement gate demonstrably drops the disagreements** —
i.e. the precision-first machinery works end-to-end on real data. Scale to the full roster +
the parallel session's expanding podcast corpus comes after the pilot clears.
