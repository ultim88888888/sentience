# Both-Engines Fusion — Redesign (investigation findings)

_2026-06-05 · status: investigated + prototyped, not yet productionized · text-LLM-only ships in the meantime_

## The bug in v1 fusion

v1 ran diarization and the LLM **independently**, each producing its own segmentation
(whisper: ~800 short timestamped segments; LLM: ~90 freely-merged turns), then joined them by
**exact text string**. Real segmentations never match char-for-char → **4% aligned** (garbage
voice→name map). Unit tests missed it because they hand-fed matching text.

## The corrected architecture (prototyped, validated direction)

Align by **segment index**, not text. One shared segmentation = whisper's timestamped segments.

1. `diarize_audio(mp3)` → voice turns (pyannote v4, **on MPS** — ~3 min/30-min episode; CPU is
   ~40× slower and was the 2-hour runaway).
2. `transcribe_timestamped(mp3)` → whisper segments (text + time).
3. `assign_segments_to_turns` → each whisper segment gets a **voice** (max time-overlap).
4. **LLM labels the *numbered* whisper segments** (speaker per index, batched, names carried
   across batches) — NOT free re-segmentation. Align by index → every segment has voice + name.
5. **Normalize** each LLM name to a roster slug (so "Eddy" / "Eddy Lazzarin" → `eddy-lazzarin`)
   **before** voting.
6. **Voice→identity by majority vote** per diarization cluster; **confidence = intra-voice
   naming consistency**. That consistency is the real cross-check (diarization says "same
   voice," does the LLM agree on who?).

## Prototype results (one Eddy podcast, "Talking trends 2025 pt2")

- LLM labeled **806/806** segments (vs 4% before). ✅
- **Target isolated cleanly: SPEAKER_01 → `eddy-lazzarin`, 0.98 consistency, 223 segments.** ✅
  This is the precision win — confidently take that whole voice cluster as Eddy, with
  diarization boundaries.
- Overall per-segment agreement ~51%, dragged down by **non-target** voices (Robert 0.46, a
  spurious "kara", one speaker split across clusters).

## Remaining issues to solve before productionizing

1. **LLM non-determinism** — labels shift run-to-run. Mitigate: lower/zero temperature if
   exposed; or multi-vote per segment; or feed diarization structure (voice tags) into the
   prompt so the LLM names *voices* rather than re-deciding boundaries.
2. **Diarization over-clustering** — one person → multiple SPEAKER_xx. Majority-vote +
   slug-normalization absorbs it (all clusters → same slug) but inflates "disagreement". Could
   merge clusters whose majority slug matches.
3. **Hallucinated names on minor speakers** ("kara"). Constrain harder to the known-participant
   roster; treat unresolvable names as GUEST_n.
4. **Metric** — global agreement is the wrong KPI; **per-target-voice consistency** is what
   matters for a doppelganger corpus. Productize as: keep voices with consistency ≥ threshold
   resolving to the target; flag the rest.

## Recommendation

Direction is right; the win (clean voice-bounded target cluster) is real. But text-LLM-only
already yields clean target corpora, so this is a **precision/recall-control enhancement, not a
blocker** — finish it as a v2 when podcast boundary precision is worth the compute (~6 min/
podcast for diarize+whisper). Productionized version: replace `fuse_segments` (text-match) with
the index-aligned, slug-normalized, voice-majority approach; gate on per-voice consistency.
Prototype logic captured in git history (`_proto_fusion.py`, this commit's parent tree).
