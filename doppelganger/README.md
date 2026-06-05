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
- **Podcast** — `data/a16z_research/attributed_transcripts.jsonl` (subject's diarized turns >= 0.8 confidence; preceding question kept as context). **Produced by the corpus-attribution pipeline — copy that file to this path before ingesting.**
- **Identity** — `data/linkedin/parsed/<slug>.json` + `data/a16z_team/team.parquet`.

### Tuning (`doppelganger/config.py`)
`MIN_REPLY_CONTENT_CHARS=50`, `PODCAST_MIN_CONFIDENCE=0.8` — documented defaults, eval-tuned later.
