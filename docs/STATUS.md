# sentience — status & handoff

_Last updated: 2026-06-04_

## What this project is

A **research** project. Classified "both / not sure yet" at kickoff — it may stay an
exploration space or concrete into a build. **The overarching research question is still
TBD** — Jax said "we're gonna do some research and see where it goes." First concrete move
was corpus collection, not a defined thesis. When the direction sharpens, run
`/brainstorming` and land a spec in `docs/`.

## ⚠️ Path note

The workspace root was renamed `entities` → `workspaces` on 2026-06-04. Real path is now
`/Users/jax/workspaces/ultim8/...`. If anything references `/Users/jax/entities/...`, it's
stale. (The `.claude` memory was mirrored to the `-Users-jax-workspaces-ultim8` slug.)

## DONE — a16z crypto research corpus (commit `feb701c`)

`scrapers/a16z_research/` scrapes every research-category post from a16zcrypto.com.
**The scrape is complete. Do NOT re-run it unless refreshing the data.**

- **235 posts** collected → `data/a16z_research/metadata.parquet` (Algolia metadata) and
  `articles.parquet` (metadata + fetched full content). 234 have a body.
- Mix: ~139 videos, 85 written articles, 10 podcasts, 1 paper.
- Full re-run is ~45s: `python -m scrapers.a16z_research.run` (see scraper README for the
  WAF/Algolia/scrape.do mechanics — they're non-obvious; read it before touching the code).
- Schema + body strategy documented in `scrapers/a16z_research/README.md`.

Setup for a clean machine:
```bash
cd /Users/jax/workspaces/ultim8/projects/sentience
python -m venv .venv && . .venv/bin/activate
pip install -r scrapers/a16z_research/requirements.txt
python -m playwright install chromium
```

## DONE — video + podcast transcripts (commit `ee4f862`)

`scrapers/a16z_transcripts/` closes the §8 content gap: full spoken transcripts for all
149 video/podcast posts → `data/a16z_research/transcripts.parquet` (join table on
`object_id`; the corpus parquet is never mutated). **148/149 OK, 8.38M chars** (median 56k
chars/post vs ~1.8k abstracts before). The one miss is a video a16z removed from YouTube.

- Videos + 3 podcasts → YouTube captions; 7 YouTube-less podcasts → Simplecast audio →
  on-device `mlx-whisper` (free, no key; needs `ffmpeg` + Apple Silicon).
- **Both fetch legs go through scrape.do.** YouTube IP-blocks this machine after ~40 direct
  caption pulls *and* blocks datacenter proxies → must use the **residential** proxy
  (`super=true`). Podcast audio uses scrape.do **API mode** (follows Simplecast's 301).
  Mechanics are non-obvious — read `scrapers/a16z_transcripts/README.md` before touching.
- Re-runs **resume** from already-OK rows. mp3s cached under `data/a16z_research/audio_cache/`
  (gitignored). Run: `python -m scrapers.a16z_transcripts.run`.

## Open threads (pick up here)

1. **Define the research question.** This is the real next step. The corpus exists; what are
   we actually investigating with it? Needs a brainstorm with Jax.
2. **Format filter.** Corpus is video-heavy (139 videos vs 85 written articles). If "research"
   means the essays, filter `formats == 'articles'`. Not yet built — trivial to add.
3. **More sources?** Only a16z so far. The project may want a broader crypto-research corpus.
4. **Data in git.** The ~10MB of parquets are committed (data is the deliverable). Revisit if
   the corpus grows — may want to gitignore `data/` and store elsewhere.

## Conventions

Each `projects/*` is its own git repo (this one: branch `main`, no remote yet). scrape.do
token lives in 1Password (`op item get scrape.do --vault local`).
