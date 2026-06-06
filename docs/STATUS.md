# sentience — status & handoff

_Last updated: 2026-06-06_

## ⚡ ACTIVE WORK — Corpus Doppelganger Engine (branch `doppelganger/engine-design`)

A corpus-only "doppelganger" engine: ingest a person's public corpus (X + research +
speaker-attributed podcast turns + LinkedIn/bio), build a frozen-at-T0 **soul**
(characterization), and answer **time-gated walk-forward** market-view queries *as that
person at date T*. First subjects: **Eddy Lazzarin**, **Ali Yahya** (a16z crypto GPs).
Design docs: `docs/superpowers/specs/2026-06-05-*doppelganger*` and `.../plans/2026-06-05-*doppelganger*`.

**All 5 units BUILT + tested (67 tests), on this branch (not merged to main):**
ingestion → soul → memory → doppelganger(`respond`) → walk-forward + scorers. Module: `doppelganger/`.
- LLM calls = `claude -p --model opus --effort max` (Max sub, no API cost) via `doppelganger/llm.py`. No SDK.
- Generated artifacts live UNTRACKED under `data/doppelganger/<slug>/` (committed here only to carry the
  in-progress run across machines).

**Core principle (Jax-enforced):** feed the model inputs, don't script its cognition. Eval is **AGENTIC**
— LLM-judge agents read the qualitative views; deterministic scores are only rough indicators.

**Subset findings (Eddy, 3 quarters) — the real results so far:**
- Leakage firewall holds (0 leaked everywhere). Soul-less ablation honestly labels 100% `extrapolated`,
  0 citations → **no quote-recitation leakage**; it just guesses stable views well.
- `confirm_rate` corpus-lift = 0 but that's **metric saturation** (stable views never contradicted), not
  "corpus useless." Real signal = `missed_changes` (doppelganger nails persistence, misses foresight).
- Eddy-vs-Ali discrimination: agentic judge graded **4/5 distinct** (distinct frameworks; shared themes,
  distinct *why*). Deterministic name-overlap gave a misleading 0.0 → why the eval is agentic.

**▶ NEXT STEP (resume here, esp. on the always-on Mac Mini):**
1. Run the full quarterly walk-forward, both subjects (resumable; skips cached steps + transient failures):
   ```bash
   tmux new -s wf 'source .venv/bin/activate && caffeinate -i python -m doppelganger.run_full'
   ```
   ~several hours: the memory feed grows to ~228k tokens by 2026 (feed-all, by design — Jax: all his
   history shapes soul + analysis context, so NO recency cap). Reattach: `tmux attach -t wf`.
2. Then score + analyze agentically → findings:
   ```bash
   python -m doppelganger.run score --subject eddy-lazzarin   # + ali-yahya
   ```
   plus dispatch a markets-analyst agent over the trajectories for discrimination + a synthesis memo.
3. Open question to revisit: replace the saturated `confirm_rate` headline with change-recall +
   groundedness (agentic), per the subset finding.

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

## DONE — twitterapi.io scraper (branch `scrape/twitterapi`)

`scrapers/twitter/` scrapes **all posts for a user since a start date** (originals,
replies, quotes, retweets) via twitterapi.io `advanced_search` → `data/twitter/<user>.parquet`,
one row per tweet (curated columns + `raw_json`). Pure async REST, shaped like
`market_data/coinglass.py`. Auth: 1Password item `twitterapi-io` (vault `local`) via `op`.

- Run: `python -m scrapers.twitter.run --user eddylazzarin --since 2025-01-01`
  (`--until`, `--no-retweets`, repeatable `--user` supported). Programmatic: `run.pull(...)`.
- **Depth-first** by design: date-windowed `from:user since_time/until_time` reaches any
  start date (no ~3,200 timeline cap). 19 unit tests (mocked), all green.
- **Retweet finding (resolved):** `from:user` excludes native retweets; `fetch_user` adds a
  second `filter:nativeretweets` pass, unioned + deduped by id. Live-verified 2026-06-05 vs
  `@eddylazzarin`: 631 tweets since 2025-01-01 (407 reply / 141 quote / 82 original / 1 RT),
  no dupes. See `scrapers/twitter/README.md` + spec `docs/superpowers/specs/2026-06-05-twitterapi-client-design.md`.
- v1 overwrites per run; no incremental/dedup across runs (YAGNI). Branch not yet merged.

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
