# sentience — status & handoff

_Last updated: 2026-06-07_

## ⚡ ACTIVE WORK — Corpus Doppelganger Engine (branch `doppelganger/engine-design`)

A corpus-only "doppelganger" engine: ingest a person's public corpus (X + research +
speaker-attributed podcast turns + LinkedIn/bio), build a frozen-at-T0 **soul**
(characterization), and answer **time-gated walk-forward** market-view queries *as that
person at date T*. First subjects: **Eddy Lazzarin**, **Ali Yahya** (a16z crypto GPs).
Design docs: `docs/superpowers/specs/2026-06-05-*doppelganger*` and `.../plans/2026-06-05-*doppelganger*`.

**All 5 units BUILT + tested (72 tests), on this branch (not merged to main):**
ingestion → soul → memory → doppelganger(`respond`) → walk-forward + scorers. Module: `doppelganger/`.
- LLM calls = `claude -p --model opus --effort high` (Max sub, no API cost) via `doppelganger/llm.py`. No SDK.
  **Effort was `max` through 2026-06-06; dialed to `high` on 2026-06-07 to conserve credits.** This leaves a
  known eval confound: **Eddy was judged at `max`, Ali at `high`.**
- Generated artifacts are COMMITTED under `data/doppelganger/<slug>/` (the data *is* the deliverable,
  same as the corpus parquets — NOT gitignored). Only transient run logs (`logs_run_full.txt`) are ignored.

**Core principle (Jax-enforced):** feed the model inputs, don't script its cognition. Eval is **AGENTIC**
— LLM-judge agents read the qualitative views; deterministic scores are only rough indicators.

**FULL walk-forward DONE — both subjects, 14 quarters each (2026-06-07).** Trajectories + ablation + judge
complete; scores written; agentic synthesis memo at `docs/doppelganger-synthesis-2026-06-07.md`. Findings:
- Leakage firewall holds (FULL arm overwhelmingly grounded/cited; soul-less ablation ~100% `extrapolated`,
  0 citations → **no quote-recitation leakage**).
- `confirm_rate` = 1.0 both arms every quarter → **mean lift 0 = metric saturation, NOT "corpus useless."**
  Real signal = `missed_changes` (nails persistence, misses foresight). Do NOT headline confirm_rate.
- **Discrimination 4/5 distinct** (agentic). The ablation-collapse is the real positive: strip the corpus and
  both subjects produce nearly the *same* generic-a16z doc → all distinctiveness is carried by soul+memory.
  The experiment's true product is **characterization fidelity**, not forecasting.
- Honest limits: N=2, ~14 quarters, single judge, the max/high effort confound.

**NEW — per-pick `conviction` score (0-100) shipped (2026-06-07, commit `8d6e906`).** Each sector/token view
+ risk_regime now self-reports conviction (intensity, NOT calibrated probability). Validated on an Eddy
sample: full range used (64-92), highest on ZK (his Jolt frontier) + memecoin concern. Conviction
trajectories show decay (onchain games 80→70→gone), growth (payments 72→90), steady anchors (ZK ~90).
- ⚠️ The committed dataset's views have **NO conviction** (generated before the feature). Only a scratch
  4-quarter Eddy sample exists (`/tmp/dg_conv_test`, uncommitted).

**NEW — blind authorship eval DONE (2026-06-07).** `doppelganger/authorship.py` (tested). Three-rung
ladder, judge blind to identity attributes a stance-only view to Eddy/Ali. Results + full writeup:
`docs/doppelganger-authorship-eval-2026-06-07.md`.
- FULL 100% (conf 86) | NAMED-ABLATION 82% (conf 78) | ANON-ABLATION split 10E/4A (conf 61).
- **This REVISED the synthesis memo:** the "ablation collapse / near-interchangeable" claim is overstated
  (name-only is still 82% attributable). Real findings: base-model default voice is **Eddy-shaped**
  (ZK/infra), so Ali is the distinctive one; the corpus's measurable contribution is the **confidence
  ladder 61→78→86** + ceiling accuracy, NOT a rescue from chance. Memo has a correction banner.

**▶ NEXT STEP (pick up here):**
1. **Full conviction backfill — DEFERRED to weekly-usage reset** (Jax was at 80% on 2026-06-07). Cost
   ~3.4M tokens (uncapped feed; Eddy's 2026 quarter alone is 228k input). Regenerate 14 full-arm views ×2
   with conviction; no re-judge needed. Then add conviction-trajectory analysis to the findings.
2. **Unknown-subject authorship control** (the clean follow-up): re-run authorship eval with a subject the
   base model does NOT know (e.g. a pseudonymous writer) → floors should drop toward chance, exposing the
   corpus lift uncompressed. The way to prove corpus-*created* (not pretrained) distinctiveness.
3. Replace the saturated `confirm_rate` headline with **change-recall + groundedness**, measured agentically.

## BUILT (unmerged) — A1 signal panel, Sprint 1a (branch `signal/corpus-strategy`)

Module: `signals/`. Turns the blended a16z corpus (research articles + transcripts + Twitter)
into a structured signal timeseries panel. **36 tests, all passing.** Final Opus code review done;
3 issues fixed + verified (C1 day-31 `rebalance_dates` crash; C2 audit fuzzy fallback was inert
+ now folds smart-quotes/dashes with a real windowed match; C3 EXITED rows now keep token
item_type/parent_sector). Firewall integrity re-verified post-fix (hallucination + post-T leak
still caught, verbatim still passes).

⚠️ **CODE-COMPLETE + UNIT-TESTED ONLY — NOT YET VALIDATED ON REAL DATA.** All tests mock the LLM.
The real Definition-of-Done gates have NOT run: (1) `signals.run distill` on the 149 transcripts
with a manual verbatim spot-check; (2) `signals.run panel` producing real artifacts; (3) `audit.json`
showing ZERO leaks across periods; (4) the validation gate; (5) token-headroom check at the largest
T to fix the final `window_months`. This is a real-`claude -p` run (Opus/high) — distill ~149 docs
+ extract/canonicalize ×~14 quarters — i.e. meaningful compute. **1a is NOT done until these pass.**

Pipeline: assemble trailing-window corpus (18mo default) → A1 consensus extraction (LLM,
recency-privileging) → agentic fit-or-mint canonicalization → leakage audit → deterministic
lifecycle/delta panel. CLI: `python -m signals.run {distill,panel,validate}`.

Output: `data/signal/signal_panel.parquet` — one row per (period, item) with columns
`as_of`, `item`, `item_type`, `parent_sector`, `stance`, `conviction`, `horizon`,
`lifecycle_state` (NEW/SUSTAINED/FLIPPED/EXITED), `delta_stance`, `delta_conviction`, `age`.

Design doc: `docs/superpowers/specs/2026-06-07-corpus-signal-strategy-design.md`.
Plan: `docs/superpowers/plans/2026-06-07-a1-signal-panel.md`.
Module README: `signals/README.md`.

**Deferred:**
- **Real-data validation run (the actual DoD above) — THE immediate next gate, needs compute sign-off.**
- `validate` full-text arm is a stand-in (Task 10b) — transcripts excluded from the "full" path, not a true full-text comparison.
- Review nits not yet addressed (safe to defer): C4 `_extract_json` duplicated across distill/extract/canonicalize (consolidate); C5 corpus assembled twice per period in `build_panel` (extract + audit) — fine quarterly, wasteful monthly; C6 redundant clause in panel FLIPPED condition (harmless).
- Sprint 1b (Coinglass market-data panel + BTC beta) — separate sprint, not started.
- Sprint 1c (strategy / backtest / eval) — separate sprint, not started.
- A2a (per-member extraction) and A2b (dispersion/consensus) — unbuilt.

Branch not yet merged to main. Worktree: `.claude/worktrees/signal-strategy`.

## What this project is

A **research** project. **Research question DEFINED 2026-06-07:** structure unstructured corpus
into a tradeable signal (corpus → features → consensus signal → strategy → backtest); deliverable
is the pipeline/finding, not a Sharpe. See the spec. (Historical note below predates that.) First concrete move
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
