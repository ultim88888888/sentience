# sentience — status & handoff

_Last updated: 2026-06-05_

## What this project is

A **research** project. The overarching research question is now defined: **does a16z
research-coverage rotation lead relative crypto-basket and token returns?** Built as a
viability study — smoke-test framing, not a production signal. First concrete move was
corpus collection (235 posts); the signal study (`study/`) followed.

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

## Research question (defined 2026-06-05)

**Hypothesis:** a16z coverage-rotation (measured as a basket or token's monthly share of
research attention vs. trailing baseline) leads relative crypto returns. Mechanism: attention
flows — a16z research creates awareness and narrative momentum that precedes price rotation.

**Two studies:**
- **Study A (basket level):** IC of `coverage_momentum` (share − trailing 3m mean) vs.
  forward 1m and 3m basket-relative returns. Verdict (both attribution modes): **no pulse**.
- **Study B (token level):** IC of per-token conviction signal (sum or mean of basket
  coverage) vs. forward token-relative returns. Verdict (all agg × mode combos): **no pulse**.

**Smoke-test caveat:** n ≈ 40–50 basket-months. Not statistically significant (IC ≈ −0.11,
t ≈ −1.5; ~35% of monthly rank-correlations are positive). "No pulse" means **no detectable
signal** — the consistently-negative IC is within noise, NOT a contrarian signal, and the six
combinations are not independent (they share one sparse corpus + overlapping baskets, so they
are one observation viewed six ways). A full whole-pipeline audit (look-ahead, joins,
sign/demean) confirmed the result is sound and not an artifact.

**Pointers:**
- Spec: `docs/superpowers/specs/2026-06-05-a16z-research-signal-design.md`
- Plan: `docs/superpowers/plans/2026-06-05-a16z-research-signal.md`
- Package: `study/` (`config`, `coverage`, `returns`, `signal`, `study_basket`, `study_token`, `findings`, `run`)
- Generated output: `findings.md` (repo root), `data/study/coverage_heatmap.png` (gitignored)

## Open threads (pick up here)

1. **v1 verdict: no pulse across both studies and both attribution modes** — IC small/negative
   (-0.03 to -0.15), within noise. **Root cause to address before any v2:** the corpus is too
   sparse for this test — ~52% of months have zero posts hitting any basket, so half the
   sample carries no cross-sectional signal at all. Options: (a) widen the corpus (more
   sources / longer history), (b) coarsen to quarterly buckets to densify monthly gaps,
   (c) different signal entirely, or (d) close this signal line. Lag variations / format
   filtering are lower-value until the sparsity is fixed.
2. **Format filter.** Corpus is video-heavy (139 videos vs 85 written articles). If "research"
   means the essays, filter `formats == 'articles'`. Not yet built — trivial to add.
3. **More sources?** Only a16z so far. The project may want a broader crypto-research corpus.
4. **Data in git.** The ~10MB of parquets are committed (data is the deliverable). Revisit if
   the corpus grows — may want to gitignore `data/` and store elsewhere.

## Conventions

Each `projects/*` is its own git repo (this one: branch `main`, no remote yet). scrape.do
token lives in 1Password (`op item get scrape.do --vault local`).
