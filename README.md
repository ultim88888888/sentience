# sentience

> Research project. Scaffolded 2026-06-04.

**📈 Featured study: [Turning a16z's Crypto Corpus into a Trading Strategy](docs/CORPUS_SIGNAL_STUDY.md)** —
a full case study (with charts + equity curves) on structuring the a16z team's public corpus into a
tradeable signal. Headline: a council-deliberation signal + momentum gate + regime hedge → Sharpe 1.26
(vs BTC 1.09), +64%/yr alpha, walk-forward validated. Honest caveats included.

Started as a "both / not sure yet" project: may stay a research/exploration space or
concrete into a build. First workstream is corpus collection (a16z crypto research).

## Layout

- `scrapers/` — data collection
  - `a16z_research/` — scrapes all a16zcrypto.com research articles → parquet ([README](scrapers/a16z_research/README.md))
  - `a16z_team/` — scrapes the a16zcrypto.com team roster (name, title, bio, socials) → parquet ([README](scrapers/a16z_team/README.md))
- `data/` — collected datasets (parquet)
- `docs/` — specs, design notes, decisions

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r scrapers/a16z_research/requirements.txt
python -m playwright install chromium
```
