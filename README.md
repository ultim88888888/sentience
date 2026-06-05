# sentience

> Research project. Exploring — direction TBD. Scaffolded 2026-06-04.

Started as a "both / not sure yet" project: may stay a research/exploration space or
concrete into a build. First workstream is corpus collection (a16z crypto research).

## Layout

- `scrapers/` — data collection
  - `a16z_research/` — scrapes all a16zcrypto.com research articles → parquet ([README](scrapers/a16z_research/README.md))
- `data/` — collected datasets (parquet)
- `docs/` — specs, design notes, decisions

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r scrapers/a16z_research/requirements.txt
python -m playwright install chromium
```
