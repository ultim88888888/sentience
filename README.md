# sentience

Research lab for turning unstructured corpora into structured, tradeable signal. Scaffolded 2026-06-04.

## 📈 Featured study — a16z's Crypto Corpus → a Trading Strategy

**Live walkthrough → https://a16z-crypto-case-study.vercel.app** · writeup: [`docs/CORPUS_SIGNAL_STUDY.md`](docs/CORPUS_SIGNAL_STUDY.md)

Can a16z's *internal* edge be reconstructed from the outside? We scrape the a16z crypto team's full public
corpus, rebuild ten of them as time-gated "doppelgangers," extract a single house market view, and trade it.

**Pipeline:** scrape (X, research, transcripts, bios) → clean & structure → per-member **doppelgangers**
(a reasoning "soul" + a ≤T memory feed, rebuilt each interval, lookahead-firewalled) → five signal
approaches (blended LLM · per-member consensus · **council deliberation — the winner** · +market digest ·
+soul) → Coinglass market backbone (liquid universe, sector baskets, OI/TWAP limits, BTC beta) →
walk-forward, beta-aware backtest + significance & validation.

**Deliverable strategy** (`scripts/strategy_final.py`): long the top-3 council-bullish sectors that are
also *outperforming BTC* over 1M **and** 3M (relative-momentum gate); equal-weight liquid baskets; hold
BTC when none qualify; beta-neutralize in bear regimes. Quarterly corpus signal + monthly rebalance.

Full cycle 2021-09 → 2026-05 (56 months, incl. the 2022 bear): **$10mm → ~$266mm (+2556%), Sharpe 1.18
(BTC 0.42), Jensen alpha t ≈ 2.4, β ≈ 0.75, max drawdown −54% (BTC −72%).**

**Honest caveats — the point of the study:**
- The *raw* conviction signal is **pro-cyclical**: it inverts out-of-sample in the 2021–22 bear (council
  sector IC t = +2.4 in-sample → −3.6 OOS). Robustness comes from the relative-momentum *construction*,
  not the raw conviction ranking.
- Roughly half the P&L is BTC-beta; the non-beta alpha rests on ~2 early theme calls that hit (privacy, AI).
- It's a small-AUM edge (alpha decays toward ~$100mm), and t-stats are N-limited.

The deliverable is the **pipeline + an honest characterization** of what it can and can't do — not a
guaranteed Sharpe.

## Layout

- `scrapers/` — data collection: `a16z_research/`, `a16z_team/`, `a16z_transcripts/`, `twitter/`, `linkedin/`
- `doppelganger/` — the doppelganger engine: ingest, soul, memory, respond ([README](doppelganger/README.md))
- `signals/` — extraction (A1–A4), vocab reconciliation, market backbone, backtest, validation ([README](signals/README.md))
- `scripts/` — the alpha hunt, strategy capstone, parameter sweeps, walk-forward
- `site/` — the interactive case-study site (single self-contained page, deployed to Vercel)
- `data/` — collected corpora, signal panels, and market data (parquet)
- `docs/` — specs, plans, the study writeup, and `STATUS.md`
- `tests/` — 300+ unit tests (`pytest`)

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
# each component ships its own requirements.txt — install what you need, e.g.:
pip install -r doppelganger/requirements.txt          # doppelganger engine
pip install -r scrapers/a16z_research/requirements.txt # a research scraper
python -m playwright install chromium                  # for the scrapers
pytest                                                 # run the suite
```

The interactive site needs no build — open `site/index.html` (data is inlined) or visit the live URL above.
