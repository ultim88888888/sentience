# a16z Research-Coverage Trading Signal — Viability Study

_Design spec · 2026-06-05 · status: approved-pending-review_

## 1. Purpose & hypothesis

The sentience project holds a scraped corpus of 235 a16z crypto research posts
(2017–2026, dense from 2022). This study tests one hypothesis cheaply:

> **When a16z's research coverage rotates *toward* a sector, that sector's tradable
> basket outperforms over the following 1–3 months — relative to the other baskets
> and to L1 majors.**

a16z crypto is a kingmaker and a holder; the plausible mechanism is **attention →
capital flows**, not the research being alpha on its own. This is a **smoke test**, not
a validated strategy. The deliverable is a verdict — *pulse / no pulse / inconclusive* —
not a deployable system. If there is a pulse, an LLM-expert encoder becomes a v2
candidate; if not, we stop.

### Non-goals (v1)
- No LLM / semantic encoding of post bodies. Tag-based attribution only.
- No live/paper trading, no execution modeling beyond a toy backtest.
- No funding/OI signal layering (endpoints validated, deferred to v2).
- No claim of statistical significance — sample is ~40–50 months; we report effect
  size and n side by side and refuse to oversell.

### Two studies
- **Study A — basket level:** does coverage rotation lead *basket* relative returns? (§3.5)
- **Study B — token level:** does a token sitting in multiple *hot* baskets outperform?
  Routes around basket-return correlation by scoring conviction per token. (§3.6)

## 2. Baskets

Six sector baskets, hand-curated and committed as `baskets.yaml` (auditable, editable).
L1 Majors is the **benchmark**, not a tradable signal target.

| Basket | Corpus tags driving it | Token members (Binance USDT perps, validated) |
|---|---|---|
| ZK/Privacy | SNARKs, zero knowledge, cryptography | ZK, STRK, MINA, SCR, MANTA, DUSK, ALT |
| L2/Scaling | scaling & throughput, rollups, data availability | ARB, OP, POL, METIS, STRK, ZK, MANTA |
| DeFi | DeFi, auction design, pricing, mechanism design | UNI, AAVE, CRV, COMP, SNX, SUSHI, DYDX, PENDLE |
| Staking/Consensus | consensus, BFT, proof of stake | LDO, RPL, ETH, ATOM, EIGEN, JTO, ANKR |
| DAOs/Governance | governance, DAOs, voting | UNI, ENS, ARB, OP, AAVE, APE |
| **L1 Majors (benchmark)** | — | BTC, ETH, SOL |

Overlap (e.g. STRK in both ZK and L2; UNI in DeFi and DAOs) is **intentional and
allowed** — a token can belong to multiple baskets. Membership is the editable knob;
the pipeline must not hardcode it.

## 3. Architecture

Six isolated units, each independently testable, wired by a thin `run.py`.

```
baskets.yaml ─┐
              ├─> coverage.py ──> coverage_panel (month × basket: share, momentum)
corpus ───────┘                          │
                                          ├─> signal.py ──> ranked signal panel ─┐
coinglass.py ─> prices ─> returns.py ─> return_panel ─────────────────┬─────────┼─> study.py (A) ─┐
                                                                       │         └─> study_token.py (B) ─┤
                                                                       └──────────────────────────────────> findings.md
```

`study.py` runs Study A (basket level); `study_token.py` runs Study B (token level). Both
execute under each attribution mode (fractional, full) and each aggregation; `run.py`
orchestrates and writes a single `findings.md`.

### 3.1 `coinglass.py` — market data client
- Reads API key from `op://local/d43bafp4am4hfsehpjdjynjipi/credential` via the `op` CLI
  (never hardcoded, never committed).
- Base: `https://open-api-v4.coinglass.com`, header `CG-API-KEY`.
- `price_history(symbol, interval='1d', limit=4500)` → daily OHLC since inception.
  Validated: BTCUSDT → 2019-09, STRKUSDT → its 2024 listing. Max limit 4500/call
  (≈12yr daily) — no pagination needed.
- All timestamps are epoch-ms UTC → convert to tz-aware UTC dates (treat as UTC always).
- On-disk cache (`data/prices/<symbol>_1d.parquet`) so re-runs don't re-hit the API.
- Symbol→pair: `{base}USDT`. Listing dates come from the `supported-exchange-pairs`
  endpoint (`onboard_date`).

### 3.2 `coverage.py` — research signal
- Load corpus (`data/a16z_research/articles.parquet`), bucket posts by **month** on
  `post_date` (UTC).
- Tag→basket map (`baskets.yaml` `tag_map` section). A post attributes to every basket
  whose tags it carries. **Attribution mode** is a parameter run both ways:
  - `fractional` — a post hitting *k* baskets contributes **1/k** to each (no double-count).
  - `full` — the post contributes **1.0** to each basket it touches (double-count allowed).
  The full study runs under both modes; results reported side by side. A verdict that
  flips between modes is itself a finding.
- Per (month, basket): `coverage_share` = basket's attributed weight that month divided by
  the month's total attributed weight (under whichever mode is active).
- `coverage_momentum` = `coverage_share(t) − mean(coverage_share[t−3 : t−1])`.
  First 3 months per basket are NaN (warm-up) and excluded from the test.

### 3.3 `returns.py` — basket returns
- Basket monthly return = **equal-weight mean** of its live constituents' monthly returns
  (month-end close to month-end close).
- A constituent is "live" in month *t* only if it has price data covering that month.
- A basket is **eligible** in month *t* if it has **≥1 live constituent**. No higher gate:
  thin baskets add idiosyncratic noise, which biases the IC *toward zero* (conservative —
  it makes a pulse harder to find, never easier), so dropping them would hide data, not
  clean it. The **live-constituent count per basket-month is recorded and reported**, and
  the survivorship/availability asymmetry (ZK/L2 are 2024+ heavy) is surfaced as a caveat
  rather than silently filtered.

### 3.4 `signal.py` — cross-sectional ranking
- Each month, rank eligible baskets (excluding L1 Majors) by `coverage_momentum`.
- Cross-sectional rank is robust to unequal histories — we ask "which basket is hot vs
  cold *this month*", not "is this basket's raw coverage high."
- Forward relative return = basket return minus the equal-weight mean of eligible
  baskets that month, computed at **t+1** and cumulatively over **t+1..t+3** (both windows).

### 3.5 `study.py` — the test & verdict
- **Primary — Information Coefficient:** Spearman rank corr between `coverage_momentum`
  rank at *t* and forward-relative-return rank, for both windows. Report IC mean, IC
  std, hit rate (sign agreement), and **n (basket-months)**.
- **Secondary — toy backtest:** monthly rebalance, overweight top-2 momentum baskets /
  underweight bottom-2, equal notional. Compare cumulative return vs (a) equal-weight all
  baskets, (b) L1 Majors benchmark. Report cumulative return, ann. vol, max drawdown.
- **Verdict logic:** pulse if IC mean is positive and directionally consistent across both
  windows AND the long/short spread beats equal-weight; no-pulse if IC ≈ 0 or negative;
  inconclusive otherwise. Sample-size caveat printed alongside every number.

### 3.6 `study_token.py` — Study B, token-level conviction
- Motivation: baskets overlap (STRK ∈ ZK ∩ L2, UNI ∈ DeFi ∩ DAOs) so basket returns are
  correlated, softening Study A's long/short. Scoring conviction *per token* routes around
  this: if a16z is leaning into multiple baskets a token belongs to, that token carries
  stacked conviction and should outperform.
- **Token conviction score** at month *t* = aggregate of `coverage_momentum(t)` across
  every basket the token is a member of. Run two aggregations: `sum` (rewards multi-basket
  membership — the STRK thesis) and `mean` (normalizes it) — report both.
- Universe: all unique tokens across the five signal baskets (exclude L1 Majors).
- **Test:** cross-sectional Spearman IC between token conviction rank at *t* and token
  forward return rank at t+1 and t+1..t+3; plus a toy long-top-quartile / short-bottom-
  quartile token backtest vs equal-weight-all-tokens and vs L1 Majors. Same n and
  sample-size caveats as Study A.

## 4. Data flow & storage

- Corpus: read-only, already committed.
- Prices: cached as parquet under `data/prices/` (gitignored — re-fetchable, not a
  deliverable; corpus stays committed because it's the irreplaceable artifact).
- Outputs: `findings.md` (IC table, backtest curve description, verdict) +
  `data/study/` panels (parquet) for inspection. `findings.md` is committed.

## 5. Error handling

- Missing/expired Coinglass key → fail fast with the `op` path in the message.
- Symbol with no perp / empty history → log, skip the constituent, continue (don't crash
  the basket).
- Month with 0 live constituents → basket ineligible that month, logged. Thin baskets
  (1–2 live) are kept but their constituent count is recorded and surfaced in `findings.md`.
- API non-zero `code` → raise with the returned message; cached data preferred on re-run.
- No silent truncation: every dropped token/month is logged and counted in `findings.md`.

## 6. Testing

- `coverage.py`: unit-test fractional split (a 2-tag post → 0.5/0.5) and momentum warm-up
  NaNs on a tiny synthetic corpus.
- `returns.py`: unit-test eligibility (0 live → NaN, ≥1 live → computed), the live-count
  recording, and equal-weight math on fixtures.
- `signal.py`: unit-test cross-sectional ranking + forward-relative-return alignment (no
  lookahead — signal at *t* uses only returns at *t+1*).
- `coinglass.py`: one integration test (live, BTCUSDT, asserts non-empty history reaching
  ≥2020) — gated so it's skipped without a key.
- `coverage.py`: also unit-test both attribution modes — a 2-tag post gives 0.5/0.5 under
  `fractional` and 1.0/1.0 under `full`.
- `study.py` / `study_token.py`: IC computation tested against a hand-constructed
  perfectly-correlated panel (IC≈1) and an anti-correlated one (IC≈−1). Study B: test that
  `sum` aggregation scores a 2-hot-basket token above a 1-hot-basket token.

## 7. Deliverable & success criteria

One reproducible command (`python -m study.run`) produces `findings.md` with, for **both
studies (A basket-level, B token-level)** under **both attribution modes (fractional,
full)**: the IC table (both forward windows, with n), the toy-backtest comparison, the
per-basket-month live-constituent counts, the list of dropped tokens/months, and a
per-study verdict. **Success = an honest, reproducible answer to "is there anything here,"
with the sample-size caveat front and center** — not a positive result. A clean "no pulse"
across both studies is a successful outcome.
