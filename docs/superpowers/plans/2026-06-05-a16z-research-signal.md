# a16z Research-Coverage Trading Signal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible viability study answering whether a16z's research-coverage rotation leads relative crypto-basket and token returns, with an honest pulse/no-pulse verdict.

**Architecture:** A small `study/` Python package. `coinglass.py` pulls daily perp prices (cached). `coverage.py` turns the committed a16z corpus into a monthly coverage-momentum signal per basket (two attribution modes). `returns.py` builds monthly token and basket returns. `signal.py` aligns signal to forward returns with no lookahead. `study_basket.py` (Study A) and `study_token.py` (Study B) compute Information Coefficient + toy backtests. `run.py` orchestrates both studies under both attribution modes and writes `findings.md` + a coverage heatmap.

**Tech Stack:** Python 3.13, pandas 3.0, pyarrow, httpx, PyYAML, scipy, matplotlib, pytest. Secrets via the `op` CLI (1Password). Follows the conventions in `scrapers/a16z_research/` (module-level config, `_log` to stderr, `op item get`).

**Spec:** `docs/superpowers/specs/2026-06-05-a16z-research-signal-design.md`

---

## File Structure

```
study/
  __init__.py
  config.py          # paths, Coinglass constants, op item name, windows
  baskets.yaml       # basket membership + benchmark + tag_map (the editable knob)
  coinglass.py       # market-data client: price_history(symbol) + on-disk cache
  coverage.py        # corpus -> monthly coverage_share + coverage_momentum per basket
  returns.py         # monthly token returns + basket equal-weight returns + n_live
  signal.py          # join coverage<->returns, forward relative returns, no lookahead
  study_basket.py    # Study A: basket-level IC + toy long/short backtest + verdict
  study_token.py     # Study B: token-level conviction IC + quartile backtest + verdict
  findings.py        # render findings.md + coverage heatmap PNG
  run.py             # orchestrate both modes x both studies -> findings.md
  requirements.txt
tests/
  conftest.py        # synthetic corpus + synthetic price fixtures
  test_coverage.py
  test_returns.py
  test_signal.py
  test_study_basket.py
  test_study_token.py
  test_coinglass.py  # one live integration test, skipped without a key
data/
  prices/            # gitignored cache: <SYMBOL>_1d.parquet
  study/             # outputs: panels (parquet) + coverage_heatmap.png
findings.md          # committed deliverable
```

**Data contracts (locked here, referenced by every task):**
- `monthly_coverage(...) -> DataFrame[month: Period[M], basket: str, coverage_share: float, coverage_momentum: float]`
- `monthly_token_returns(...) -> DataFrame[month: Period[M], token: str, ret: float]`
- `basket_returns(...) -> DataFrame[month: Period[M], basket: str, ret: float, n_live: int]`
- `basket_signal_panel(...) -> DataFrame[month, basket, coverage_momentum, fwd_rel_1m, fwd_rel_3m]`
- `token_conviction(...) -> DataFrame[month: Period[M], token: str, conviction: float]`
- `information_coefficient(panel, signal_col, fwd_col, group='month') -> dict{ic_mean, ic_std, hit_rate, n}`

`month` is always a pandas `Period[M]` (monthly). All datetimes are UTC.

---

## Task 0: Scaffold package, config, baskets.yaml

**Files:**
- Create: `study/__init__.py` (empty)
- Create: `study/requirements.txt`
- Create: `study/config.py`
- Create: `study/baskets.yaml`
- Create: `tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create `study/requirements.txt`**

```
httpx==0.28.1
pandas==3.0.3
pyarrow==24.0.0
PyYAML==6.0.2
scipy==1.15.1
matplotlib==3.10.0
pytest==8.3.4
```

- [ ] **Step 2: Create `study/__init__.py`** (empty file)

- [ ] **Step 3: Create `study/config.py`**

```python
"""Configuration for the a16z research-coverage trading-signal study."""
from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PARQUET = DATA_DIR / "a16z_research" / "articles.parquet"
PRICE_CACHE_DIR = DATA_DIR / "prices"
STUDY_DIR = DATA_DIR / "study"
BASKETS_YAML = Path(__file__).resolve().parent / "baskets.yaml"
FINDINGS_MD = PROJECT_ROOT / "findings.md"
HEATMAP_PNG = STUDY_DIR / "coverage_heatmap.png"

# --- Coinglass ---
COINGLASS_BASE = "https://open-api-v4.coinglass.com"
COINGLASS_OP_ITEM = "coinglass"
COINGLASS_OP_VAULT = "local"
EXCHANGE = "Binance"
PRICE_INTERVAL = "1d"
PRICE_LIMIT = 4500          # ~12yr daily; validated max per call
HTTP_TIMEOUT = 60.0

# --- signal params ---
MOMENTUM_LOOKBACK = 3        # months of trailing average for coverage_momentum
FWD_WINDOWS = (1, 3)         # forward-return windows in months
ATTRIBUTION_MODES = ("fractional", "full")
CONVICTION_AGGS = ("sum", "mean")
```

- [ ] **Step 4: Create `study/baskets.yaml`** (the editable knob — membership + tag map)

```yaml
# Tokens are Binance USDT-perp base assets (validated to exist). Overlap is intentional.
baskets:
  ZK/Privacy:        [ZK, STRK, MINA, SCR, MANTA, DUSK, ALT]
  L2/Scaling:        [ARB, OP, POL, METIS, STRK, ZK, MANTA]
  DeFi:              [UNI, AAVE, CRV, COMP, SNX, SUSHI, DYDX, PENDLE]
  Staking/Consensus: [LDO, RPL, ETH, ATOM, EIGEN, JTO, ANKR]
  DAOs/Governance:   [UNI, ENS, ARB, OP, AAVE, APE]
  L1 Majors:         [BTC, ETH, SOL]

# L1 Majors is the benchmark, not a signal target. It has no tag_map entry.
benchmark: L1 Majors

# Exact corpus tag strings -> basket. A post attributes to a basket if it carries any tag here.
tag_map:
  ZK/Privacy:        ["SNARKs", "zero knowledge & succinct proof systems", "cryptography"]
  L2/Scaling:        ["scaling & throughput", "rollups", "data availability"]
  DeFi:              ["DeFi", "auction design", "pricing", "mechanism design"]
  Staking/Consensus: ["consensus", "BFT", "proof of stake"]
  DAOs/Governance:   ["governance", "DAOs", "voting"]
```

- [ ] **Step 5: Append to `.gitignore`**

```
projects/sentience/data/prices/
projects/sentience/data/study/
```

(If editing the sentience repo's own `.gitignore`, use repo-relative paths `data/prices/` and `data/study/` instead. The price cache and intermediate panels are re-fetchable; only `findings.md` is committed.)

- [ ] **Step 6: Create `tests/conftest.py`** (shared synthetic fixtures — no network, no real corpus)

```python
"""Synthetic fixtures: a tiny corpus and tiny price history, fully deterministic."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def baskets_cfg():
    return {
        "baskets": {
            "ZK/Privacy": ["ZK", "STRK"],
            "L2/Scaling": ["ARB", "STRK"],   # STRK overlaps ZK on purpose
            "DeFi": ["UNI", "AAVE"],
            "L1 Majors": ["BTC", "ETH"],
        },
        "benchmark": "L1 Majors",
        "tag_map": {
            "ZK/Privacy": ["SNARKs", "cryptography"],
            "L2/Scaling": ["rollups"],
            "DeFi": ["DeFi"],
        },
    }


@pytest.fixture
def synthetic_corpus():
    # 6 monthly posts; tags chosen so attribution is hand-checkable.
    rows = [
        ("2024-01-10", ["SNARKs"]),               # ZK only
        ("2024-01-20", ["SNARKs", "rollups"]),    # ZK + L2 (multi-tag)
        ("2024-02-05", ["DeFi"]),                 # DeFi only
        ("2024-02-15", ["rollups"]),              # L2 only
        ("2024-03-01", ["cryptography", "DeFi"]), # ZK + DeFi
        ("2024-03-10", ["unmapped tag"]),         # attributes to nothing
    ]
    return pd.DataFrame(
        {"post_date": [r[0] for r in rows],
         "tags": [np.array(r[1], dtype=object) for r in rows]}
    )


@pytest.fixture
def synthetic_token_returns():
    # month x token monthly returns, long form.
    months = pd.period_range("2024-01", "2024-04", freq="M")
    data = {"ZK": [0.10, 0.20, -0.05, 0.00], "STRK": [0.15, 0.25, 0.00, 0.10],
            "ARB": [0.05, 0.10, 0.05, 0.05], "UNI": [-0.10, -0.05, 0.10, 0.00],
            "AAVE": [-0.05, 0.00, 0.05, 0.05], "BTC": [0.02, 0.03, 0.01, 0.02],
            "ETH": [0.03, 0.04, 0.00, 0.01]}
    recs = [{"month": m, "token": t, "ret": data[t][i]}
            for i, m in enumerate(months) for t in data]
    return pd.DataFrame.from_records(recs)
```

- [ ] **Step 7: Commit**

```bash
git add study/__init__.py study/requirements.txt study/config.py study/baskets.yaml tests/conftest.py .gitignore
git commit -m "feat(study): scaffold package, config, baskets.yaml, test fixtures"
```

---

## Task 1: Coinglass market-data client

**Files:**
- Create: `study/coinglass.py`
- Test: `tests/test_coinglass.py`

- [ ] **Step 1: Write the integration test** (live, gated — skipped without a key)

```python
# tests/test_coinglass.py
import shutil
import pandas as pd
import pytest
from study import coinglass


def _have_key():
    return shutil.which("op") is not None


@pytest.mark.skipif(not _have_key(), reason="op CLI / Coinglass key unavailable")
def test_price_history_btc_reaches_back_to_2020():
    df = coinglass.price_history("BTC", use_cache=False)
    assert not df.empty
    assert set(["date", "close"]).issubset(df.columns)
    assert df["date"].min() <= pd.Timestamp("2020-12-31", tz="UTC")
    assert (df["close"] > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_coinglass.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module 'study.coinglass' has no attribute 'price_history'`

- [ ] **Step 3: Write `study/coinglass.py`**

```python
"""Coinglass v4 client: daily perp OHLC since inception, with an on-disk parquet cache.

Key lives in 1Password (vault 'local', item 'coinglass'); read via the `op` CLI, never
hardcoded. Validated: BTCUSDT history reaches 2019-09, STRKUSDT starts at its 2024 listing.
"""
import subprocess
import sys

import httpx
import pandas as pd

from .config import (COINGLASS_BASE, COINGLASS_OP_ITEM, COINGLASS_OP_VAULT, EXCHANGE,
                     HTTP_TIMEOUT, PRICE_CACHE_DIR, PRICE_INTERVAL, PRICE_LIMIT)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def api_key() -> str:
    """Read the Coinglass API key from 1Password (vault 'local')."""
    return subprocess.check_output(
        ["op", "item", "get", COINGLASS_OP_ITEM, "--vault", COINGLASS_OP_VAULT,
         "--fields", "credential", "--reveal"], text=True).strip()


def _symbol_to_pair(symbol: str) -> str:
    return f"{symbol}USDT"


def price_history(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLC for a token's Binance USDT perp. Returns columns [date, close] (UTC).

    Empty DataFrame (same columns) if the pair has no data — caller skips the constituent.
    """
    cache = PRICE_CACHE_DIR / f"{symbol}_{PRICE_INTERVAL}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    pair = _symbol_to_pair(symbol)
    url = f"{COINGLASS_BASE}/api/futures/price/history"
    params = {"exchange": EXCHANGE, "symbol": pair,
              "interval": PRICE_INTERVAL, "limit": PRICE_LIMIT}
    r = httpx.get(url, params=params, headers={"CG-API-KEY": api_key()},
                  timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"Coinglass error for {pair}: {payload.get('msg')}")

    rows = payload.get("data") or []
    if not rows:
        _log(f"  no price history for {symbol} ({pair}); skipping")
        empty = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns, UTC]"),
                              "close": pd.Series([], dtype="float64")})
        return empty

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)

    if use_cache:
        PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_coinglass.py -v`
Expected: PASS (or SKIP if no `op`). To confirm live, run with a key present.

- [ ] **Step 5: Commit**

```bash
git add study/coinglass.py tests/test_coinglass.py
git commit -m "feat(study): Coinglass client with cached daily perp price history"
```

---

## Task 2: Coverage signal from corpus

**Files:**
- Create: `study/coverage.py`
- Test: `tests/test_coverage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_coverage.py
import pandas as pd
from study import coverage


def test_fractional_split_halves_a_two_basket_post(synthetic_corpus, baskets_cfg):
    cov = coverage.monthly_coverage(synthetic_corpus, baskets_cfg, mode="fractional")
    jan = cov[cov["month"] == pd.Period("2024-01", "M")].set_index("basket")["coverage_share"]
    # Jan posts: post1 -> ZK(1.0); post2 -> ZK(0.5)+L2(0.5). Totals: ZK=1.5, L2=0.5, sum=2.0
    assert abs(jan["ZK/Privacy"] - 0.75) < 1e-9
    assert abs(jan["L2/Scaling"] - 0.25) < 1e-9


def test_full_mode_double_counts(synthetic_corpus, baskets_cfg):
    cov = coverage.monthly_coverage(synthetic_corpus, baskets_cfg, mode="full")
    jan = cov[cov["month"] == pd.Period("2024-01", "M")].set_index("basket")["coverage_share"]
    # Jan full weights: ZK=1+1=2, L2=1; share denominator = total weight = 3
    assert abs(jan["ZK/Privacy"] - 2/3) < 1e-9
    assert abs(jan["L2/Scaling"] - 1/3) < 1e-9


def test_momentum_warmup_is_nan(synthetic_corpus, baskets_cfg):
    cov = coverage.monthly_coverage(synthetic_corpus, baskets_cfg, mode="fractional")
    # Only 3 months of data; with lookback=3 every momentum value is NaN (no full window).
    assert cov["coverage_momentum"].isna().all()


def test_unmapped_tags_attribute_nowhere(synthetic_corpus, baskets_cfg):
    cov = coverage.monthly_coverage(synthetic_corpus, baskets_cfg, mode="fractional")
    mar = cov[cov["month"] == pd.Period("2024-03", "M")]
    # March: post5 -> ZK+DeFi; post6 -> unmapped (drops). So only ZK & DeFi present, each 0.5.
    shares = mar.set_index("basket")["coverage_share"]
    assert abs(shares["ZK/Privacy"] - 0.5) < 1e-9
    assert abs(shares["DeFi"] - 0.5) < 1e-9
    assert "L2/Scaling" not in shares.index or shares.get("L2/Scaling", 0) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_coverage.py -v`
Expected: FAIL with `AttributeError: module 'study.coverage' has no attribute 'monthly_coverage'`

- [ ] **Step 3: Write `study/coverage.py`**

```python
"""Turn the a16z corpus into a monthly coverage-momentum signal per basket.

Uses only post_date + tags (all 235 posts qualify regardless of body depth). Every post
weights 1.0 (uniform across formats — see spec section 3.2). Two attribution modes:
  fractional: a post hitting k baskets contributes 1/k to each.
  full:       the post contributes 1.0 to each basket it touches.
"""
import pandas as pd

from .config import CORPUS_PARQUET, MOMENTUM_LOOKBACK


def load_corpus() -> pd.DataFrame:
    df = pd.read_parquet(CORPUS_PARQUET, columns=["post_date", "tags"])
    return df


def _signal_baskets(baskets_cfg: dict) -> list[str]:
    return list(baskets_cfg["tag_map"].keys())  # benchmark has no tag_map -> excluded


def _attribute(tags, tag_map: dict, mode: str) -> dict[str, float]:
    """Return {basket: weight} for one post's tags under the given mode."""
    tagset = set(tags)
    hit = [b for b, btags in tag_map.items() if tagset & set(btags)]
    if not hit:
        return {}
    w = 1.0 / len(hit) if mode == "fractional" else 1.0
    return {b: w for b in hit}


def monthly_coverage(corpus: pd.DataFrame, baskets_cfg: dict, mode: str) -> pd.DataFrame:
    if mode not in ("fractional", "full"):
        raise ValueError(f"unknown attribution mode: {mode}")
    tag_map = baskets_cfg["tag_map"]
    baskets = _signal_baskets(baskets_cfg)

    df = corpus.copy()
    df["month"] = pd.to_datetime(df["post_date"], utc=True).dt.to_period("M")

    recs = []
    for _, row in df.iterrows():
        for basket, w in _attribute(row["tags"], tag_map, mode).items():
            recs.append({"month": row["month"], "basket": basket, "weight": w})
    weights = pd.DataFrame.from_records(recs)

    # full grid of (month, basket) so absent baskets are explicit zeros
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    grid = pd.MultiIndex.from_product([months, baskets], names=["month", "basket"])

    by_mb = weights.groupby(["month", "basket"])["weight"].sum()
    by_month_total = weights.groupby("month")["weight"].sum()

    cov = by_mb.reindex(grid, fill_value=0.0).rename("w").reset_index()
    cov["total"] = cov["month"].map(by_month_total).fillna(0.0)
    cov["coverage_share"] = (cov["w"] / cov["total"]).where(cov["total"] > 0, 0.0)

    cov = cov.sort_values(["basket", "month"]).reset_index(drop=True)
    # momentum = share(t) - trailing mean over the prior LOOKBACK months (per basket)
    cov["coverage_momentum"] = cov.groupby("basket")["coverage_share"].transform(
        lambda s: s - s.shift(1).rolling(MOMENTUM_LOOKBACK).mean()
    )
    return cov[["month", "basket", "coverage_share", "coverage_momentum"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_coverage.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add study/coverage.py tests/test_coverage.py
git commit -m "feat(study): monthly coverage signal with fractional/full attribution modes"
```

---

## Task 3: Token and basket returns

**Files:**
- Create: `study/returns.py`
- Test: `tests/test_returns.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_returns.py
import pandas as pd
from study import returns


def test_basket_returns_equal_weight_and_n_live(synthetic_token_returns, baskets_cfg):
    br = returns.basket_returns(synthetic_token_returns, baskets_cfg)
    jan_zk = br[(br["month"] == pd.Period("2024-01", "M")) & (br["basket"] == "ZK/Privacy")]
    # ZK basket = [ZK, STRK]; Jan returns 0.10 & 0.15 -> mean 0.125, n_live 2
    assert abs(jan_zk["ret"].iloc[0] - 0.125) < 1e-9
    assert jan_zk["n_live"].iloc[0] == 2


def test_basket_ineligible_when_no_live_constituents(baskets_cfg):
    # DeFi tokens (UNI, AAVE) have no rows -> DeFi should be absent / NaN, not crash.
    tr = pd.DataFrame({"month": [pd.Period("2024-01", "M")], "token": ["BTC"], "ret": [0.02]})
    br = returns.basket_returns(tr, baskets_cfg)
    defi = br[br["basket"] == "DeFi"]
    assert defi.empty or defi["n_live"].fillna(0).eq(0).all()


def test_monthly_token_returns_from_prices():
    # Two month-end closes -> one monthly return.
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-02-29"], utc=True),
        "close": [100.0, 110.0],
    })
    tr = returns._monthly_returns_from_prices(prices)
    feb = tr[tr["month"] == pd.Period("2024-02", "M")]["ret"].iloc[0]
    assert abs(feb - 0.10) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_returns.py -v`
Expected: FAIL with `AttributeError` (functions undefined)

- [ ] **Step 3: Write `study/returns.py`**

```python
"""Monthly returns: per token (from cached prices) and per basket (equal-weight, n_live)."""
import sys

import pandas as pd

from . import coinglass


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """[date, close] (daily, UTC) -> [month, ret] using month-end close to month-end close."""
    if prices.empty:
        return pd.DataFrame({"month": pd.Series([], dtype="period[M]"),
                             "ret": pd.Series([], dtype="float64")})
    s = prices.set_index("date")["close"].sort_index()
    month_end = s.resample("ME").last()
    ret = month_end.pct_change()
    out = ret.dropna().rename("ret").reset_index()
    out["month"] = out["date"].dt.to_period("M")
    return out[["month", "ret"]]


def monthly_token_returns(tokens: list[str], use_cache: bool = True) -> pd.DataFrame:
    """[month, token, ret] for every token that has price history; missing tokens skipped."""
    frames = []
    for tok in sorted(set(tokens)):
        prices = coinglass.price_history(tok, use_cache=use_cache)
        tr = _monthly_returns_from_prices(prices)
        if tr.empty:
            _log(f"  {tok}: no returns (no price history)")
            continue
        tr["token"] = tok
        frames.append(tr)
    if not frames:
        return pd.DataFrame({"month": pd.Series([], dtype="period[M]"),
                             "token": pd.Series([], dtype="object"),
                             "ret": pd.Series([], dtype="float64")})
    return pd.concat(frames, ignore_index=True)[["month", "token", "ret"]]


def basket_returns(token_returns: pd.DataFrame, baskets_cfg: dict) -> pd.DataFrame:
    """Equal-weight mean of live constituents per (month, basket), with n_live count.

    A basket-month with 0 live constituents is omitted (eligibility = >=1 live, per spec).
    """
    membership = baskets_cfg["baskets"]
    recs = []
    for basket, tokens in membership.items():
        sub = token_returns[token_returns["token"].isin(tokens)]
        if sub.empty:
            continue
        g = sub.groupby("month")["ret"].agg(ret="mean", n_live="count").reset_index()
        g["basket"] = basket
        recs.append(g)
    if not recs:
        return pd.DataFrame({"month": pd.Series([], dtype="period[M]"),
                             "basket": pd.Series([], dtype="object"),
                             "ret": pd.Series([], dtype="float64"),
                             "n_live": pd.Series([], dtype="int64")})
    out = pd.concat(recs, ignore_index=True)
    return out[["month", "basket", "ret", "n_live"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_returns.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add study/returns.py tests/test_returns.py
git commit -m "feat(study): monthly token & basket returns with n_live eligibility"
```

---

## Task 4: Signal panel — forward relative returns, no lookahead

**Files:**
- Create: `study/signal.py`
- Test: `tests/test_signal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_signal.py
import numpy as np
import pandas as pd
from study import signal


def _toy_inputs():
    months = pd.period_range("2024-01", "2024-04", freq="M")
    coverage = pd.DataFrame([
        {"month": m, "basket": b, "coverage_share": 0.0,
         "coverage_momentum": v}
        for m, vs in zip(months, [(0.1, -0.1), (0.2, -0.2), (0.0, 0.0), (0.0, 0.0)])
        for b, v in zip(["A", "B"], vs)
    ])
    basket_rets = pd.DataFrame([
        {"month": m, "basket": b, "ret": v, "n_live": 2}
        for m, vs in zip(months, [(0.05, 0.01), (0.06, 0.00), (0.02, 0.03), (0.01, 0.02)])
        for b, v in zip(["A", "B"], vs)
    ])
    return coverage, basket_rets


def test_forward_relative_return_no_lookahead():
    coverage, basket_rets = _toy_inputs()
    panel = signal.basket_signal_panel(coverage, basket_rets, benchmark="L1 Majors")
    # fwd_rel_1m at month=2024-01 must use month=2024-02 returns, demeaned across baskets.
    jan = panel[panel["month"] == pd.Period("2024-01", "M")].set_index("basket")
    feb_mean = (0.06 + 0.00) / 2
    assert abs(jan.loc["A", "fwd_rel_1m"] - (0.06 - feb_mean)) < 1e-9
    assert abs(jan.loc["B", "fwd_rel_1m"] - (0.00 - feb_mean)) < 1e-9


def test_last_month_forward_return_is_nan():
    coverage, basket_rets = _toy_inputs()
    panel = signal.basket_signal_panel(coverage, basket_rets, benchmark="L1 Majors")
    apr = panel[panel["month"] == pd.Period("2024-04", "M")]
    assert apr["fwd_rel_1m"].isna().all()  # no month after April
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_signal.py -v`
Expected: FAIL with `AttributeError: module 'study.signal' has no attribute 'basket_signal_panel'`

- [ ] **Step 3: Write `study/signal.py`**

```python
"""Join coverage signal to basket returns and compute forward relative returns.

No lookahead: the signal at month t is paired with returns realized at t+1 (and the
cumulative t+1..t+W). Relative = basket return minus the cross-sectional mean of eligible
baskets that month. The benchmark basket is excluded from the signal universe.
"""
import pandas as pd

from .config import FWD_WINDOWS


def _forward_cumulative(rets_wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """rets_wide: index=month (sorted), columns=basket. Returns forward cumulative return
    over months [t+1 .. t+window], aligned back to t."""
    fwd = (1.0 + rets_wide).rolling(window).apply(lambda x: x.prod(), raw=True) - 1.0
    # rolling is backward-looking and ends at t; shift up by `window` so it sits at t.
    return fwd.shift(-window)


def basket_signal_panel(coverage: pd.DataFrame, basket_rets: pd.DataFrame,
                        benchmark: str) -> pd.DataFrame:
    sig = coverage[coverage["basket"] != benchmark].copy()
    rets = basket_rets[basket_rets["basket"] != benchmark].copy()

    rets_wide = rets.pivot(index="month", columns="basket", values="ret").sort_index()

    out = sig.copy()
    for w in FWD_WINDOWS:
        fwd = _forward_cumulative(rets_wide, w)            # month x basket, forward cum ret
        rel = fwd.sub(fwd.mean(axis=1), axis=0)            # demean cross-sectionally per month
        rel_long = rel.stack().rename(f"fwd_rel_{w}m").reset_index()
        out = out.merge(rel_long, on=["month", "basket"], how="left")

    keep = ["month", "basket", "coverage_momentum"] + [f"fwd_rel_{w}m" for w in FWD_WINDOWS]
    return out[keep]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_signal.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add study/signal.py tests/test_signal.py
git commit -m "feat(study): signal panel with no-lookahead forward relative returns"
```

---

## Task 5: Study A — basket-level IC, backtest, verdict

**Files:**
- Create: `study/study_basket.py`
- Test: `tests/test_study_basket.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_study_basket.py
import numpy as np
import pandas as pd
from study import study_basket


def _panel(corr_sign):
    """Build a panel where signal rank perfectly (anti)correlates with fwd return rank."""
    months = pd.period_range("2024-01", "2024-06", freq="M")
    rows = []
    for m in months:
        for i, b in enumerate(["A", "B", "C"]):
            mom = float(i)
            fwd = corr_sign * float(i)
            rows.append({"month": m, "basket": b, "coverage_momentum": mom,
                         "fwd_rel_1m": fwd, "fwd_rel_3m": fwd})
    return pd.DataFrame(rows)


def test_ic_positive_when_signal_predicts():
    ic = study_basket.information_coefficient(_panel(+1), "coverage_momentum", "fwd_rel_1m")
    assert ic["ic_mean"] > 0.99
    assert ic["n"] == 6


def test_ic_negative_when_signal_anticorrelated():
    ic = study_basket.information_coefficient(_panel(-1), "coverage_momentum", "fwd_rel_1m")
    assert ic["ic_mean"] < -0.99


def test_verdict_pulse_on_positive_consistent():
    panel = _panel(+1)
    res = study_basket.run_study_a(panel)
    assert res["verdict"] == "pulse"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_study_basket.py -v`
Expected: FAIL with `AttributeError` (functions undefined)

- [ ] **Step 3: Write `study/study_basket.py`**

```python
"""Study A (basket level): does coverage-momentum rank lead forward relative-return rank?

Information Coefficient = mean over months of the cross-sectional Spearman correlation
between signal rank and forward-return rank. Plus a toy long/short backtest. Sample size
is small (~40-50 basket-months); we report n alongside every statistic and never claim
significance.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import FWD_WINDOWS


def information_coefficient(panel: pd.DataFrame, signal_col: str, fwd_col: str,
                            group: str = "month") -> dict:
    ics, hits = [], []
    for _, g in panel.dropna(subset=[signal_col, fwd_col]).groupby(group):
        if len(g) < 2:
            continue
        rho, _ = spearmanr(g[signal_col], g[fwd_col])
        if np.isnan(rho):
            continue
        ics.append(rho)
        # hit rate: did the top-ranked-signal basket beat the cross-sectional median fwd?
        top = g.loc[g[signal_col].idxmax()]
        hits.append(1.0 if top[fwd_col] > g[fwd_col].median() else 0.0)
    ics = np.array(ics)
    return {"ic_mean": float(ics.mean()) if len(ics) else float("nan"),
            "ic_std": float(ics.std(ddof=1)) if len(ics) > 1 else float("nan"),
            "hit_rate": float(np.mean(hits)) if hits else float("nan"),
            "n": int(len(ics))}


def toy_backtest(panel: pd.DataFrame, fwd_col: str, top_k: int = 2) -> dict:
    """Long top-k momentum baskets, short bottom-k, equal notional, per month.
    Spread return = mean(long fwd_rel) - mean(short fwd_rel), averaged over months."""
    spreads = []
    for _, g in panel.dropna(subset=["coverage_momentum", fwd_col]).groupby("month"):
        if len(g) < 2 * top_k:
            continue
        ordered = g.sort_values("coverage_momentum", ascending=False)
        longs = ordered.head(top_k)[fwd_col].mean()
        shorts = ordered.tail(top_k)[fwd_col].mean()
        spreads.append(longs - shorts)
    spreads = np.array(spreads)
    return {"mean_spread": float(spreads.mean()) if len(spreads) else float("nan"),
            "n_months": int(len(spreads))}


def run_study_a(panel: pd.DataFrame) -> dict:
    ic = {w: information_coefficient(panel, "coverage_momentum", f"fwd_rel_{w}m")
          for w in FWD_WINDOWS}
    bt = {w: toy_backtest(panel, f"fwd_rel_{w}m") for w in FWD_WINDOWS}
    pos_consistent = all(ic[w]["ic_mean"] > 0 for w in FWD_WINDOWS
                         if not np.isnan(ic[w]["ic_mean"]))
    spread_ok = any(bt[w]["mean_spread"] > 0 for w in FWD_WINDOWS
                    if not np.isnan(bt[w]["mean_spread"]))
    any_signal = any(not np.isnan(ic[w]["ic_mean"]) for w in FWD_WINDOWS)
    if not any_signal:
        verdict = "inconclusive"
    elif pos_consistent and spread_ok:
        verdict = "pulse"
    elif all(ic[w]["ic_mean"] <= 0 for w in FWD_WINDOWS if not np.isnan(ic[w]["ic_mean"])):
        verdict = "no pulse"
    else:
        verdict = "inconclusive"
    return {"ic": ic, "backtest": bt, "verdict": verdict}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_study_basket.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add study/study_basket.py tests/test_study_basket.py
git commit -m "feat(study): Study A basket-level IC, toy backtest, verdict"
```

---

## Task 6: Study B — token-level conviction

**Files:**
- Create: `study/study_token.py`
- Test: `tests/test_study_token.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_study_token.py
import pandas as pd
from study import study_token


def test_sum_aggregation_rewards_multi_basket_membership(baskets_cfg):
    # STRK is in ZK/Privacy AND L2/Scaling; ARB only in L2. With both baskets hot,
    # STRK conviction (sum) must exceed ARB conviction.
    coverage = pd.DataFrame([
        {"month": pd.Period("2024-02", "M"), "basket": "ZK/Privacy", "coverage_momentum": 0.3},
        {"month": pd.Period("2024-02", "M"), "basket": "L2/Scaling", "coverage_momentum": 0.2},
        {"month": pd.Period("2024-02", "M"), "basket": "DeFi", "coverage_momentum": 0.0},
    ])
    conv = study_token.token_conviction(coverage, baskets_cfg, agg="sum")
    feb = conv[conv["month"] == pd.Period("2024-02", "M")].set_index("token")["conviction"]
    assert feb["STRK"] == 0.5   # 0.3 + 0.2
    assert feb["ARB"] == 0.2
    assert feb["STRK"] > feb["ARB"]


def test_mean_aggregation_normalizes(baskets_cfg):
    coverage = pd.DataFrame([
        {"month": pd.Period("2024-02", "M"), "basket": "ZK/Privacy", "coverage_momentum": 0.3},
        {"month": pd.Period("2024-02", "M"), "basket": "L2/Scaling", "coverage_momentum": 0.2},
    ])
    conv = study_token.token_conviction(coverage, baskets_cfg, agg="mean")
    feb = conv[conv["month"] == pd.Period("2024-02", "M")].set_index("token")["conviction"]
    assert abs(feb["STRK"] - 0.25) < 1e-9   # mean(0.3, 0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_study_token.py -v`
Expected: FAIL with `AttributeError: module 'study.study_token' has no attribute 'token_conviction'`

- [ ] **Step 3: Write `study/study_token.py`**

```python
"""Study B (token level): does a token in multiple HOT baskets outperform?

Token conviction at month t = aggregate of coverage_momentum across every basket the
token belongs to (sum rewards multi-basket membership; mean normalizes it). Then the same
IC + a long-top-quartile / short-bottom-quartile backtest, at token granularity. This
routes around basket-return correlation (overlapping membership). Reuses Study A's IC.
"""
import numpy as np
import pandas as pd

from .config import FWD_WINDOWS
from .study_basket import information_coefficient


def _token_to_baskets(baskets_cfg: dict) -> dict[str, list[str]]:
    benchmark = baskets_cfg.get("benchmark")
    out: dict[str, list[str]] = {}
    for basket, tokens in baskets_cfg["baskets"].items():
        if basket == benchmark:
            continue
        for t in tokens:
            out.setdefault(t, []).append(basket)
    return out


def token_conviction(coverage: pd.DataFrame, baskets_cfg: dict, agg: str) -> pd.DataFrame:
    if agg not in ("sum", "mean"):
        raise ValueError(f"unknown aggregation: {agg}")
    membership = _token_to_baskets(baskets_cfg)
    mom = coverage.set_index(["month", "basket"])["coverage_momentum"]

    recs = []
    for month in coverage["month"].unique():
        for token, baskets in membership.items():
            vals = [mom.get((month, b)) for b in baskets]
            vals = [v for v in vals if v is not None and not pd.isna(v)]
            if not vals:
                continue
            score = float(np.sum(vals)) if agg == "sum" else float(np.mean(vals))
            recs.append({"month": month, "token": token, "conviction": score})
    return pd.DataFrame.from_records(recs)


def _forward_token_relative(token_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """[month, token, fwd_rel_{w}m] — forward cum return over t+1..t+window, demeaned
    cross-sectionally across tokens each month."""
    wide = token_returns.pivot(index="month", columns="token", values="ret").sort_index()
    fwd = (1.0 + wide).rolling(window).apply(lambda x: x.prod(), raw=True) - 1.0
    fwd = fwd.shift(-window)
    rel = fwd.sub(fwd.mean(axis=1), axis=0)
    return rel.stack().rename(f"fwd_rel_{window}m").reset_index()


def run_study_b(coverage: pd.DataFrame, token_returns: pd.DataFrame,
                baskets_cfg: dict, agg: str) -> dict:
    conv = token_conviction(coverage, baskets_cfg, agg)
    panel = conv.copy()
    for w in FWD_WINDOWS:
        rel = _forward_token_relative(token_returns, w)
        panel = panel.merge(rel, on=["month", "token"], how="left")

    ic = {w: information_coefficient(panel, "conviction", f"fwd_rel_{w}m")
          for w in FWD_WINDOWS}
    # quartile backtest: long top quartile conviction, short bottom quartile
    spreads = {w: _quartile_spread(panel, f"fwd_rel_{w}m") for w in FWD_WINDOWS}
    pos = all(ic[w]["ic_mean"] > 0 for w in FWD_WINDOWS if not np.isnan(ic[w]["ic_mean"]))
    any_signal = any(not np.isnan(ic[w]["ic_mean"]) for w in FWD_WINDOWS)
    if not any_signal:
        verdict = "inconclusive"
    elif pos and any(s > 0 for s in spreads.values() if not np.isnan(s)):
        verdict = "pulse"
    elif all(ic[w]["ic_mean"] <= 0 for w in FWD_WINDOWS if not np.isnan(ic[w]["ic_mean"])):
        verdict = "no pulse"
    else:
        verdict = "inconclusive"
    return {"agg": agg, "ic": ic, "quartile_spread": spreads, "verdict": verdict}


def _quartile_spread(panel: pd.DataFrame, fwd_col: str) -> float:
    spreads = []
    for _, g in panel.dropna(subset=["conviction", fwd_col]).groupby("month"):
        if len(g) < 4:
            continue
        q = g["conviction"].quantile([0.25, 0.75])
        longs = g[g["conviction"] >= q[0.75]][fwd_col].mean()
        shorts = g[g["conviction"] <= q[0.25]][fwd_col].mean()
        spreads.append(longs - shorts)
    return float(np.mean(spreads)) if spreads else float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_study_token.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add study/study_token.py tests/test_study_token.py
git commit -m "feat(study): Study B token-level conviction IC and quartile backtest"
```

---

## Task 7: Findings renderer + coverage heatmap

**Files:**
- Create: `study/findings.py`
- Test: `tests/test_findings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_findings.py
import pandas as pd
from study import findings


def test_render_findings_includes_verdict_and_n():
    results = {
        "fractional": {
            "study_a": {"ic": {1: {"ic_mean": 0.12, "ic_std": 0.3, "hit_rate": 0.6, "n": 42},
                               3: {"ic_mean": 0.08, "ic_std": 0.3, "hit_rate": 0.55, "n": 40}},
                        "backtest": {1: {"mean_spread": 0.01, "n_months": 42},
                                     3: {"mean_spread": 0.02, "n_months": 40}},
                        "verdict": "pulse"},
            "study_b": {"sum": {"agg": "sum",
                                "ic": {1: {"ic_mean": 0.05, "ic_std": 0.2, "hit_rate": 0.5, "n": 50},
                                       3: {"ic_mean": 0.04, "ic_std": 0.2, "hit_rate": 0.5, "n": 48}},
                                "quartile_spread": {1: 0.01, 3: 0.02}, "verdict": "inconclusive"}},
        }
    }
    md = findings.render_markdown(results)
    assert "pulse" in md
    assert "n=42" in md
    assert "Sample size" in md  # the caveat must be present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_findings.py -v`
Expected: FAIL with `AttributeError: module 'study.findings' has no attribute 'render_markdown'`

- [ ] **Step 3: Write `study/findings.py`**

```python
"""Render findings.md and the coverage heatmap. The sample-size caveat is non-negotiable
and printed alongside every verdict."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FWD_WINDOWS, HEATMAP_PNG, STUDY_DIR

CAVEAT = ("> **Sample size caveat.** This is a smoke test over ~40-50 basket-months. "
          "No result here is statistically significant — at best it is suggestive. "
          "Effect size and n are reported together; a positive IC means 'worth a real "
          "study', not 'alpha found'.")


def _ic_row(label: str, ic: dict) -> str:
    cells = " | ".join(
        f"{ic[w]['ic_mean']:+.3f} (n={ic[w]['n']})" for w in FWD_WINDOWS)
    return f"| {label} | {cells} |"


def render_markdown(results: dict) -> str:
    lines = ["# a16z Research-Coverage Signal — Findings", "",
             "_Generated by `python -m study.run`._", "", CAVEAT, ""]
    win_hdr = " | ".join(f"fwd {w}m IC" for w in FWD_WINDOWS)
    for mode, r in results.items():
        lines += [f"## Attribution mode: `{mode}`", ""]
        a = r["study_a"]
        lines += [f"### Study A — basket level — **verdict: {a['verdict']}**", "",
                  f"| signal | {win_hdr} |", "|" + "---|" * (len(FWD_WINDOWS) + 1),
                  _ic_row("coverage_momentum", a["ic"]), ""]
        bt = " | ".join(f"{a['backtest'][w]['mean_spread']:+.4f}" for w in FWD_WINDOWS)
        lines += [f"Toy long/short mean spread: {bt}", ""]
        for agg, b in r["study_b"].items():
            lines += [f"### Study B — token level (`{agg}`) — **verdict: {b['verdict']}**", "",
                      f"| signal | {win_hdr} |", "|" + "---|" * (len(FWD_WINDOWS) + 1),
                      _ic_row(f"conviction ({agg})", b["ic"]), ""]
            qs = " | ".join(f"{b['quartile_spread'][w]:+.4f}" for w in FWD_WINDOWS)
            lines += [f"Quartile long/short spread: {qs}", ""]
    lines += ["## Coverage heatmap", "", f"![coverage]({HEATMAP_PNG.name})", ""]
    return "\n".join(lines)


def render_heatmap(coverage: pd.DataFrame) -> None:
    """coverage: long-form [month, basket, coverage_share]. Saves HEATMAP_PNG."""
    wide = coverage.pivot(index="basket", columns="month", values="coverage_share").fillna(0.0)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, wide.shape[1] * 0.18), 3 + wide.shape[0] * 0.4))
    im = ax.imshow(wide.values, aspect="auto", cmap="magma")
    ax.set_yticks(range(wide.shape[0]))
    ax.set_yticklabels(wide.index)
    step = max(1, wide.shape[1] // 24)
    ax.set_xticks(range(0, wide.shape[1], step))
    ax.set_xticklabels([str(m) for m in wide.columns[::step]], rotation=90, fontsize=7)
    ax.set_title("a16z research coverage share by basket over time")
    fig.colorbar(im, ax=ax, label="coverage share")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/test_findings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add study/findings.py tests/test_findings.py
git commit -m "feat(study): findings markdown renderer + coverage heatmap"
```

---

## Task 8: Orchestrator (`run.py`) + STATUS update

**Files:**
- Create: `study/run.py`
- Modify: `docs/STATUS.md`

- [ ] **Step 1: Write `study/run.py`**

```python
"""Orchestrate the viability study: both attribution modes x Study A + Study B.

Run:  python -m study.run            (uses cached prices if present)
      python -m study.run --refresh  (re-fetch all prices from Coinglass)

Writes findings.md (committed) + data/study/coverage_heatmap.png + intermediate panels.
"""
import argparse
import sys

import yaml

from . import coverage as coverage_mod
from . import returns as returns_mod
from . import signal as signal_mod
from . import study_basket, study_token, findings
from .config import (ATTRIBUTION_MODES, BASKETS_YAML, CONVICTION_AGGS, FINDINGS_MD,
                     STUDY_DIR)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_baskets() -> dict:
    with open(BASKETS_YAML) as f:
        return yaml.safe_load(f)


def all_tokens(baskets_cfg: dict) -> list[str]:
    toks = set()
    for tokens in baskets_cfg["baskets"].values():
        toks.update(tokens)
    return sorted(toks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch prices (ignore cache)")
    args = ap.parse_args()
    use_cache = not args.refresh

    cfg = load_baskets()
    benchmark = cfg["benchmark"]
    corpus = coverage_mod.load_corpus()

    _log("Fetching token returns...")
    token_rets = returns_mod.monthly_token_returns(all_tokens(cfg), use_cache=use_cache)
    basket_rets = returns_mod.basket_returns(token_rets, cfg)

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    heatmap_coverage = None
    for mode in ATTRIBUTION_MODES:
        _log(f"Running attribution mode: {mode}")
        cov = coverage_mod.monthly_coverage(corpus, cfg, mode=mode)
        if heatmap_coverage is None:
            heatmap_coverage = cov  # heatmap uses fractional (first) mode
        panel_a = signal_mod.basket_signal_panel(cov, basket_rets, benchmark)
        res_a = study_basket.run_study_a(panel_a)
        res_b = {agg: study_token.run_study_b(cov, token_rets, cfg, agg)
                 for agg in CONVICTION_AGGS}
        results[mode] = {"study_a": res_a, "study_b": res_b}
        panel_a.to_parquet(STUDY_DIR / f"panel_a_{mode}.parquet")

    findings.render_heatmap(heatmap_coverage)
    md = findings.render_markdown(results)
    FINDINGS_MD.write_text(md)
    _log(f"Wrote {FINDINGS_MD}")
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full study end-to-end (live, requires Coinglass key)**

Run: `cd projects/sentience && .venv/bin/python -m study.run --refresh`
Expected: writes `findings.md`, `data/study/coverage_heatmap.png`, panels; prints the markdown. Verify the verdict lines render and the caveat is present.

- [ ] **Step 3: Run the whole test suite**

Run: `cd projects/sentience && .venv/bin/python -m pytest tests/ -v`
Expected: all tests PASS (integration test PASS or SKIP).

- [ ] **Step 4: Update `docs/STATUS.md`**

Replace the "What this project is" uncertainty and the "Open threads" item 1 with the now-defined direction. Set the top line to `_Last updated: 2026-06-05_`, and under a new "## Research question (defined 2026-06-05)" heading, state: the hypothesis (coverage rotation leads relative basket/token returns), the two studies, the smoke-test framing with the n≈40-50 caveat, and point to the spec, this plan, and `findings.md`. Move corpus details under a "## Corpus" heading and keep the path note.

- [ ] **Step 5: Commit (including findings.md — the deliverable)**

```bash
git add study/run.py findings.md docs/STATUS.md
git commit -m "feat(study): orchestrator, generated findings, STATUS update"
```

---

## Self-Review

**Spec coverage:**
- §2 baskets + benchmark + editable membership → Task 0 `baskets.yaml`. ✓
- §3.1 Coinglass client, key from op, cache, since-inception → Task 1. ✓
- §3.2 coverage, tag→basket, fractional+full modes, momentum, uniform format weight → Task 2. ✓
- §3.3 basket returns, ≥1-live eligibility, n_live recorded → Task 3. ✓
- §3.4 cross-sectional signal, forward relative returns, no lookahead, benchmark excluded → Task 4. ✓
- §3.5 Study A IC + toy backtest + verdict → Task 5. ✓
- §3.6 Study B token conviction (sum/mean), IC + quartile backtest → Task 6. ✓
- §7 findings.md + heatmap, both studies × both modes, n + caveat → Tasks 7, 8. ✓
- §5 error handling: missing key fails fast (Task 1 `api_key`/`raise_for_status`), no-perp skipped (Task 1/3), 0-live omitted (Task 3), non-zero code raises (Task 1). ✓
- §6 testing: every module has unit tests; coinglass has the gated integration test. ✓
- §8 corpus content note: informational, no code; reflected by uniform format weight (Task 2). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; STATUS edit (Task 8 Step 4) is prose-editing a doc, not code, so its description is sufficient.

**Type consistency:** `month` is `Period[M]` everywhere; `information_coefficient` defined in Task 5 and reused in Task 6 with the same signature; `monthly_token_returns`/`basket_returns`/`monthly_coverage`/`basket_signal_panel`/`token_conviction` signatures match the data-contracts block and all call sites in `run.py`. `FWD_WINDOWS` used consistently as the dict keys (ints 1, 3) across signal/study/findings.

**Note for executor:** Tasks 1 and 8 Step 2 hit the live Coinglass API and need the `op` CLI signed in. All other tasks are fully offline (synthetic fixtures). The integration test self-skips without `op`.
