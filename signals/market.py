"""1B market-data backbone: liquid universe, LLM sector-classification, BTC beta, sector baskets.
OI floor (TWAP-feasibility derived) gates universe/basket inclusion."""
from __future__ import annotations

import json

import pandas as pd

from doppelganger.llm import run_claude
from signals.extract import _extract_json

# OI floor: floor >= max_position$ / (0.05 * max_TWAP_days). $500k pos / (0.05*5d) = $2M.
OI_FLOOR_USD = 2_000_000
MIN_VOLUME_USD = 1_000_000

NON_CRYPTO = {"GOLD", "SILVER", "SP500", "SPX500", "BRENTOIL", "USOIL", "XYZ100", "NDX", "DXY"}
STABLECOINS = {"USDT", "USDC", "DAI", "TUSD", "FDUSD", "USDE", "PYUSD"}
EXCLUDE = NON_CRYPTO | STABLECOINS


def build_universe(coins_markets: list[dict], *, oi_floor: float = OI_FLOOR_USD) -> list[str]:
    """Candidate liquid universe: symbols whose snapshot open_interest_usd clears the floor,
    excluding macro/index/fiat tickers and stablecoins. Output is sorted alphabetically."""
    return sorted(
        c["symbol"] for c in coins_markets
        if isinstance(c, dict)
        and (c.get("open_interest_usd") or 0) >= oi_floor
        and c.get("symbol") not in EXCLUDE
    )


_SECTOR_SYS = """You classify crypto tokens into sectors. You are given a list of token TICKERS and the
controlled SECTOR vocabulary. Assign each ticker to the ONE sector it best belongs to, or "other" if
none fit. Be accurate (ETH->pos-l1 or l2-scaling per its role; ARB/OP/STRK->l2-scaling; UNI/AAVE->defi;
RENDER/AKT->depin; etc.). Output JSON only: {"map":[{"ticker":"ARB","sector":"l2-scaling"}, ...]}.
Every input ticker must appear exactly once."""


def classify_sectors(tickers: list[str], sectors: list[str], *, batch: int = 120) -> dict:
    """LLM → {ticker: sector}. Tickers the LLM omits default to 'other' (never dropped)."""
    out = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        payload = json.dumps({"tickers": chunk, "sectors": sectors}, indent=2)
        for m in _extract_json(run_claude(_SECTOR_SYS, payload, effort="low")).get("map", []):
            if isinstance(m, dict) and m.get("ticker"):
                out[m["ticker"]] = m.get("sector") or "other"
    for tk in tickers:
        out.setdefault(tk, "other")
    return out


def compute_beta(returns: pd.DataFrame, *, window: int = 90, market: str = "BTC") -> pd.DataFrame:
    """Rolling point-in-time beta of each column vs `market` returns. returns: date-indexed daily
    returns, one column per ticker (must include `market`). beta_t uses only the trailing `window`."""
    mkt = returns[market]
    var = mkt.rolling(window).var()
    betas = {col: returns[col].rolling(window).cov(mkt) / var for col in returns.columns}
    return pd.DataFrame(betas, index=returns.index)


def sector_basket(sector_map: dict, oi_at_t: dict, sector: str, *,
                  oi_floor: float = OI_FLOOR_USD) -> list[str]:
    """Equal-weight basket = tickers mapped to `sector` whose OI at time t clears the floor."""
    return sorted(tk for tk, s in sector_map.items()
                  if s == sector and (oi_at_t.get(tk) or 0) >= oi_floor)
