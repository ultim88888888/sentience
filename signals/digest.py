"""Period digest for A3 — lookahead-safe market context + news, fed to member doppelgangers so they
REASON (with their framework) about the interval rather than regurgitate corpus stance.

Two blocks, both strictly <= eval date T (no lookahead, by construction):
  - market_block: trailing returns (7D/1M/3M/6M/1Y/2Y) per sector basket and major token, as of T, plus
    BTC-relative performance (price-normalized, per the `hm` finding that raw levels are price-contaminated).
    Deterministic, from Coinglass OHLCV. Backward-looking only.
  - news_block: GDELT 2.0 crypto headlines in (T-interval, T], aggregated by Sonnet into a dense factual
    digest. Lookahead-safe because GDELT enddatetime = T (news published before T only).

Sonnet (not Opus) does the aggregation — it is summary work, not reasoning (Jax's call)."""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
from datetime import timedelta
import pandas as pd

from signals.informativeness import _asof_price
from signals.market import sector_basket, OI_FLOOR_USD
from doppelganger.llm import run_claude

HORIZONS = {"7D": 7, "1M": 30, "3M": 91, "6M": 182, "1Y": 365, "2Y": 730}
MAJORS = ["BTC", "ETH", "SOL"]


# ── market block (deterministic, lookahead-safe) ─────────────────────────────
def trailing_return(prices: pd.DataFrame, sym: str, t, days: int) -> float | None:
    """close(<=t) / close(<= t-days) - 1. Both anchors are <= t ⇒ no lookahead."""
    p1 = _asof_price(prices, sym, t)
    p0 = _asof_price(prices, sym, pd.Timestamp(t) - timedelta(days=days))
    return (p1 / p0 - 1.0) if (p0 and p1 and p0 > 0) else None


def _basket_trailing(basket: list[str], prices: pd.DataFrame, t, days: int) -> float | None:
    rs = [r for s in basket if (r := trailing_return(prices, s, t, days)) is not None]
    return sum(rs) / len(rs) if rs else None


def market_block(t, sector_map: dict, prices: pd.DataFrame, oi_panel: pd.DataFrame,
                 *, oi_floor: float = OI_FLOOR_USD) -> str:
    """Sector-basket + major-token trailing-return tables as of T, with BTC-relative columns."""
    from signals.informativeness import oi_at
    oi_now = oi_at(oi_panel, t)
    btc = {h: trailing_return(prices, "BTC", t, d) for h, d in HORIZONS.items()}

    def fmt(v):
        return f"{v*100:+6.0f}%" if v is not None else "    n/a"

    lines = [f"MARKET PERFORMANCE as of {pd.Timestamp(t).date()} (trailing; all data <= this date).",
             "Returns are price change to the eval date. 'vsBTC' = sector/token return minus BTC return.",
             "", "SECTORS (equal-weight liquid basket):",
             f"{'sector':<24}" + "".join(f"{h:>8}" for h in HORIZONS) + f"{'1Y vsBTC':>10}"]
    sectors = sorted(set(sector_map.values()))
    rows = []
    for sec in sectors:
        basket = sector_basket(sector_map, oi_now, sec, oi_floor=oi_floor)
        if not basket:
            continue
        rets = {h: _basket_trailing(basket, prices, t, d) for h, d in HORIZONS.items()}
        vsbtc = (rets["1Y"] - btc["1Y"]) if (rets["1Y"] is not None and btc["1Y"] is not None) else None
        rows.append((sec, rets, vsbtc, len(basket)))
    # order by 3M performance (most recent material horizon) so the digest leads with what moved
    rows.sort(key=lambda r: (r[1]["3M"] is None, -(r[1]["3M"] or 0)))
    for sec, rets, vsbtc, n in rows:
        lines.append(f"{sec[:23]:<24}" + "".join(fmt(rets[h]) for h in HORIZONS) + fmt(vsbtc) + f"  (n={n})")
    lines += ["", "MAJOR TOKENS:",
              f"{'token':<24}" + "".join(f"{h:>8}" for h in HORIZONS)]
    for sym in MAJORS:
        rets = {h: trailing_return(prices, sym, t, d) for h, d in HORIZONS.items()}
        if any(v is not None for v in rets.values()):
            lines.append(f"{sym:<24}" + "".join(fmt(rets[h]) for h in HORIZONS))
    return "\n".join(lines)


# ── news block (CryptoCompare data-api + Sonnet, lookahead-safe) ──────────────
# Keyless current host (the old min-api now requires a key). `to_ts` returns articles published
# BEFORE that unix ts ⇒ historical pagination + lookahead safety by construction.
_CC = "https://data-api.cryptocompare.com/news/v1/article/list"


def _cc_page(to_ts: int, page: int) -> list[dict]:
    url = _CC + "?" + urllib.parse.urlencode(
        {"lang": "EN", "limit": page, "to_ts": to_ts, "sortOrder": "latest"})
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", "replace")).get("Data", []) or []
    except Exception:
        return []


def cc_news(t, interval_days: int, *, anchors: int | None = None, page: int = 100) -> list[dict]:
    """Crypto headlines spanning (T-interval_days, T]. Crypto news volume is ~hundreds/day, so a
    naive backward page only covers ~1 day. Instead SAMPLE the window: fetch one page anchored at
    several evenly-spaced to_ts points across the interval → temporal coverage of the period's arc.
    Hard lookahead guard: drop any article with PUBLISHED_ON >= T. Returns [{date,title,source}]."""
    end_ts = int(pd.Timestamp(t).timestamp())
    start_ts = int((pd.Timestamp(t) - timedelta(days=interval_days)).timestamp())
    n = anchors if anchors is not None else max(2, min(8, round(interval_days / 14)))
    # anchor at T, then evenly back to the window start
    step = (end_ts - start_ts) / n
    anchor_ts = [int(end_ts - i * step) for i in range(n)]
    seen, out = set(), []
    for ats in anchor_ts:
        for a in _cc_page(ats, page):
            ts, gid = a.get("PUBLISHED_ON"), a.get("ID")
            if not ts or gid in seen or ts >= end_ts or ts < start_ts:
                continue
            seen.add(gid)
            src = (a.get("SOURCE_DATA") or {}).get("NAME") or a.get("SOURCE_ID", "")
            out.append({"ts": ts, "date": pd.Timestamp(ts, unit="s").date().isoformat(),
                        "title": a.get("TITLE", ""), "source": src})
    out.sort(key=lambda x: -x["ts"])
    return out


_NEWS_SYS = """You aggregate raw crypto news headlines into a dense, factual digest of a single period.
It is {t}. The future has not happened — use ONLY these headlines (all published before {t}); never add
events, outcomes, or prices from your own knowledge, and never reference anything after {t}.
Produce 8-15 bullet points capturing the period's MATERIAL events and themes — regulation/enforcement,
hacks/exploits, major launches/upgrades, macro, institutional flows, notable token-specific news — grouped
loosely by theme. Each bullet: one factual line. No speculation, no forward-looking statements, no advice.
Output plain bullets ('- ...'), nothing else."""


def news_block(t, interval_days: int, *, model: str = "sonnet", anchors: int | None = None) -> str:
    arts = cc_news(t, interval_days, anchors=anchors)
    if not arts:
        return "(no news retrieved for this period)"
    head = "\n".join(f"- {a['date']} | {a['title']} ({a['source']})" for a in arts)
    sys = _NEWS_SYS.format(t=pd.Timestamp(t).date())
    try:
        digest = run_claude(sys, head, model=model, effort="low", timeout=300)
        return f"{digest}\n[aggregated from {len(arts)} headlines]"
    except Exception as e:
        return f"(news aggregation failed: {e})\nRAW HEADLINES:\n{head[:2000]}"


def build_digest(t, interval_days: int, sector_map: dict, prices: pd.DataFrame,
                 oi_panel: pd.DataFrame, *, news: bool = True, model: str = "sonnet") -> dict:
    """One period digest: {as_of, market, news}. Both blocks strictly <= t."""
    out = {"as_of": pd.Timestamp(t).date().isoformat(),
           "market": market_block(t, sector_map, prices, oi_panel)}
    out["news"] = news_block(t, interval_days, model=model) if news else ""
    return out


def digest_text(d: dict) -> str:
    """Render a digest dict as the text block fed to a member."""
    return f"=== PERIOD DIGEST (as of {d['as_of']}) ===\n\n{d['market']}\n\nKEY EVENTS / NEWS:\n{d['news']}"
