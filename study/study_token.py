"""Study B (token level): does a token in multiple HOT baskets outperform?

Token conviction at month t = aggregate of coverage_momentum across every basket the
token belongs to (sum rewards multi-basket membership; mean normalizes it). Then the same
IC + a long-top-quartile / short-bottom-quartile backtest, at token granularity. This
routes around basket-return correlation (overlapping membership). Reuses Study A's IC.
"""
import numpy as np
import pandas as pd

from .config import FWD_WINDOWS
from .study_basket import _verdict, information_coefficient


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
    for month in sorted(coverage["month"].unique()):
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
    return rel.stack(future_stack=True).rename(f"fwd_rel_{window}m").reset_index()


def run_study_b(coverage: pd.DataFrame, token_returns: pd.DataFrame,
                baskets_cfg: dict, agg: str) -> dict:
    conv = token_conviction(coverage, baskets_cfg, agg)
    panel = conv.copy()
    for w in FWD_WINDOWS:
        rel = _forward_token_relative(token_returns, w)
        panel = panel.merge(rel, on=["month", "token"], how="left")

    ic = {w: information_coefficient(panel, "conviction", f"fwd_rel_{w}m")
          for w in FWD_WINDOWS}
    spreads = {w: _quartile_spread(panel, f"fwd_rel_{w}m") for w in FWD_WINDOWS}
    return {"agg": agg, "ic": ic, "quartile_spread": spreads,
            "verdict": _verdict(ic, spreads)}


def _quartile_spread(panel: pd.DataFrame, fwd_col: str) -> float:
    spreads = []
    for _, g in panel.dropna(subset=["conviction", fwd_col]).groupby("month"):
        if len(g) < 4:
            continue
        q = g["conviction"].quantile([0.25, 0.75])
        # On ties (e.g. all-equal conviction) a token can fall in both long and short;
        # spread then computes to 0.0, not NaN. Cannot occur with the real ~25-token universe.
        longs = g[g["conviction"] >= q[0.75]][fwd_col].mean()
        shorts = g[g["conviction"] <= q[0.25]][fwd_col].mean()
        spreads.append(longs - shorts)
    return float(np.mean(spreads)) if spreads else float("nan")
