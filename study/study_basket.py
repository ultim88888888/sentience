"""Study A (basket level): does coverage-momentum rank lead forward relative-return rank?

Information Coefficient = mean over months of the cross-sectional Spearman correlation
between signal rank and forward-return rank. Plus a toy long/short backtest. Sample size
is small (~40-50 basket-months); we report n alongside every statistic and never claim
significance.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from .config import FWD_WINDOWS


def information_coefficient(panel: pd.DataFrame, signal_col: str, fwd_col: str,
                            group: str = "month") -> dict:
    ics, hits = [], []
    for _, g in panel.dropna(subset=[signal_col, fwd_col]).groupby(group):
        if len(g) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)  # constant month -> NaN, skipped below
            rho, _ = spearmanr(g[signal_col], g[fwd_col])
        if np.isnan(rho):
            continue
        ics.append(rho)
        # hit rate: did the top-ranked-signal basket beat the cross-sectional median fwd?
        top = g.loc[g[signal_col].idxmax()]  # ties broken by label order — acceptable for a smoke test
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
    bt_has_data = any(not np.isnan(bt[w]["mean_spread"]) for w in FWD_WINDOWS)
    # Bypass: if no month had enough baskets for the L/S backtest (need >=2*top_k=4),
    # treat spread as uninformative rather than contradictory — IC alone decides the verdict.
    spread_ok = (not bt_has_data) or any(bt[w]["mean_spread"] > 0 for w in FWD_WINDOWS
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
