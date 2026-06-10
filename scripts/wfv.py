"""Walk-forward validation (Jax-required) — no full-sample param picking.

Anchored WFV: precompute each month's realized return for every param combo (K × momentum-gate), then walk
forward — at each test month t, pick the params that maximized the chosen objective on the TRAIN window
[0,t) only, apply to month t, accumulate OOS-only returns. Reports OOS performance vs the (cheating)
in-sample-optimal, so we can see how much of the edge survives honest param selection.

Strategy form fixed (long top-K A2b-bullish ∩ momentum≥gate, BTC fallback); WFV only chooses K and gate."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
import scripts.alpha_hunt3 as ah
from signals.backtest import realize_period, metrics
from signals.informativeness import forward_return

A2B = ah.load("data/signal/a2b_council/signal_panel.parquet")
KS = [2, 3, 4, 5]
GATES = [0.0, 0.05, 0.10, 0.15]
COMBOS = [(k, g) for k in KS for g in GATES]


def month_ret(t, tnext, k, gate):
    secs = ah.topk(A2B, t, k, mom_thr=gate)
    w = ah.ew(t, secs) if secs else {"BTC": 1.0}
    return realize_period(w, ah.prices, ah.funding, t, tnext, cost_bps=10)


def main():
    dates = [d for d in ah.MD if d in ah.NXT]
    # precompute return matrix: month × combo
    R = np.full((len(dates), len(COMBOS)), np.nan)
    for i, t in enumerate(dates):
        for j, (k, g) in enumerate(COMBOS):
            R[i, j] = month_ret(t, ah.NXT[t], k, g)
    btc = np.array([forward_return(ah.prices, "BTC", t, ah.NXT[t]) or 0.0 for t in dates])

    def sharpe(x):
        x = x[~np.isnan(x)]
        return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(12)) if len(x) > 1 and np.std(x) > 0 else -9

    MIN_TRAIN = 12
    for objective in ["sharpe", "total"]:
        oos = []
        picks = []
        for i in range(MIN_TRAIN, len(dates)):
            train = R[:i]  # [0,i) — no peek at month i
            if objective == "sharpe":
                scores = [sharpe(train[:, j]) for j in range(len(COMBOS))]
            else:
                scores = [np.nansum(np.log1p(train[:, j])) for j in range(len(COMBOS))]
            best = int(np.argmax(scores))
            oos.append(R[i, best])
            picks.append(COMBOS[best])
        oos = np.array(oos)
        oos_btc = btc[MIN_TRAIN:]
        tot = float(np.prod(1 + oos) - 1)
        tot_btc = float(np.prod(1 + oos_btc) - 1)
        # jensen alpha t on OOS
        X = np.column_stack([np.ones(len(oos)), oos_btc])
        coef, *_ = np.linalg.lstsq(X, oos, rcond=None)
        resid = oos - X @ coef
        s2 = (resid @ resid) / (len(oos) - 2)
        ta = coef[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
        from collections import Counter
        pc = Counter(picks)
        print(f"=== WFV objective={objective} (min-train {MIN_TRAIN}mo, {len(oos)} OOS months) ===")
        print(f"  OOS total {tot:+.0%} (BTC same window {tot_btc:+.0%}) | OOS Sharpe {sharpe(oos):.2f} "
              f"(BTC {sharpe(oos_btc):.2f}) | OOS alpha {coef[0]*12:+.0%}/yr t={ta:.2f} beta={coef[1]:.2f}")
        print(f"  params chosen (most common): {pc.most_common(3)}")
    # in-sample-optimal (cheating) for contrast
    full = [sharpe(R[:, j]) for j in range(len(COMBOS))]
    bj = int(np.argmax(full))
    print(f"\n  in-sample-optimal (CHEATS): {COMBOS[bj]} Sharpe {sharpe(R[:, bj]):.2f} "
          f"total {np.prod(1+R[~np.isnan(R[:,bj]),bj])-1:+.0%}")


if __name__ == "__main__":
    main()
