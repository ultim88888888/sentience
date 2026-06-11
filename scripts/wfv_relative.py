"""Walk-forward validation of the RELATIVE-momentum gate + rigor checks (Jax: validate, likely promote).
WFV on in-sample (params on train only); threshold robustness; and the full 2021->2026 spliced cycle
(OOS 2021-22 bear + in-sample 2023-26) as one continuous backtest — the real test through a full cycle."""
from __future__ import annotations
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from datetime import date
import scripts.alpha_hunt3 as ah, scripts.new_ideas as ni, scripts.relative_momentum as rm
from signals.market import sector_basket, OI_FLOOR_USD
from signals.backtest import realize_period, beta_neutralize, metrics
from signals.informativeness import oi_at, forward_return
from signals.run import rebalance_dates

IS = ni.A2B
OOS = rm.OOS_PANEL


def month_ret(panel, t, tnext, k, thr):
    oin = oi_at(ah.oi, t); cand = []
    for sec, conv in rm.sconv(panel, t).items():
        b = sector_basket(ah.sm, oin, sec, oi_floor=OI_FLOOR_USD)
        if not b:
            continue
        m1 = rm.rel_mom(t, sec, 30); m3 = rm.rel_mom(t, sec, 91)
        if m1 is not None and m1 >= thr and m3 is not None and m3 >= thr:
            cand.append((conv, sec, b))
    cand.sort(key=lambda x: -x[0]); cand = cand[:k]
    w = {"BTC": 1.0} if not cand else None
    if cand:
        raw = {}
        for conv, sec, b in cand:
            for s in b:
                raw[s] = raw.get(s, 0.0) + 1.0 / len(b)
        g = sum(abs(x) for x in raw.values()); w = {s: x / g for s, x in raw.items()}
    if ni.trailing_btc(t) < -0.10:
        w = beta_neutralize(w, ah.BETAS)
    return realize_period(w, ah.prices, ah.funding, t, tnext, cost_bps=10)


def sh(x):
    x = x[~np.isnan(x)]
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(12)) if len(x) > 1 and np.std(x) > 0 else -9


def jt(ret, btc):
    X = np.column_stack([np.ones(len(ret)), btc]); c, *_ = np.linalg.lstsq(X, ret, rcond=None)
    res = ret - X @ c; s2 = (res @ res) / (len(ret) - 2)
    return c[0] * 12, c[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])


def main():
    ISD = [pd.Timestamp(d) for d in rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "monthly")]
    ISD = [d for d in ISD]
    NXT = {d: ISD[i + 1] for i, d in enumerate(ISD[:-1])}
    dates = [d for d in ISD if d in NXT]
    COMBOS = [(k, thr) for k in [2, 3] for thr in [0.0, 0.05, 0.10]]
    R = np.array([[month_ret(IS, t, NXT[t], k, thr) for (k, thr) in COMBOS] for t in dates])
    btc = np.array([forward_return(ah.prices, "BTC", t, NXT[t]) or 0.0 for t in dates])

    print("=== WFV relative-gate (params on train only, min-train 12) ===")
    for obj in ["sharpe", "total"]:
        oos = []
        for i in range(12, len(dates)):
            tr = R[:i]; sc = [sh(tr[:, j]) if obj == "sharpe" else np.nansum(np.log1p(tr[:, j])) for j in range(len(COMBOS))]
            oos.append(R[i, int(np.argmax(sc))])
        oos = np.array(oos); ob = btc[12:]; a, ta = jt(oos, ob)
        print(f"  obj={obj}: OOS tot {np.prod(1+oos)-1:+.0%} (BTC {np.prod(1+ob)-1:+.0%}) Sharpe {sh(oos):.2f} alpha {a:+.0%} t={ta:.2f}")

    print("\n=== threshold robustness (full in-sample) ===")
    for thr in [0.0, 0.05, 0.10, 0.15]:
        r = np.array([month_ret(IS, t, NXT[t], 3, thr) for t in dates])
        a, ta = jt(r, btc)
        print(f"  rel-thr {thr:.0%}: tot {np.prod(1+r)-1:+.0%} Sharpe {sh(r):.2f} alpha {a:+.0%} t={ta:.2f}")

    print("\n=== FULL CYCLE 2021-09..2026-03 spliced (OOS bear + in-sample), relative gate K=3 thr=0 ===")
    OOSD = [pd.Timestamp(d) for d in rebalance_dates(date(2021, 9, 30), date(2022, 11, 30), "monthly")]
    seg = []
    for panel, dd in [(OOS, OOSD), (IS, ISD)]:
        nx = {d: dd[i + 1] for i, d in enumerate(dd[:-1])}
        for t in dd:
            if t not in nx:
                continue
            seg.append({"m": t.strftime("%Y-%m"), "ret": month_ret(panel, t, nx[t], 3, 0.0),
                        "btc": forward_return(ah.prices, "BTC", t, nx[t]) or 0.0})
    df = pd.DataFrame(seg).drop_duplicates("m").sort_values("m")
    m = metrics(df.set_index("m")["ret"], periods_per_year=12); mb = metrics(df.set_index("m")["btc"], periods_per_year=12)
    a, ta = jt(df["ret"].values, df["btc"].values)
    tot = lambda x: (1 + x).prod() - 1
    print(f"  FULL CYCLE ({len(df)} mo): strat {tot(df.ret):+.0%} (Sharpe {m['sharpe']:.2f} DD {m['max_dd']:.0%}) | "
          f"BTC {tot(df.btc):+.0%} (Sharpe {mb['sharpe']:.2f} DD {mb['max_dd']:.0%}) | alpha {a:+.0%} t={ta:.2f}")


if __name__ == "__main__":
    main()
