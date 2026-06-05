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
        rel_long = rel.stack(future_stack=True).rename(f"fwd_rel_{w}m").reset_index()
        out = out.merge(rel_long, on=["month", "basket"], how="left")

    keep = ["month", "basket", "coverage_momentum"] + [f"fwd_rel_{w}m" for w in FWD_WINDOWS]
    return out[keep]
