"""Monthly returns: per token (from cached prices) and per basket (equal-weight, n_live)."""
import sys
import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
