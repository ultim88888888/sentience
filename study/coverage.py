"""Turn the a16z corpus into a monthly coverage-momentum signal per basket.

Uses only post_date + tags (all 235 posts qualify regardless of body depth). Every post
weights 1.0 (uniform across formats — see spec section 3.2). Two attribution modes:
  fractional: a post hitting k baskets contributes 1/k to each.
  full:       the post contributes 1.0 to each basket it touches.
"""
import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Period has no tz; drop is intentional
        df["month"] = pd.to_datetime(df["post_date"], utc=True).dt.to_period("M")

    recs = []
    for _, row in df.iterrows():
        for basket, w in _attribute(row["tags"], tag_map, mode).items():
            recs.append({"month": row["month"], "basket": basket, "weight": w})
    weights = pd.DataFrame(recs, columns=["month", "basket", "weight"])

    # full grid of (month, basket) so absent baskets are explicit zeros
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    grid = pd.MultiIndex.from_product([months, baskets], names=["month", "basket"])

    by_mb = weights.groupby(["month", "basket"])["weight"].sum()
    by_month_total = weights.groupby("month")["weight"].sum()

    cov = by_mb.reindex(grid, fill_value=0.0).rename("w").reset_index()
    cov["w"] = cov["w"].astype(float)
    cov["total"] = cov["month"].map(by_month_total).fillna(0.0)
    cov["coverage_share"] = (cov["w"] / cov["total"]).where(cov["total"] > 0, 0.0)

    cov = cov.sort_values(["basket", "month"]).reset_index(drop=True)
    # momentum = share(t) - trailing mean over the prior LOOKBACK months (per basket)
    cov["coverage_momentum"] = cov.groupby("basket")["coverage_share"].transform(
        lambda s: s - s.shift(1).rolling(MOMENTUM_LOOKBACK).mean()
    )
    return cov[["month", "basket", "coverage_share", "coverage_momentum"]]
