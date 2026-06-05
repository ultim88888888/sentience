import numpy as np
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


def test_all_unmapped_posts_return_zeros(baskets_cfg):
    corpus = pd.DataFrame({
        "post_date": ["2024-01-10", "2024-01-20"],
        "tags": [np.array(["unmapped"], dtype=object), np.array(["also_unmapped"], dtype=object)],
    })
    cov = coverage.monthly_coverage(corpus, baskets_cfg, mode="fractional")
    assert (cov["coverage_share"] == 0.0).all()
    assert cov["basket"].nunique() == len(baskets_cfg["tag_map"])


def test_benchmark_basket_never_appears(synthetic_corpus, baskets_cfg):
    cov = coverage.monthly_coverage(synthetic_corpus, baskets_cfg, mode="fractional")
    assert "L1 Majors" not in cov["basket"].values
