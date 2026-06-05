import pandas as pd
from study import findings
from study.config import HEATMAP_PNG


def _study_b(agg, verdict):
    return {"agg": agg,
            "ic": {1: {"ic_mean": 0.05, "ic_std": 0.2, "hit_rate": 0.5, "n": 50},
                   3: {"ic_mean": 0.04, "ic_std": 0.2, "hit_rate": 0.5, "n": 48}},
            "quartile_spread": {1: 0.01, 3: 0.02}, "verdict": verdict}


def _study_a(verdict):
    return {"ic": {1: {"ic_mean": 0.12, "ic_std": 0.3, "hit_rate": 0.6, "n": 42},
                   3: {"ic_mean": 0.08, "ic_std": 0.3, "hit_rate": 0.55, "n": 40}},
            "backtest": {1: {"mean_spread": 0.01, "n_months": 42},
                         3: {"mean_spread": 0.02, "n_months": 40}},
            "verdict": verdict}


def test_render_findings_covers_modes_aggs_verdicts_and_caveat():
    results = {
        "fractional": {"study_a": _study_a("pulse"),
                       "study_b": {"sum": _study_b("sum", "inconclusive"),
                                   "mean": _study_b("mean", "no pulse")}},
        "full": {"study_a": _study_a("no pulse"),
                 "study_b": {"sum": _study_b("sum", "pulse"),
                             "mean": _study_b("mean", "inconclusive")}},
    }
    md = findings.render_markdown(results)
    # both modes rendered
    assert "fractional" in md and "full" in md
    # both Study B aggregations rendered
    assert "conviction (sum)" in md and "conviction (mean)" in md
    # verdicts surface
    assert "pulse" in md and "no pulse" in md and "inconclusive" in md
    # sample size always visible
    assert "n=42" in md and "Sample size" in md
    # heatmap link present
    assert HEATMAP_PNG.name in md


def test_render_heatmap_writes_png():
    cov = pd.DataFrame([
        {"month": pd.Period("2024-01", "M"), "basket": "ZK/Privacy", "coverage_share": 0.5},
        {"month": pd.Period("2024-01", "M"), "basket": "DeFi", "coverage_share": 0.5},
        {"month": pd.Period("2024-02", "M"), "basket": "ZK/Privacy", "coverage_share": 0.3},
        {"month": pd.Period("2024-02", "M"), "basket": "DeFi", "coverage_share": 0.7},
    ])
    findings.render_heatmap(cov)
    assert HEATMAP_PNG.exists()


def test_render_heatmap_empty_input_is_noop():
    empty = pd.DataFrame(columns=["month", "basket", "coverage_share"])
    findings.render_heatmap(empty)  # must not raise
