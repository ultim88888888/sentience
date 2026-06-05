import pandas as pd
from study import findings


def test_render_findings_includes_verdict_and_n():
    results = {
        "fractional": {
            "study_a": {"ic": {1: {"ic_mean": 0.12, "ic_std": 0.3, "hit_rate": 0.6, "n": 42},
                               3: {"ic_mean": 0.08, "ic_std": 0.3, "hit_rate": 0.55, "n": 40}},
                        "backtest": {1: {"mean_spread": 0.01, "n_months": 42},
                                     3: {"mean_spread": 0.02, "n_months": 40}},
                        "verdict": "pulse"},
            "study_b": {"sum": {"agg": "sum",
                                "ic": {1: {"ic_mean": 0.05, "ic_std": 0.2, "hit_rate": 0.5, "n": 50},
                                       3: {"ic_mean": 0.04, "ic_std": 0.2, "hit_rate": 0.5, "n": 48}},
                                "quartile_spread": {1: 0.01, 3: 0.02}, "verdict": "inconclusive"}},
        }
    }
    md = findings.render_markdown(results)
    assert "pulse" in md
    assert "n=42" in md
    assert "Sample size" in md  # the caveat must be present
