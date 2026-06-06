"""TDD tests for doppelganger.score."""
from __future__ import annotations

from doppelganger.score import discrimination, coverage_trajectory


def test_discrimination_overlap():
    a = {"sectors_excited": [{"name": "ZK"}, {"name": "Games"}], "sectors_concerned": [],
         "tokens_excited": [{"name": "ETH"}], "tokens_concerned": []}
    b = {"sectors_excited": [{"name": "zk"}, {"name": "DAOs"}], "sectors_concerned": [],
         "tokens_excited": [{"name": "SOL"}], "tokens_concerned": []}
    d = discrimination(a, b)
    # sectors: {zk,games} vs {zk,daos} -> intersection {zk}=1, union 3 -> 1/3
    assert round(d["sector_overlap"], 2) == 0.33
    assert d["token_overlap"] == 0.0          # eth vs sol, no overlap
    assert "zk" in d["shared_sectors"]


def test_coverage_trajectory():
    rows = [
        {"date": "2022-12-31", "variant": "full", "grounded": 13, "persisted": 0, "extrapolated": 0},
        {"date": "2022-12-31", "variant": "ablation", "grounded": 5, "persisted": 2, "extrapolated": 2},
        {"date": "2023-03-31", "variant": "full", "grounded": 11, "persisted": 0, "extrapolated": 0},
    ]
    cov = coverage_trajectory(rows)
    assert cov == [{"date": "2022-12-31", "grounded": 13, "persisted": 0, "extrapolated": 0},
                   {"date": "2023-03-31", "grounded": 11, "persisted": 0, "extrapolated": 0}]


import json
from datetime import date
from pathlib import Path
from unittest.mock import patch
from doppelganger.score import score_subject


def _setup(tmp_path):
    slug = "s"
    base = tmp_path / slug
    (base / "views").mkdir(parents=True, exist_ok=True)
    (base / "views_ablation").mkdir(parents=True, exist_ok=True)
    view = {"sectors_excited": [{"name": "ZK", "provenance": "grounded"}], "sectors_concerned": [],
            "tokens_excited": [], "tokens_concerned": [], "risk_regime": {"stance": "risk_on"}}
    for d in ["2022-12-31", "2023-03-31"]:
        (base / "views" / f"{d}.json").write_text(json.dumps(view))
        (base / "views_ablation" / f"{d}.json").write_text(json.dumps(view))
    (base / "walkforward.json").write_text(json.dumps({"subject": slug,
        "dates": ["2022-12-31", "2023-03-31"],
        "rows": [{"date": "2022-12-31", "variant": "full", "grounded": 1, "persisted": 0, "extrapolated": 0}]}))
    import pandas as pd
    pd.DataFrame([{"id": "x", "timestamp": pd.Timestamp("2023-05-01", tz="UTC"),
                   "source_type": "x_original", "text": "still ZK", "context": None}]).to_parquet(base / "evidence.parquet")
    return slug


def test_score_subject_computes_lift(tmp_path):
    slug = _setup(tmp_path)
    def fake_judge(view, post_t, name, t0, *, judge_path=None):
        rate = 0.9 if "views_ablation" not in str(judge_path) else 0.4
        return {"confirm_rate": rate, "n_confirmed": 1, "n_contradicted": 0,
                "missed_changes": ["m"] if rate == 0.9 else [], "claims": []}
    with patch("doppelganger.score.judge_step", side_effect=fake_judge):
        m = score_subject(slug, out_dir=tmp_path, evidence_path=tmp_path / slug / "evidence.parquet")
    assert m["subject"] == slug
    assert round(m["mean_lift"], 2) == 0.5            # 0.9 - 0.4 per step
    assert len(m["steps"]) == 2
    assert (tmp_path / slug / "metrics.json").exists()


from doppelganger.score import write_memo


def test_write_memo_renders(tmp_path):
    metrics = {"subject": "eddy-lazzarin", "horizon_months": 6, "mean_lift": 0.25,
               "steps": [{"date": "2022-12-31", "full_confirm_rate": 0.9, "ablation_confirm_rate": 0.65,
                          "lift": 0.25, "missed_changes": ["picked up restaking"], "n_missed_changes": 1}],
               "coverage": [{"date": "2022-12-31", "grounded": 13, "persisted": 0, "extrapolated": 0}]}
    p = write_memo(metrics, out_dir=tmp_path)
    txt = Path(p).read_text()
    assert p == tmp_path / "eddy-lazzarin" / "findings.md"
    assert "corpus-lift" in txt.lower() and "0.25" in txt          # headline
    assert "2022-12-31" in txt and "0.9" in txt and "0.65" in txt  # per-step table
    assert "picked up restaking" in txt                            # missed change
    assert "persistence" in txt.lower() and "soul-less" in txt.lower()  # caveats stated


def test_run_cli_has_score_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["score", "--subject", "eddy-lazzarin"])
    assert ns.cmd == "score" and ns.subject == "eddy-lazzarin" and ns.horizon_months == 6
    ns2 = r.build_parser().parse_args(["score", "--subject", "x", "--horizon-months", "3"])
    assert ns2.horizon_months == 3
