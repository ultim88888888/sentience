"""TDD tests for doppelganger.judge."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger.judge import post_t_evidence


def _ev(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([
        {"id": "a", "timestamp": pd.Timestamp("2022-11-01", tz="UTC"), "source_type": "x_original",
         "text": "before T", "context": None},
        {"id": "b", "timestamp": pd.Timestamp("2023-02-01", tz="UTC"), "source_type": "x_original",
         "text": "inside window", "context": None},
        {"id": "c", "timestamp": pd.Timestamp("2023-09-01", tz="UTC"), "source_type": "x_original",
         "text": "after window", "context": None},
    ]).to_parquet(p)
    return p


def test_post_t_evidence_slices_window(tmp_path):
    out = post_t_evidence("s", date(2022, 12, 31), horizon_months=6, evidence_path=_ev(tmp_path))
    assert "inside window" in out          # 2023-02-01 in (2022-12-31, 2023-06-30]
    assert "before T" not in out           # 2022-11-01 excluded
    assert "after window" not in out       # 2023-09-01 excluded
    assert "[2023-02-01]" in out


def test_post_t_evidence_empty(tmp_path):
    out = post_t_evidence("s", date(2024, 1, 1), horizon_months=6, evidence_path=_ev(tmp_path))
    assert out == ""


import json
from unittest.mock import patch, MagicMock
from doppelganger.judge import judge_step

_VIEW = {"sectors_excited": [{"name": "ZK", "why": "w", "provenance": "grounded"}],
         "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
         "risk_regime": {"stance": "risk_on"}}

_VERDICT = ('{"claims":[{"claim":"excited about ZK","axis":"sectors_excited","label":"confirmed"},'
            '{"claim":"risk on","axis":"risk_regime","label":"contradicted"}],'
            '"n_confirmed":1,"n_contradicted":1,"n_absent":0,'
            '"missed_changes":["picked up restaking"],"notes":"ok"}')


def test_judge_step_parses_and_computes_confirm_rate(tmp_path):
    with patch("doppelganger.judge.run_claude", return_value=f"```json\n{_VERDICT}\n```"):
        v = judge_step(_VIEW, "he kept tweeting about ZK", "Eddy", date(2022, 12, 31),
                       judge_path=tmp_path / "j.json")
    assert v["n_confirmed"] == 1 and v["n_contradicted"] == 1
    assert v["confirm_rate"] == 0.5                       # 1 / (1+1)
    assert v["missed_changes"] == ["picked up restaking"]
    assert (tmp_path / "j.json").exists()                 # cached


def test_judge_step_confirm_rate_none_when_no_scored(tmp_path):
    verdict = '{"claims":[],"n_confirmed":0,"n_contradicted":0,"n_absent":3,"missed_changes":[]}'
    with patch("doppelganger.judge.run_claude", return_value=verdict):
        v = judge_step(_VIEW, "unrelated", "Eddy", date(2022, 12, 31), judge_path=tmp_path / "j.json")
    assert v["confirm_rate"] is None                      # no confirmed+contradicted -> undefined


def test_judge_step_uses_cache(tmp_path):
    jp = tmp_path / "j.json"
    jp.write_text(json.dumps({"n_confirmed": 9, "n_contradicted": 0, "confirm_rate": 1.0,
                              "missed_changes": [], "claims": []}))
    m = MagicMock()
    with patch("doppelganger.judge.run_claude", m):
        v = judge_step(_VIEW, "x", "Eddy", date(2022, 12, 31), judge_path=jp)
    m.assert_not_called()                                 # cached -> no claude -p
    assert v["n_confirmed"] == 9
