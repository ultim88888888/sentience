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
