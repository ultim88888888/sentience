"""TDD tests for doppelganger.memory."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger.memory import load_memory, MemoryView


def _ev(tmp_path) -> Path:
    p = tmp_path / "evidence.parquet"
    pd.DataFrame([
        {"id": "a", "timestamp": pd.Timestamp("2022-07-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives.", "context": None},
        {"id": "b", "timestamp": pd.Timestamp("2021-01-01", tz="UTC"),
         "source_type": "podcast", "text": "Points are a balance.", "context": "What about points?"},
        {"id": "c", "timestamp": pd.Timestamp("2023-03-01", tz="UTC"),
         "source_type": "x_original", "text": "Future take after t0.", "context": None},
    ]).to_parquet(p)
    return p


def test_firewall_excludes_future_and_sets_max_date(tmp_path):
    mv = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    assert isinstance(mv, MemoryView)
    assert set(mv.items["id"]) == {"a", "b"}          # 2023 item "c" excluded
    assert (mv.items["timestamp"].dt.date <= date(2022, 12, 31)).all()
    assert mv.max_date == date(2022, 7, 1)            # latest <= t0
    assert mv.n_items == 2


def test_items_sorted_chronologically(tmp_path):
    mv = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    assert list(mv.items["id"]) == ["b", "a"]         # 2021 before 2022
    assert mv.text.index("Points are a balance") < mv.text.index("Tokens align incentives")


def test_formatting_includes_date_type_text_context(tmp_path):
    mv = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    assert "[2021-01-01] (podcast)" in mv.text
    assert "(context: What about points?)" in mv.text
    assert "Tokens align incentives." in mv.text
    assert "[2022-07-01] (x_original)" in mv.text


def test_empty_when_t0_before_all(tmp_path):
    mv = load_memory("x", date(2018, 1, 1), evidence_path=_ev(tmp_path))
    assert mv.n_items == 0 and mv.max_date is None and mv.text == ""


def test_query_is_ignored_seam(tmp_path):
    full = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path))
    queried = load_memory("x", date(2022, 12, 31), evidence_path=_ev(tmp_path), query="rollups")
    assert list(queried.items["id"]) == list(full.items["id"])   # query ignored in v1
    assert queried.text == full.text


def test_run_cli_has_memory_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["memory", "--subject", "eddy-lazzarin", "--t0", "2022-12-31"])
    assert ns.cmd == "memory" and ns.subject == "eddy-lazzarin" and ns.t0 == "2022-12-31"
