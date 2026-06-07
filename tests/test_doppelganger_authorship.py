"""Tests for the pure logic in doppelganger.authorship (no LLM calls)."""
from __future__ import annotations

from doppelganger.authorship import blind, _tally, SUBJECTS


def test_blind_strips_identity_keeps_stance():
    view = {
        "subject": "eddy-lazzarin", "as_of": "2024-06-30", "abstained": False,
        "sectors_excited": [{"name": "ZK", "why": "frontier", "conviction": 92,
                             "provenance": "grounded", "age_note": "x",
                             "citations": [{"date": "2024-04-09", "quote": "leaks identity"}]}],
        "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
        "risk_regime": {"stance": "neutral", "why": "w", "conviction": 70, "provenance": "grounded"},
        "notes": "n",
    }
    b = blind(view)
    # identity / date / evidence gone
    assert "subject" not in b and "as_of" not in b
    flat = str(b)
    assert "leaks identity" not in flat and "grounded" not in flat and "2024-04-09" not in flat
    # stance substance kept
    assert b["sectors_excited"][0] == {"name": "ZK", "why": "frontier", "conviction": 92}
    assert b["risk_regime"] == {"stance": "neutral", "why": "w", "conviction": 70}
    assert b["notes"] == "n"


def test_blind_handles_empty_and_missing():
    b = blind({})
    assert b["sectors_excited"] == [] and b["tokens_concerned"] == []
    assert b["risk_regime"] == {"stance": None, "why": None, "conviction": None}


def test_tally_accuracy_and_confusion():
    rows = [
        {"true": "eddy-lazzarin", "pred": "eddy-lazzarin", "conf": 90},
        {"true": "eddy-lazzarin", "pred": "ali-yahya", "conf": 60},
        {"true": "ali-yahya", "pred": "ali-yahya", "conf": 80},
        {"true": "ali-yahya", "pred": "ali-yahya", "conf": 70},
    ]
    t = _tally(rows)
    assert t["correct"] == 3 and t["n"] == 4 and t["accuracy"] == 0.75
    assert t["mean_confidence"] == 75.0
    assert t["per_subject"]["eddy-lazzarin"] == {"correct": 1, "n": 2}
    assert t["per_subject"]["ali-yahya"] == {"correct": 2, "n": 2}


def test_subjects_constant():
    assert SUBJECTS == ["eddy-lazzarin", "ali-yahya"]
