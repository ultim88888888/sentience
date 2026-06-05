"""Regression tests for orchestrator helpers (real corpus stores list cols as numpy arrays)."""
import numpy as np
import pandas as pd
from attribution.run import _participants
from attribution.roster import Roster

def _roster():
    return Roster(pd.DataFrame([
        {"slug": "scott-kominers", "name": "Scott Duke Kominers", "title": "x"},
    ]))

def test_participants_handles_numpy_array_author_slugs():
    # the bug: `row.get("author_slugs") or []` raised on a non-empty ndarray.
    row = pd.Series({"author_slugs": np.array(["scott-kominers"], dtype=object)})
    assert _participants(row, _roster()) == ["Scott Duke Kominers"]

def test_participants_handles_empty_array_and_none():
    assert _participants(pd.Series({"author_slugs": np.array([], dtype=object)}), _roster()) == []
    assert _participants(pd.Series({"author_slugs": None}), _roster()) == []
