"""TDD tests for doppelganger.walkforward."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.walkforward import quarter_ends


def test_quarter_ends_inclusive_span():
    out = quarter_ends(date(2022, 12, 31), date(2023, 9, 30))
    assert out == [date(2022, 12, 31), date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30)]


def test_quarter_ends_skips_partial_quarters():
    out = quarter_ends(date(2023, 1, 15), date(2023, 7, 1))
    assert out == [date(2023, 3, 31), date(2023, 6, 30)]   # Dec-31-2022 excluded, Sep-30 excluded
