"""TDD tests for doppelganger.soul_audit."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.soul_audit import parse_citations, Citation


def test_parse_citations_extracts_date_and_quote():
    card = (
        "## Bio Lens\n"
        'Quant lens. [2022-06-01] "Tokens align incentives." and more.\n'
        '## How He Thinks\n'
        'Mechanism-first. [2022-07-01] "rollups inherit security"\n'
        "No citation on this sentence.\n"
    )
    cites = parse_citations(card)
    assert cites == [
        Citation(date(2022, 6, 1), "Tokens align incentives."),
        Citation(date(2022, 7, 1), "rollups inherit security"),
    ]
