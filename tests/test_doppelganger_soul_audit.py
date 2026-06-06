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


import pandas as pd
from doppelganger.soul_audit import audit_soul, AuditReport


def _evidence(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([
        {"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives and that is the core thesis."},
        {"id": "2", "timestamp": pd.Timestamp("2023-03-01", tz="UTC"),
         "source_type": "x_original", "text": "ZK rollups are the endgame for scaling."},
    ]).to_parquet(p)
    return p


def test_audit_passes_clean_card(tmp_path):
    ev = _evidence(tmp_path)
    card = tmp_path / "soul.md"
    card.write_text('## X\nClaim. [2022-06-01] "Tokens align incentives"\n')
    rep = audit_soul(card, ev, date(2022, 12, 31))
    assert rep.ok and rep.checked == 1 and rep.matched == 1
    assert not rep.hallucinated and not rep.leaked


def test_audit_flags_hallucinated(tmp_path):
    ev = _evidence(tmp_path)
    card = tmp_path / "soul.md"
    card.write_text('## X\nClaim. [2022-06-01] "this quote does not exist anywhere"\n')
    rep = audit_soul(card, ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.hallucinated) == 1


def test_audit_flags_leaked_future_quote(tmp_path):
    ev = _evidence(tmp_path)
    card = tmp_path / "soul.md"
    # quote is real but from a 2023 item — leakage past a 2022 t0
    card.write_text('## X\nClaim. [2023-03-01] "ZK rollups are the endgame"\n')
    rep = audit_soul(card, ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.leaked) == 1


def test_audit_folds_smart_quotes(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([{"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                   "source_type": "x_original",
                   "text": "tvl isn’t even a particularly useful metric for lending"}]).to_parquet(p)
    card = tmp_path / "soul.md"
    card.write_text('## X\nClaim. [2022-06-01] "TVL isn\'t even a particularly useful metric for lending"\n')
    rep = audit_soul(card, p, date(2022, 12, 31))
    assert rep.ok and rep.matched == 1


def test_audit_fuzzy_matches_near_verbatim_fragment(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([{"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                   "source_type": "x_original",
                   "text": "we believe privacy unlocks a whole new region of the design space for builders"}]).to_parquet(p)
    card = tmp_path / "soul.md"
    # compressed/trailing-period citation of a real fragment -> grounded under fuzzy match
    card.write_text('## X\nClaim. [2022-06-01] "Privacy unlocks a whole new region of the design space."\n')
    rep = audit_soul(card, p, date(2022, 12, 31))
    assert rep.ok and rep.matched == 1
    # a true fabrication still fails
    card.write_text('## X\nClaim. [2022-06-01] "quantum teleportation solves the oracle problem entirely"\n')
    rep2 = audit_soul(card, p, date(2022, 12, 31))
    assert not rep2.ok and len(rep2.hallucinated) == 1


import json as _json
from doppelganger.soul_audit import audit_answer


def _ev_for_answer(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([
        {"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives and that is the core thesis."},
        {"id": "2", "timestamp": pd.Timestamp("2023-03-01", tz="UTC"),
         "source_type": "x_original", "text": "ZK rollups are the endgame for scaling."},
    ]).to_parquet(p)
    return p


def _view(citation_date, quote):
    return {"sectors_excited": [{"name": "DeFi", "why": "x", "provenance": "grounded",
            "citations": [{"date": citation_date, "quote": quote}]}],
            "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
            "risk_regime": {"stance": "no_view"}}


def test_audit_answer_passes_clean(tmp_path):
    ev = _ev_for_answer(tmp_path)
    rep = audit_answer(_view("2022-06-01", "Tokens align incentives"), ev, date(2022, 12, 31))
    assert rep.ok and rep.matched == 1


def test_audit_answer_flags_hallucinated(tmp_path):
    ev = _ev_for_answer(tmp_path)
    rep = audit_answer(_view("2022-06-01", "this was never said by anyone at all"), ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.hallucinated) == 1


def test_audit_answer_flags_leaked(tmp_path):
    ev = _ev_for_answer(tmp_path)
    rep = audit_answer(_view("2023-03-01", "ZK rollups are the endgame"), ev, date(2022, 12, 31))
    assert not rep.ok and len(rep.leaked) == 1


def test_audit_answer_reads_path(tmp_path):
    ev = _ev_for_answer(tmp_path)
    f = tmp_path / "view.json"
    f.write_text(_json.dumps(_view("2022-06-01", "Tokens align incentives")))
    rep = audit_answer(f, ev, date(2022, 12, 31))
    assert rep.ok and rep.matched == 1
