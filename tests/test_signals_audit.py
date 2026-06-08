from datetime import date
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal
from signals.audit import audit_period

def _period(citations):
    item = SignalItem(item="zk", item_type="sector", parent_sector=None, stance="bullish",
                      conviction=80, horizon="structural", rationale="r",
                      provenance="grounded", age_note=None, citations=tuple(citations))
    return PeriodSignal(as_of="2024-06-30", approach="A1", items=(item,),
                        risk_regime=RiskRegime("risk_on", 60, "w", "grounded"))

CORPUS = "[2024-02-01] (transcript) zk is the endgame for scaling and i am very bullish"

def test_grounded_citation_passes():
    rep = audit_period(_period([Citation("2024-02-01", "zk is the endgame for scaling")]),
                       CORPUS, t=date(2024, 6, 30))
    assert rep.ok
    assert rep.matched == 1 and not rep.hallucinated

def test_hallucinated_quote_flagged():
    rep = audit_period(_period([Citation("2024-02-01", "solana will flip ethereum by 2025")]),
                       CORPUS, t=date(2024, 6, 30))
    assert not rep.ok
    assert len(rep.hallucinated) == 1

def test_post_t_quote_flagged_as_leaked():
    rep = audit_period(_period([Citation("2025-01-01", "zk is the endgame for scaling")]),
                       CORPUS, t=date(2024, 6, 30))
    assert not rep.ok
    assert len(rep.leaked) == 1


def test_smart_punctuation_quote_matches_after_folding():
    # corpus has an em-dash and a curly apostrophe; quote uses plain hyphen/apostrophe
    corpus = "[2024-02-01] (x) zk is the long—term endgame and we’re very bullish"
    rep = audit_period(_period([Citation("2024-02-01", "the long-term endgame and we're very bullish")]),
                       corpus, t=date(2024, 6, 30))
    assert rep.matched == 1 and rep.ok
