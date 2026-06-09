import json
from datetime import date
from unittest.mock import patch
import pandas as pd
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal
from signals.consensus import compute_dispersion, build_consensus, load_member_views, run_a2a_consensus


def _pv(slug_items, risk="risk_on"):
    items = tuple(SignalItem(item=i, item_type="sector", parent_sector=None, stance=s, conviction=c,
                             horizon="structural", rationale="r", provenance="grounded", age_note=None,
                             citations=()) for i, s, c in slug_items)
    return PeriodSignal(as_of="2024-06-30", approach="A2a:x", items=items,
                        risk_regime=RiskRegime(risk, 60, "w", "grounded"))


def test_dispersion_measures_disagreement():
    mv = {"a": _pv([("zk","bullish",80)]), "b": _pv([("zk","bearish",70)]), "c": _pv([("zk","bullish",60)])}
    d = compute_dispersion(mv, n_members=3)
    assert d["zk"]["coverage"] == 3
    assert d["zk"]["stance_dispersion"] > 0          # members disagree on direction
    # all-agree item has zero stance dispersion
    mv2 = {"a": _pv([("defi","bullish",80)]), "b": _pv([("defi","bullish",75)])}
    assert compute_dispersion(mv2, 2)["defi"]["stance_dispersion"] == 0.0


def test_singular_coverage_recorded_not_dropped():
    mv = {"a": _pv([("niche","bullish",90)])}
    d = compute_dispersion(mv, n_members=10)
    assert d["niche"]["coverage"] == 1 and d["niche"]["coverage_frac"] == 0.1  # kept, low coverage


def test_build_consensus_parses_llm_output():
    mv = {"a": _pv([("zk","bullish",80)])}
    out = json.dumps({"sectors_excited":[{"name":"zk","why":"house likes it","conviction":78,
        "horizon":"structural","provenance":"grounded","age_note":None,"citations":[]}],
        "risk_regime":{"stance":"risk_on","conviction":60,"why":"w","provenance":"grounded"}})
    with patch("signals.consensus.run_claude", return_value=out):
        p = build_consensus(date(2024,6,30), mv, compute_dispersion(mv,1))
    assert p.items[0].item == "zk" and p.items[0].stance == "bullish"


def test_run_a2a_consensus_end_to_end(tmp_path):
    # seed two member period files at one date
    root = tmp_path/"members"
    for slug, items in [("m1",[("zk","bullish",80)]), ("m2",[("zk","bullish",70),("defi","bearish",60)])]:
        (root/slug/"periods").mkdir(parents=True)
        (root/slug/"periods"/"2024-06-30.json").write_text(json.dumps(_pv(items).to_dict()))
    cons = json.dumps({"sectors_excited":[{"name":"zk","why":"w","conviction":75,"horizon":"structural",
        "provenance":"grounded","age_note":None,"citations":[]}],
        "risk_regime":{"stance":"risk_on","conviction":60,"why":"w","provenance":"grounded"}})
    canon = json.dumps({"mapping":[{"raw":"zk","canonical":"zk","item_type":"sector","parent_sector":None,"is_new":False}]})
    def fake(system, user, **kw):
        return canon if "registry" in user else cons
    out = tmp_path/"a2a"
    with patch("signals.consensus.run_claude", side_effect=fake), \
         patch("signals.canonicalize.run_claude", side_effect=fake):
        df = run_a2a_consensus([date(2024,6,30)], members_root=root, out_dir=out,
                               registry_path=tmp_path/"registry.json")
    assert (out/"signal_panel.parquet").exists()
    assert (out/"dispersion.parquet").exists()
    disp = pd.read_parquet(out/"dispersion.parquet")
    assert "coverage" in disp.columns and "stance_dispersion" in disp.columns
    assert (df["item"]=="zk").any()


def test_dispersion_coverage_frac_uses_n_members():
    """coverage_frac is coverage/n_members, even if n_members > distinct members with a view."""
    mv = {"a": _pv([("eth","bullish",70)]), "b": _pv([("eth","bullish",80)])}
    d = compute_dispersion(mv, n_members=10)
    assert d["eth"]["coverage_frac"] == pytest.approx(0.2)
    assert d["eth"]["coverage"] == 2


def test_dispersion_mean_stance_and_conviction():
    mv = {"a": _pv([("sol","bullish",80)]), "b": _pv([("sol","neutral",40)])}
    d = compute_dispersion(mv, n_members=2)
    # bullish=+1, neutral=0 → mean=0.5
    assert d["sol"]["mean_stance_sign"] == pytest.approx(0.5)
    assert d["sol"]["mean_conviction"] == pytest.approx(60.0)


def test_dispersion_single_member_zero_spread():
    """Single-member item has pstdev=0 (population stdev of one element)."""
    mv = {"only": _pv([("zk","bullish",90)])}
    d = compute_dispersion(mv, n_members=1)
    assert d["zk"]["stance_dispersion"] == 0.0
    assert d["zk"]["conviction_spread"] == 0.0


def test_load_member_views_picks_correct_date(tmp_path):
    root = tmp_path/"members"
    slug = "alice"
    (root/slug/"periods").mkdir(parents=True)
    t1 = date(2024, 3, 31)
    t2 = date(2024, 6, 30)
    pv1 = _pv([("zk","bullish",80)])
    pv2 = _pv([("defi","bearish",60)])
    (root/slug/"periods"/f"{t1.isoformat()}.json").write_text(json.dumps(pv1.to_dict()))
    (root/slug/"periods"/f"{t2.isoformat()}.json").write_text(json.dumps(pv2.to_dict()))
    views = load_member_views(t1, root)
    assert "alice" in views
    assert views["alice"].items[0].item == "zk"
    views2 = load_member_views(t2, root)
    assert views2["alice"].items[0].item == "defi"


def test_run_a2a_consensus_resumable(tmp_path):
    """If a period json already exists, skip LLM and load from disk."""
    root = tmp_path/"members"
    slug = "m1"
    (root/slug/"periods").mkdir(parents=True)
    (root/slug/"periods"/"2024-06-30.json").write_text(json.dumps(_pv([("zk","bullish",80)]).to_dict()))
    # Pre-seed the consensus period json
    out = tmp_path/"a2a"
    (out/"periods").mkdir(parents=True)
    preseeded = _pv([("zk","bullish",75)])
    preseeded = PeriodSignal(as_of="2024-06-30", approach="A2a",
                             items=preseeded.items, risk_regime=preseeded.risk_regime, notes="cached")
    (out/"periods"/"2024-06-30.json").write_text(json.dumps(preseeded.to_dict()))
    call_count = {"n": 0}
    def fake(*a, **kw):
        call_count["n"] += 1
        return "{}"
    with patch("signals.consensus.run_claude", side_effect=fake), \
         patch("signals.canonicalize.run_claude", side_effect=fake):
        run_a2a_consensus([date(2024,6,30)], members_root=root, out_dir=out,
                          registry_path=tmp_path/"registry.json")
    assert call_count["n"] == 0  # pre-existing period → no LLM


def test_run_a2a_consensus_skip_continues_on_error(tmp_path):
    """A date that raises during processing is skipped; the run still completes."""
    root = tmp_path/"members"
    slug = "m1"
    (root/slug/"periods").mkdir(parents=True)
    # date 1: will fail (LLM raises)
    (root/slug/"periods"/"2024-03-31.json").write_text(json.dumps(_pv([("zk","bullish",80)]).to_dict()))
    # date 2: will succeed
    (root/slug/"periods"/"2024-06-30.json").write_text(json.dumps(_pv([("defi","bullish",70)]).to_dict()))
    good_cons = json.dumps({"sectors_excited":[{"name":"defi","why":"w","conviction":70,
        "horizon":"structural","provenance":"grounded","age_note":None,"citations":[]}],
        "risk_regime":{"stance":"risk_on","conviction":60,"why":"w","provenance":"grounded"}})
    good_canon = json.dumps({"mapping":[{"raw":"defi","canonical":"defi","item_type":"sector",
        "parent_sector":None,"is_new":False}]})
    call_count = {"n": 0}
    def fake(system, user, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated LLM failure")
        return good_canon if "registry" in user else good_cons
    out = tmp_path/"a2a"
    with patch("signals.consensus.run_claude", side_effect=fake), \
         patch("signals.canonicalize.run_claude", side_effect=fake):
        df = run_a2a_consensus([date(2024,3,31), date(2024,6,30)], members_root=root,
                               out_dir=out, registry_path=tmp_path/"registry.json")
    # date 2 should have made it through
    assert (df["item"]=="defi").any()


def test_run_a2a_consensus_no_members_for_date(tmp_path):
    """Date with no member files is skipped gracefully (no output row)."""
    root = tmp_path/"members"
    root.mkdir(parents=True)
    out = tmp_path/"a2a"
    with patch("signals.consensus.run_claude") as m:
        df = run_a2a_consensus([date(2024,6,30)], members_root=root, out_dir=out,
                               registry_path=tmp_path/"registry.json")
    m.assert_not_called()
    assert len(df) == 0


import pytest
