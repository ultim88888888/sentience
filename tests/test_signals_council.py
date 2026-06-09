import json
from datetime import date
from unittest.mock import patch
import pandas as pd
from signals.schema import SignalItem, RiskRegime, PeriodSignal
from signals.council import deliberate, run_a2b, _hedge_from

def _pv(items):
    its=tuple(SignalItem(item=i,item_type="sector",parent_sector=None,stance=s,conviction=c,horizon="structural",
              rationale="r",provenance="grounded",age_note=None,citations=()) for i,s,c in items)
    return PeriodSignal(as_of="2024-06-30",approach="A2a:x",items=its,risk_regime=RiskRegime("risk_on",60,"w","grounded"))

COUNCIL_OUT=json.dumps({"sectors_excited":[{"item":"zk","stance":"bullish","conviction":82,"horizon":"structural","provenance":"grounded","citations":[]}],
  "risk_regime":{"stance":"neutral","conviction":55,"why":"w","provenance":"grounded"},
  "hedge_decision":{"stance":"hedge","reason":"members flagged froth"},"notes":"debated"})

def test_hedge_from_parses_decision():
    assert _hedge_from(COUNCIL_OUT)=="hedge"
    assert _hedge_from(json.dumps({"hedge_decision":{"stance":"no_hedge"}}))=="no_hedge"
    assert _hedge_from("garbage")=="no_hedge"   # default

def test_deliberate_returns_period_and_hedge():
    mv={"a":_pv([("zk","bullish",80)]),"b":_pv([("zk","bullish",70)])}
    with patch("signals.council.run_claude", return_value=COUNCIL_OUT):
        p,h=deliberate(date(2024,6,30),mv)
    assert p.items[0].item=="zk" and h=="hedge"

def test_run_a2b_end_to_end(tmp_path):
    root=tmp_path/"members"
    for slug,items in [("m1",[("zk","bullish",80)]),("m2",[("zk","bullish",70)])]:
        (root/slug/"periods").mkdir(parents=True)
        (root/slug/"periods"/"2024-06-30.json").write_text(json.dumps(_pv(items).to_dict()))
    canon=json.dumps({"mapping":[{"raw":"zk","canonical":"zk","item_type":"sector","parent_sector":None,"is_new":False}]})
    def fake(system,user,**kw): return canon if "registry" in user else COUNCIL_OUT
    out=tmp_path/"a2b"
    with patch("signals.council.run_claude",side_effect=fake), patch("signals.canonicalize.run_claude",side_effect=fake):
        df=run_a2b([date(2024,6,30)], members_root=root, out_dir=out, registry_path=tmp_path/"reg.json")
    assert (out/"signal_panel.parquet").exists() and (out/"hedge_decisions.json").exists()
    assert json.load(open(out/"hedge_decisions.json"))["2024-06-30"]=="hedge"
    assert (df["item"]=="zk").any()
