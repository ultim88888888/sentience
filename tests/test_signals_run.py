from datetime import date
from unittest.mock import patch
import json
import pandas as pd
from signals.run import build_panel, rebalance_dates

def test_rebalance_dates_quarterly():
    ds = rebalance_dates(date(2023, 1, 1), date(2023, 12, 31), "quarterly")
    assert ds == [date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30), date(2023, 12, 31)]

def test_rebalance_dates_monthly_count():
    ds = rebalance_dates(date(2023, 1, 1), date(2023, 12, 31), "monthly")
    assert len(ds) == 12

def test_build_panel_end_to_end_with_mocked_llm(tmp_path):
    # one sector, two quarters: NEW then SUSTAINED
    extraction = json.dumps({
        "sectors_excited": [{"name": "zk rollups", "why": "w", "conviction": 80,
            "horizon": "structural", "provenance": "grounded", "age_note": None,
            "citations": [{"date": "2023-01-15", "quote": "zk good"}]}],
        "risk_regime": {"stance": "risk_on", "conviction": 60, "why": "w", "provenance": "grounded"},
    })
    canon = json.dumps({"mapping": [{"raw": "zk rollups", "canonical": "zk",
        "item_type": "sector", "parent_sector": None, "is_new": False}]})
    arts = pd.DataFrame({"post_date": ["2023-01-15"], "extracted_text": ["zk good"],
                         "permalink": ["p1"], "object_id": ["o1"]})

    def fake_llm(system, user, **kw):
        return canon if "registry" in user else extraction
    with patch("signals.extract.run_claude", side_effect=fake_llm), \
         patch("signals.canonicalize.run_claude", side_effect=fake_llm):
        df = build_panel(date(2023, 1, 1), date(2023, 6, 30), "quarterly",
                         window_months=18, twitter_paths=[], articles=arts,
                         distillates={}, out_dir=tmp_path)
    zk = df[df["item"] == "zk"].sort_values("as_of").reset_index(drop=True)
    assert list(zk["lifecycle_state"]) == ["NEW", "SUSTAINED"]
    assert (tmp_path / "signal_panel.parquet").exists()
    assert (tmp_path / "registry.json").exists()


def test_rebalance_quarterly_handles_day31_start():
    from datetime import date
    from signals.run import rebalance_dates
    ds = rebalance_dates(date(2023, 5, 31), date(2023, 12, 31), "quarterly")
    assert ds == [date(2023, 6, 30), date(2023, 9, 30), date(2023, 12, 31)]

def test_build_panel_resumes_from_existing_period(tmp_path):
    from datetime import date
    import json, pandas as pd
    from unittest.mock import patch
    from signals.run import build_panel
    # pre-seed a period json so it should be loaded, not re-extracted
    (tmp_path / "periods").mkdir(parents=True)
    seeded = {"as_of": "2023-03-31", "approach": "A1", "items": [], "risk_regime":
              {"stance": "neutral", "conviction": 50, "rationale": "", "provenance": "extrapolated"}, "notes": "seed"}
    (tmp_path / "periods" / "2023-03-31.json").write_text(json.dumps(seeded))
    arts = pd.DataFrame({"post_date": ["2023-01-15"], "extracted_text": ["x"], "permalink": ["p"], "object_id": ["o"]})
    called = {"n": 0}
    def fake(system, user, **kw):
        called["n"] += 1
        return '{"sectors_excited": [], "risk_regime": {"stance":"neutral","conviction":50,"why":"","provenance":"extrapolated"}}'
    with patch("signals.extract.run_claude", side_effect=fake), patch("signals.canonicalize.run_claude", side_effect=fake):
        build_panel(date(2023,1,1), date(2023,3,31), "quarterly", window_months=18,
                    twitter_paths=[], articles=arts, distillates={}, out_dir=tmp_path)
    assert called["n"] == 0  # the only period was seeded → no LLM call

def test_member_panel_runs_with_mocked_llm(tmp_path):
    from datetime import date
    import pandas as pd
    from unittest.mock import patch
    from signals.run import build_member_panels
    tw = pd.DataFrame({"created_at": pd.to_datetime(["2023-01-10"], utc=True), "type":["original"],
                       "text":["A substantive thesis about zk rollups and modular data availability layers"], "url":["u"]})
    p = tmp_path / "eddy.parquet"; tw.to_parquet(p)
    ext = '{"sectors_excited":[{"name":"zk","why":"w","conviction":80,"horizon":"structural","provenance":"grounded","age_note":null,"citations":[]}],"risk_regime":{"stance":"risk_on","conviction":60,"why":"w","provenance":"grounded"}}'
    canon = '{"mapping":[{"raw":"zk","canonical":"zk","item_type":"sector","parent_sector":null,"is_new":false}]}'
    def fake(system, user, **kw):
        return canon if "registry" in user else ext
    with patch("signals.extract.run_claude", side_effect=fake), patch("signals.canonicalize.run_claude", side_effect=fake):
        res = build_member_panels([("eddy-lazzarin","Eddy Lazzarin",p)], start=date(2023,1,1),
                                   end=date(2023,3,31), interval="quarterly", window_months=18, out_root=tmp_path)
    assert res["eddy-lazzarin"] >= 1
    assert (tmp_path / "members" / "eddy-lazzarin" / "signal_panel.parquet").exists()
