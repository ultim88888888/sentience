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
