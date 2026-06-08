import pandas as pd
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal
from signals.panel import derive_panel

def _it(item, stance, conv=70, typ="sector", parent=None, horizon="structural"):
    return SignalItem(item=item, item_type=typ, parent_sector=parent, stance=stance,
                      conviction=conv, horizon=horizon, rationale="r",
                      provenance="grounded", age_note=None,
                      citations=(Citation("2023-01-01", "q"),))

def _period(date, items):
    return PeriodSignal(as_of=date, approach="A1", items=tuple(items),
                        risk_regime=RiskRegime("risk_on", 60, "w", "grounded"))

def test_new_then_sustained_then_flip_then_exit():
    periods = [
        _period("2023-03-31", [_it("zk", "bullish", 70)]),
        _period("2023-06-30", [_it("zk", "bullish", 85)]),     # sustained, conv up
        _period("2023-09-30", [_it("zk", "bearish", 60)]),     # flip
        _period("2023-12-31", []),                              # zk exits
    ]
    df = derive_panel(periods)
    zk = df[df["item"] == "zk"].sort_values("as_of").reset_index(drop=True)
    assert list(zk["lifecycle_state"]) == ["NEW", "SUSTAINED", "FLIPPED", "EXITED"]
    assert list(zk["age"]) == [1, 2, 3, 0]                     # age resets at exit
    assert zk.loc[1, "delta_conviction"] == 15                 # 85 - 70
    assert zk.loc[2, "delta_stance"] == -2                     # bullish(+1) -> bearish(-1)

def test_exit_row_is_synthetic_with_zero_conviction():
    periods = [
        _period("2023-03-31", [_it("defi", "bullish", 50)]),
        _period("2023-06-30", []),
    ]
    df = derive_panel(periods)
    exit_row = df[(df["item"] == "defi") & (df["as_of"] == "2023-06-30")].iloc[0]
    assert exit_row["lifecycle_state"] == "EXITED"
    assert exit_row["conviction"] == 0
    assert exit_row["stance"] == "neutral"

def test_token_parent_sector_carried_into_panel():
    periods = [_period("2023-03-31", [_it("HYPE", "bullish", typ="token", parent="perp-dex")])]
    df = derive_panel(periods)
    assert df.iloc[0]["parent_sector"] == "perp-dex"

def test_reentry_after_exit_is_new_again_with_age_reset():
    periods = [
        _period("2023-03-31", [_it("gaming", "bullish")]),
        _period("2023-06-30", []),                              # exit
        _period("2023-09-30", [_it("gaming", "bullish")]),      # re-enter
    ]
    df = derive_panel(periods)
    g = df[df["item"] == "gaming"].sort_values("as_of").reset_index(drop=True)
    assert list(g["lifecycle_state"]) == ["NEW", "EXITED", "NEW"]
    assert list(g["age"]) == [1, 0, 1]


def test_exited_token_retains_type_and_parent():
    periods = [
        _period("2023-03-31", [_it("HYPE", "bullish", typ="token", parent="perp-dex")]),
        _period("2023-06-30", []),
    ]
    df = derive_panel(periods)
    ex = df[(df["item"] == "HYPE") & (df["as_of"] == "2023-06-30")].iloc[0]
    assert ex["lifecycle_state"] == "EXITED"
    assert ex["item_type"] == "token"
    assert ex["parent_sector"] == "perp-dex"
