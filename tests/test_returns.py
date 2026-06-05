import pandas as pd
from study import returns


def test_basket_returns_equal_weight_and_n_live(synthetic_token_returns, baskets_cfg):
    br = returns.basket_returns(synthetic_token_returns, baskets_cfg)
    jan_zk = br[(br["month"] == pd.Period("2024-01", "M")) & (br["basket"] == "ZK/Privacy")]
    # ZK basket = [ZK, STRK]; Jan returns 0.10 & 0.15 -> mean 0.125, n_live 2
    assert abs(jan_zk["ret"].iloc[0] - 0.125) < 1e-9
    assert jan_zk["n_live"].iloc[0] == 2


def test_basket_ineligible_when_no_live_constituents(baskets_cfg):
    # DeFi tokens (UNI, AAVE) have no rows -> DeFi should be absent / NaN, not crash.
    tr = pd.DataFrame({"month": [pd.Period("2024-01", "M")], "token": ["BTC"], "ret": [0.02]})
    br = returns.basket_returns(tr, baskets_cfg)
    defi = br[br["basket"] == "DeFi"]
    assert defi.empty or defi["n_live"].fillna(0).eq(0).all()


def test_monthly_token_returns_from_prices():
    # Multiple daily observations per month -> resample("ME").last() must pick the
    # month-end close. Jan ends at 100, Feb ends at 110 -> Feb return 0.10.
    prices = pd.DataFrame({
        "date": pd.to_datetime(
            ["2024-01-05", "2024-01-20", "2024-01-31",
             "2024-02-10", "2024-02-29"], utc=True),
        "close": [90.0, 95.0, 100.0, 105.0, 110.0],
    })
    tr = returns._monthly_returns_from_prices(prices)
    feb = tr[tr["month"] == pd.Period("2024-02", "M")]["ret"].iloc[0]
    assert abs(feb - 0.10) < 1e-9
    # Jan is the first month -> its pct_change is NaN and dropped, so only Feb remains.
    assert tr["month"].tolist() == [pd.Period("2024-02", "M")]
