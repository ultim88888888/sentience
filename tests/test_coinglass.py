import shutil
import httpx
import pandas as pd
import pytest
from study import coinglass


def _have_key():
    return shutil.which("op") is not None


@pytest.mark.skipif(not _have_key(), reason="op CLI / Coinglass key unavailable")
def test_price_history_btc_reaches_back_to_2020():
    df = coinglass.price_history("BTC", use_cache=False)
    assert not df.empty
    assert set(["date", "close"]).issubset(df.columns)
    assert df["date"].min() <= pd.Timestamp("2020-12-31", tz="UTC")
    assert (df["close"] > 0).all()


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 429:
            raise httpx.HTTPStatusError("err", request=None, response=None)
    def json(self):
        return self._payload


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    import httpx as _httpx
    sleeps = []
    monkeypatch.setattr(coinglass, "_throttle", lambda: None)
    monkeypatch.setattr(coinglass.time, "sleep", lambda s=0: sleeps.append(s))
    monkeypatch.setattr(coinglass, "api_key", lambda: "dummy")
    good = {"code": "0", "data": [{"time": 1700000000000, "close": "100.5"}]}
    responses = iter([
        _FakeResp(200, {"code": "50001", "msg": "Too Many Requests"}),
        _FakeResp(429, {}),
        _FakeResp(200, good),
    ])
    monkeypatch.setattr(_httpx, "get", lambda *a, **k: next(responses))
    df = coinglass.price_history("BTC", use_cache=False)
    assert len(df) == 1
    assert df["close"].iloc[0] == 100.5
    assert len(sleeps) == 2  # two rate-limited responses -> two backoff sleeps, no wasted final sleep


def test_genuine_error_raises_immediately(monkeypatch):
    import httpx as _httpx
    monkeypatch.setattr(coinglass.time, "sleep", lambda *_: None)
    monkeypatch.setattr(coinglass, "api_key", lambda: "dummy")
    monkeypatch.setattr(_httpx, "get",
                        lambda *a, **k: _FakeResp(200, {"code": "400", "msg": "bad symbol"}))
    import pytest
    with pytest.raises(RuntimeError, match="bad symbol"):
        coinglass.price_history("NOPE", use_cache=False)
