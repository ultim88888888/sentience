"""
TDD tests for market_data.coinglass — Coinglass async market-data client.

Execution: .venv/bin/python -m pytest tests/test_market_data_coinglass.py -v
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────

OHLCV_PAYLOAD = {
    "code": "0",
    "data": [
        {"time": 1_577_836_800_000, "open": "7200.0", "high": "7350.0", "low": "7100.0", "close": "7300.0", "volume_usd": "1234567.89"},
        {"time": 1_577_923_200_000, "open": "7300.0", "high": "7500.0", "low": "7250.0", "close": "7450.0", "volume_usd": "2345678.90"},
    ],
}

FUNDING_PAYLOAD = {
    "code": "0",
    "data": [
        {"time": 1_577_836_800_000, "open": "0.0001", "high": "0.0002", "low": "0.00005", "close": "0.00015"},
        {"time": 1_577_923_200_000, "open": "0.00015", "high": "0.00025", "low": "0.0001", "close": "0.0002"},
    ],
}

OI_PAYLOAD = {
    "code": "0",
    "data": [
        {"time": 1_577_836_800_000, "open": 5_000_000.0, "high": 5_500_000.0, "low": 4_800_000.0, "close": 5_200_000.0},
        {"time": 1_577_923_200_000, "open": 5_200_000.0, "high": 5_700_000.0, "low": 5_100_000.0, "close": 5_600_000.0},
    ],
}

EMPTY_PAYLOAD = {"code": "0", "data": []}

RATE_LIMIT_PAYLOAD = {"code": "1", "msg": "Too Many Requests"}


def _make_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def _async_resp(payload: dict, status_code: int = 200) -> AsyncMock:
    """AsyncMock that returns a response with the given payload."""
    resp = _make_response(payload, status_code)
    m = AsyncMock(return_value=resp)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper: build a Coinglass client with a no-op limiter/semaphore
# ──────────────────────────────────────────────────────────────────────────────

def _make_client(semaphore_value: int = 100) -> "Coinglass":
    """Return a Coinglass client with a very-high-rate limiter and given semaphore."""
    from market_data.coinglass import Coinglass, RateLimiter
    import httpx

    # Very high rate so tests don't sleep
    limiter = RateLimiter(per_minute=100_000.0)
    sem = asyncio.Semaphore(semaphore_value)
    http = MagicMock(spec=httpx.AsyncClient)
    return Coinglass(http=http, limiter=limiter, semaphore=sem)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Rate-limiter pacing
# ──────────────────────────────────────────────────────────────────────────────

async def test_rate_limiter_pacing():
    """
    With a controllable fake clock, N acquire() calls should produce waits
    that enforce ~0.2667s spacing (60 / 225 per issuance).
    """
    from market_data.coinglass import RateLimiter

    EXPECTED_INTERVAL = 60.0 / 225  # ≈ 0.26667

    fake_time = 0.0
    recorded_waits: list[float] = []

    def fake_monotonic() -> float:
        return fake_time

    async def fake_sleep(seconds: float) -> None:
        nonlocal fake_time
        recorded_waits.append(seconds)
        fake_time += seconds  # advance fake clock by the sleep duration

    limiter = RateLimiter(per_minute=225.0)

    with (
        patch("market_data.coinglass.time.monotonic", side_effect=fake_monotonic),
        patch("market_data.coinglass.asyncio.sleep", side_effect=fake_sleep),
    ):
        N = 5
        for _ in range(N):
            await limiter.acquire()

    # First call: no prior issuance, so the first wait may be 0.
    # Calls 2..N should each wait ≈ EXPECTED_INTERVAL.
    assert len(recorded_waits) >= N - 1, f"Expected at least {N-1} sleep calls, got {len(recorded_waits)}"
    for w in recorded_waits:
        assert w > 0, "All recorded sleep waits must be positive"
        assert abs(w - EXPECTED_INTERVAL) < 0.01, (
            f"Wait {w:.4f}s deviates too far from expected {EXPECTED_INTERVAL:.4f}s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. OHLCV parse
# ──────────────────────────────────────────────────────────────────────────────

async def test_ohlcv_parse():
    """Mocked OHLCV response parses to correct columns, types, and sort order."""
    from market_data.coinglass import Coinglass

    client = _make_client()
    client._http.get = AsyncMock(return_value=_make_response(OHLCV_PAYLOAD))

    df = await client.ohlcv("BTC")
    _assert_ohlcv_frame(df)


def _assert_ohlcv_frame(df: pd.DataFrame) -> None:
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume_usd"]
    assert len(df) == 2
    assert str(df["date"].dtype) == "datetime64[ns, UTC]", f"date dtype: {df['date'].dtype}"
    assert df["date"].is_monotonic_increasing
    for col in ["open", "high", "low", "close", "volume_usd"]:
        assert df[col].dtype == float, f"{col} dtype: {df[col].dtype}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Funding parse
# ──────────────────────────────────────────────────────────────────────────────

async def test_funding_parse():
    """Mocked funding response parses to correct columns and types."""
    from market_data.coinglass import Coinglass

    client = _make_client()
    client._http.get = AsyncMock(return_value=_make_response(FUNDING_PAYLOAD))

    df = await client.funding("BTC")
    _assert_ohlc_frame(df, "funding")


# ──────────────────────────────────────────────────────────────────────────────
# 4. OI parse
# ──────────────────────────────────────────────────────────────────────────────

async def test_oi_parse():
    """Mocked OI response parses to correct columns and types."""
    from market_data.coinglass import Coinglass

    client = _make_client()
    client._http.get = AsyncMock(return_value=_make_response(OI_PAYLOAD))

    df = await client.open_interest("BTC")
    _assert_ohlc_frame(df, "oi")


def _assert_ohlc_frame(df: pd.DataFrame, name: str) -> None:
    assert list(df.columns) == ["date", "open", "high", "low", "close"], f"{name} columns: {list(df.columns)}"
    assert len(df) == 2
    assert str(df["date"].dtype) == "datetime64[ns, UTC]", f"{name} date dtype: {df['date'].dtype}"
    assert df["date"].is_monotonic_increasing
    for col in ["open", "high", "low", "close"]:
        assert df[col].dtype == float, f"{name} {col} dtype: {df[col].dtype}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Empty data
# ──────────────────────────────────────────────────────────────────────────────

async def test_empty_ohlcv():
    """Empty data payload returns a correctly-typed empty frame, no exception."""
    from market_data.coinglass import Coinglass

    client = _make_client()
    client._http.get = AsyncMock(return_value=_make_response(EMPTY_PAYLOAD))

    df = await client.ohlcv("BTC")

    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume_usd"]
    assert len(df) == 0
    assert str(df["date"].dtype) == "datetime64[ns, UTC]", f"empty date dtype: {df['date'].dtype}"


async def test_empty_funding():
    from market_data.coinglass import Coinglass

    client = _make_client()
    client._http.get = AsyncMock(return_value=_make_response(EMPTY_PAYLOAD))

    df = await client.funding("BTC")

    assert list(df.columns) == ["date", "open", "high", "low", "close"]
    assert len(df) == 0
    assert str(df["date"].dtype) == "datetime64[ns, UTC]"


async def test_empty_oi():
    from market_data.coinglass import Coinglass

    client = _make_client()
    client._http.get = AsyncMock(return_value=_make_response(EMPTY_PAYLOAD))

    df = await client.open_interest("BTC")

    assert list(df.columns) == ["date", "open", "high", "low", "close"]
    assert len(df) == 0
    assert str(df["date"].dtype) == "datetime64[ns, UTC]"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Rate-limit retry
# ──────────────────────────────────────────────────────────────────────────────

async def test_rate_limit_retry_succeeds():
    """
    When the first response is rate-limited and the second is good,
    the client retries and returns parsed data.
    """
    from market_data.coinglass import Coinglass

    client = _make_client()

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_response(RATE_LIMIT_PAYLOAD)
        return _make_response(OHLCV_PAYLOAD)

    client._http.get = side_effect

    with patch("market_data.coinglass.asyncio.sleep", new=AsyncMock()):
        df = await client.ohlcv("BTC")

    assert call_count == 2, f"Expected 2 calls (1 retry), got {call_count}"
    _assert_ohlcv_frame(df)


async def test_rate_limit_retry_429():
    """HTTP 429 status triggers retry and eventually succeeds."""
    from market_data.coinglass import Coinglass

    client = _make_client()

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_response({"code": "1", "msg": "rate limited"}, status_code=429)
        return _make_response(OHLCV_PAYLOAD)

    client._http.get = side_effect

    with patch("market_data.coinglass.asyncio.sleep", new=AsyncMock()):
        df = await client.ohlcv("BTC")

    assert call_count == 2
    _assert_ohlcv_frame(df)


async def test_genuine_error_raises_runtime_error():
    """Non-rate-limit API error raises RuntimeError immediately with no retry."""
    from market_data.coinglass import Coinglass

    client = _make_client()
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_response({"code": "1", "msg": "Invalid symbol"})

    client._http.get = side_effect

    with (
        patch("market_data.coinglass.asyncio.sleep", new=AsyncMock()),
        pytest.raises(RuntimeError, match="Invalid symbol"),
    ):
        await client.ohlcv("INVALID")

    assert call_count == 1, f"Expected exactly 1 call (no retry), got {call_count}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Partial-failure universe
# ──────────────────────────────────────────────────────────────────────────────

async def test_fetch_universe_tolerates_partial_failure():
    """fetch_universe skips failed symbols and returns only the successes."""
    from market_data.coinglass import Coinglass

    client = _make_client()

    good_frames = {"ohlcv": _sample_ohlcv_df(), "funding": _sample_funding_df(), "oi": _sample_oi_df()}

    async def fetch_symbol_side_effect(symbol: str):
        if symbol == "BAD":
            raise RuntimeError("simulated failure for BAD")
        return good_frames

    with patch.object(Coinglass, "fetch_symbol", side_effect=fetch_symbol_side_effect):
        result = await client.fetch_universe(["GOOD1", "BAD", "GOOD2"])

    assert "GOOD1" in result, "GOOD1 should be in result"
    assert "GOOD2" in result, "GOOD2 should be in result"
    assert "BAD" not in result, "BAD should be absent from result"
    assert len(result) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 8. Concurrency bound
# ──────────────────────────────────────────────────────────────────────────────

async def test_concurrency_bound():
    """
    fetch_universe over 20 fake symbols never exceeds MAX_CONCURRENCY concurrent requests.
    The client is constructed with semaphore=MAX_CONCURRENCY (not the permissive 100).

    Implementation note:
    patch("market_data.coinglass.asyncio.sleep") patches the asyncio module object
    globally (since the module holds a reference to asyncio itself), which would also
    no-op the slow_get's sleep and prevent real concurrency observation.
    We therefore capture the real asyncio.sleep BEFORE patching and call it explicitly
    inside slow_get to guarantee actual event-loop suspension.
    """
    from market_data.coinglass import Coinglass, MAX_CONCURRENCY, RateLimiter
    import httpx

    # Capture real sleep before any patching
    _real_sleep = asyncio.sleep

    # Build client with the actual concurrency bound
    limiter = RateLimiter(per_minute=100_000.0)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    http = MagicMock(spec=httpx.AsyncClient)
    client = Coinglass(http=http, limiter=limiter, semaphore=sem)

    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()
    call_num = 0

    async def slow_get(*args, **kwargs):
        nonlocal concurrent_count, max_concurrent, call_num
        async with lock:
            concurrent_count += 1
            if concurrent_count > max_concurrent:
                max_concurrent = concurrent_count
            call_num += 1
            cn = call_num

        # Use the real (pre-patch) asyncio.sleep to ensure actual event-loop suspension,
        # regardless of whether the module-level asyncio.sleep is mocked.
        await _real_sleep(0.002)

        async with lock:
            concurrent_count -= 1

        # Cycle through response types: ohlcv, funding, oi per symbol
        remainder = (cn - 1) % 3
        if remainder == 0:
            return _make_response(OHLCV_PAYLOAD)
        elif remainder == 1:
            return _make_response(FUNDING_PAYLOAD)
        else:
            return _make_response(OI_PAYLOAD)

    symbols = [f"TOKEN{i}" for i in range(20)]

    client._http.get = slow_get

    # Patch the module-level asyncio.sleep to no-op rate-limiter and backoff waits.
    # slow_get uses _real_sleep directly so it is unaffected by this patch.
    with patch("market_data.coinglass.asyncio.sleep", new=AsyncMock()):
        await client.fetch_universe(symbols)

    assert max_concurrent <= MAX_CONCURRENCY, (
        f"Max concurrent {max_concurrent} exceeded MAX_CONCURRENCY={MAX_CONCURRENCY}"
    )
    assert max_concurrent > 1, f"Expected some concurrency (>1), got {max_concurrent}"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Persistence round-trip
# ──────────────────────────────────────────────────────────────────────────────

def test_persistence_round_trip(tmp_path: Path):
    """save_symbol writes three parquets; reloaded frames equal originals."""
    from market_data.coinglass import save_symbol
    import pandas as pd

    ohlcv_df = _sample_ohlcv_df()
    funding_df = _sample_funding_df()
    oi_df = _sample_oi_df()

    frames = {"ohlcv": ohlcv_df, "funding": funding_df, "oi": oi_df}
    save_symbol("BTC", frames, out_dir=tmp_path)

    for series, expected in frames.items():
        fpath = tmp_path / f"BTC_{series}_1d.parquet"
        assert fpath.exists(), f"Missing parquet: {fpath}"
        loaded = pd.read_parquet(fpath)
        pd.testing.assert_frame_equal(loaded, expected)


def _sample_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [7200.0, 7300.0],
        "high": [7350.0, 7500.0],
        "low": [7100.0, 7250.0],
        "close": [7300.0, 7450.0],
        "volume_usd": [1234567.89, 2345678.90],
    })


def _sample_funding_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [0.0001, 0.00015],
        "high": [0.0002, 0.00025],
        "low": [0.00005, 0.0001],
        "close": [0.00015, 0.0002],
    })


def _sample_oi_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [5_000_000.0, 5_200_000.0],
        "high": [5_500_000.0, 5_700_000.0],
        "low": [4_800_000.0, 5_100_000.0],
        "close": [5_200_000.0, 5_600_000.0],
    })


# ──────────────────────────────────────────────────────────────────────────────
# 9. Live integration (gated)
# ──────────────────────────────────────────────────────────────────────────────

_op_available = shutil.which("op") is not None


@pytest.mark.skipif(not _op_available, reason="1Password CLI (op) not available")
async def test_live_btc_pull():
    """
    Real network call — pulls BTC only. Asserts non-empty OHLCV reaching <= 2020,
    non-empty funding, non-empty OI. Skipped if op is absent.

    NOTE: Uses the shared API key. Keep to ONE symbol.
    """
    from market_data.coinglass import Coinglass, api_key, RateLimiter, RATE_LIMIT_PER_MIN, RATE_BUDGET, MAX_CONCURRENCY
    import httpx

    key = api_key()
    assert key, "API key must be non-empty"

    limiter = RateLimiter(RATE_LIMIT_PER_MIN * RATE_BUDGET)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(
        headers={"CG-API-KEY": key},
        timeout=60.0,
    ) as http:
        client = Coinglass(http=http, limiter=limiter, semaphore=sem)
        result = await client.fetch_symbol("BTC")

    ohlcv = result["ohlcv"]
    funding = result["funding"]
    oi = result["oi"]

    assert len(ohlcv) > 0, "OHLCV should be non-empty for BTC"
    assert len(funding) > 0, "Funding should be non-empty for BTC"
    assert len(oi) > 0, "OI should be non-empty for BTC"

    # BTC OHLCV should reach back to at least 2020
    earliest = ohlcv["date"].min()
    assert earliest.year <= 2020, f"Expected OHLCV data back to <= 2020, got {earliest}"

    # All dates tz-aware UTC
    assert str(ohlcv["date"].dtype) == "datetime64[ns, UTC]"
    assert str(funding["date"].dtype) == "datetime64[ns, UTC]"
    assert str(oi["date"].dtype) == "datetime64[ns, UTC]"
