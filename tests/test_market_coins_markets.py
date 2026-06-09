"""
TDD tests for market_data.coinglass.Coinglass.coins_markets.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from market_data.coinglass import Coinglass, RateLimiter


_COINS_MARKETS_PAYLOAD = {
    "code": "0",
    "data": [
        {
            "symbol": "BTC",
            "open_interest_usd": 4.4e10,
            "market_cap_usd": 1.2e12,
            "current_price": 62000.0,
            "avg_funding_rate_by_oi": 0.003,
        }
    ],
}


def _make_client() -> Coinglass:
    limiter = RateLimiter(per_minute=100_000.0)
    sem = asyncio.Semaphore(100)
    http = MagicMock(spec=httpx.AsyncClient)
    return Coinglass(http=http, limiter=limiter, semaphore=sem)


async def test_coins_markets_returns_list():
    """coins_markets() returns the data list from the API response."""
    client = _make_client()
    with patch.object(client, "_get", new=AsyncMock(return_value=_COINS_MARKETS_PAYLOAD)):
        result = await client.coins_markets()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["symbol"] == "BTC"
    assert result[0]["open_interest_usd"] == 4.4e10
    assert result[0]["market_cap_usd"] == 1.2e12
    assert result[0]["current_price"] == 62000.0
    assert result[0]["avg_funding_rate_by_oi"] == 0.003


async def test_coins_markets_passes_correct_path():
    """coins_markets() calls _get with the coins-markets endpoint."""
    client = _make_client()
    mock_get = AsyncMock(return_value={"code": "0", "data": []})
    with patch.object(client, "_get", new=mock_get):
        await client.coins_markets()
    mock_get.assert_called_once_with("/api/futures/coins-markets", {})


async def test_coins_markets_empty_data():
    """coins_markets() returns [] when data is empty."""
    client = _make_client()
    with patch.object(client, "_get", new=AsyncMock(return_value={"code": "0", "data": []})):
        result = await client.coins_markets()
    assert result == []


async def test_coins_markets_missing_data_key():
    """coins_markets() returns [] when the data key is absent."""
    client = _make_client()
    with patch.object(client, "_get", new=AsyncMock(return_value={"code": "0"})):
        result = await client.coins_markets()
    assert result == []


async def test_coins_markets_non_list_data():
    """coins_markets() returns [] when data is not a list (defensive)."""
    client = _make_client()
    with patch.object(client, "_get", new=AsyncMock(return_value={"code": "0", "data": None})):
        result = await client.coins_markets()
    assert result == []
