"""
market_data.coinglass — standalone async Coinglass market-data client.

Pulls daily OHLCV + funding-rate + open-interest history for any token universe.
Rate-limited to 75% of the standard 300 req/min plan (shared API key).

Usage (sync entrypoint):
    from market_data.coinglass import pull
    data = pull(["BTC", "ETH"])          # persists parquets to DATA_DIR

Usage (async):
    async with httpx.AsyncClient(...) as http:
        client = Coinglass(http=http, limiter=..., semaphore=...)
        result = await client.fetch_symbol("BTC")
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

COINGLASS_BASE = "https://open-api-v4.coinglass.com"
COINGLASS_OP_ITEM = "coinglass"
COINGLASS_OP_VAULT = "local"

EXCHANGE = "Binance"
INTERVAL = "1d"
LIMIT = 4500

HTTP_TIMEOUT = 60.0

RATE_LIMIT_PER_MIN = 300      # standard plan ceiling
RATE_BUDGET = 0.75            # fraction we may consume (shared key)
MAX_CONCURRENCY = 8           # max simultaneous in-flight requests

# Backoff for rate-limit retries
_BACKOFF_BASE = 8.0           # seconds
_BACKOFF_FACTOR = 2.0
_BACKOFF_MAX_RETRIES = 4

# Persistence
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "market_data"

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# API-key helper
# ──────────────────────────────────────────────────────────────────────────────

def api_key() -> str:
    """Retrieve the Coinglass API key from 1Password (subprocess, never hardcoded)."""
    result = subprocess.run(
        [
            "op", "item", "get", COINGLASS_OP_ITEM,
            "--vault", COINGLASS_OP_VAULT,
            "--fields", "credential",
            "--reveal",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Paces async requests to at most `per_minute` issuances, shared across all callers."""

    def __init__(self, per_minute: float) -> None:
        self._interval = 60.0 / per_minute        # min seconds between issuances
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait < 0:
                wait = 0.0
                self._next_at = now
            self._next_at += self._interval
        if wait > 0:
            await asyncio.sleep(wait)


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

_OHLCV_COLS = ["date", "open", "high", "low", "close", "volume_usd"]
_OHLC_COLS = ["date", "open", "high", "low", "close"]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(columns=columns)
    df = df.astype({c: float for c in columns[1:]})
    df["date"] = df["date"].astype("datetime64[ns, UTC]")
    return df


def _parse_ohlcv(data: list[dict[str, Any]]) -> pd.DataFrame:
    if not data:
        return _empty_frame(_OHLCV_COLS)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume_usd"] = df["volume_usd"].astype(float)
    df = df[_OHLCV_COLS].sort_values("date").reset_index(drop=True)
    return df


def _parse_ohlc(data: list[dict[str, Any]]) -> pd.DataFrame:
    if not data:
        return _empty_frame(_OHLC_COLS)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df = df[_OHLC_COLS].sort_values("date").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Rate-limit detection
# ──────────────────────────────────────────────────────────────────────────────

def _is_rate_limited(status_code: int, payload: dict[str, Any]) -> bool:
    if status_code == 429:
        return True
    if payload.get("code") != "0":
        msg = payload.get("msg", "")
        if "Too Many Requests" in msg or "rate" in msg.lower():
            return True
    return False


def _is_error(status_code: int, payload: dict[str, Any]) -> bool:
    return payload.get("code") != "0"


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────

class Coinglass:
    """Async Coinglass market-data client."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        limiter: RateLimiter,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._http = http
        self._limiter = limiter
        self._semaphore = semaphore

    # ── Low-level request ────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Issue one rate-limited GET with retry on rate-limit responses."""
        url = f"{COINGLASS_BASE}{path}"

        for attempt in range(_BACKOFF_MAX_RETRIES + 1):
            await self._limiter.acquire()
            async with self._semaphore:
                resp = await self._http.get(url, params=params)
                payload: dict[str, Any] = resp.json()

            if _is_rate_limited(resp.status_code, payload):
                if attempt < _BACKOFF_MAX_RETRIES:
                    sleep_secs = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                    logger.warning(
                        "Rate limited on %s (attempt %d/%d), sleeping %.1fs",
                        path, attempt + 1, _BACKOFF_MAX_RETRIES, sleep_secs,
                    )
                    await asyncio.sleep(sleep_secs)
                    continue
                else:
                    raise RuntimeError(
                        f"Rate limited after {_BACKOFF_MAX_RETRIES} retries: {payload}"
                    )

            if _is_error(resp.status_code, payload):
                msg = payload.get("msg", str(payload))
                raise RuntimeError(f"Coinglass API error: {msg}")

            return payload

        raise RuntimeError("Unexpected: exhausted retry loop")  # pragma: no cover

    # ── Public data methods ──────────────────────────────────────────────────

    async def ohlcv(self, symbol: str) -> pd.DataFrame:
        """Daily OHLCV for `symbol` (e.g. "BTC"). Returns tz-aware-UTC DataFrame."""
        pair = f"{symbol}USDT"
        try:
            payload = await self._get(
                "/api/futures/price/history",
                params={"exchange": EXCHANGE, "symbol": pair, "interval": INTERVAL, "limit": LIMIT},
            )
        except RuntimeError:
            raise
        data = payload.get("data", [])
        if not data:
            logger.info("ohlcv(%s): empty data returned", symbol)
            return _empty_frame(_OHLCV_COLS)
        return _parse_ohlcv(data)

    async def funding(self, symbol: str) -> pd.DataFrame:
        """Daily funding-rate OHLC for `symbol`. Returns tz-aware-UTC DataFrame."""
        pair = f"{symbol}USDT"
        try:
            payload = await self._get(
                "/api/futures/funding-rate/history",
                params={"exchange": EXCHANGE, "symbol": pair, "interval": INTERVAL, "limit": LIMIT},
            )
        except RuntimeError:
            raise
        data = payload.get("data", [])
        if not data:
            logger.info("funding(%s): empty data returned", symbol)
            return _empty_frame(_OHLC_COLS)
        return _parse_ohlc(data)

    async def open_interest(self, symbol: str) -> pd.DataFrame:
        """Daily open-interest OHLC for `symbol`. Returns tz-aware-UTC DataFrame."""
        pair = f"{symbol}USDT"
        try:
            payload = await self._get(
                "/api/futures/open-interest/history",
                params={"exchange": EXCHANGE, "symbol": pair, "interval": INTERVAL, "limit": LIMIT},
            )
        except RuntimeError:
            raise
        data = payload.get("data", [])
        if not data:
            logger.info("open_interest(%s): empty data returned", symbol)
            return _empty_frame(_OHLC_COLS)
        return _parse_ohlc(data)

    # ── Composite fetch methods ──────────────────────────────────────────────

    async def fetch_symbol(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV + funding + OI for one symbol concurrently."""
        ohlcv_df, funding_df, oi_df = await asyncio.gather(
            self.ohlcv(symbol),
            self.funding(symbol),
            self.open_interest(symbol),
        )
        return {"ohlcv": ohlcv_df, "funding": funding_df, "oi": oi_df}

    async def fetch_universe(
        self, symbols: list[str]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetch all three series for every symbol, all under the shared limiter/semaphore."""
        tasks = [self.fetch_symbol(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_symbol(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    out_dir: Path | None = None,
) -> None:
    """
    Write one parquet per series to <out_dir>/<symbol>_<series>_1d.parquet.
    Series keys: ohlcv, funding, oi.
    Creates out_dir if it does not exist.
    """
    if out_dir is None:
        out_dir = DATA_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for series, df in frames.items():
        path = out_dir / f"{symbol}_{series}_1d.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Sync convenience entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def pull(
    symbols: list[str],
    persist: bool = True,
    out_dir: Path | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Sync entrypoint: fetch OHLCV + funding + OI for each symbol and optionally
    persist parquets to out_dir (default DATA_DIR). Returns the universe dict.
    """
    key = api_key()
    limiter = RateLimiter(RATE_LIMIT_PER_MIN * RATE_BUDGET)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _run() -> dict[str, dict[str, pd.DataFrame]]:
        async with httpx.AsyncClient(
            headers={"CG-API-KEY": key},
            timeout=HTTP_TIMEOUT,
        ) as http:
            client = Coinglass(http=http, limiter=limiter, semaphore=semaphore)
            return await client.fetch_universe(symbols)

    universe = asyncio.run(_run())

    if persist:
        for symbol, frames in universe.items():
            save_symbol(symbol, frames, out_dir=out_dir)

    return universe
