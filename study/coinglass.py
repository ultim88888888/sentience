"""Coinglass v4 client: daily perp OHLC since inception, with an on-disk parquet cache.

Key lives in 1Password (vault 'local', item 'coinglass'); read via the `op` CLI, never
hardcoded. Validated: BTCUSDT history reaches 2019-09, STRKUSDT starts at its 2024 listing.
The public API rate-limits aggressively, so live calls are throttled and retried with
exponential backoff; genuine (non-rate-limit) errors raise immediately.
"""
import subprocess
import sys
import time

import httpx
import pandas as pd

from .config import (COINGLASS_BASE, COINGLASS_OP_ITEM, COINGLASS_OP_VAULT, EXCHANGE,
                     HTTP_TIMEOUT, PRICE_CACHE_DIR, PRICE_INTERVAL, PRICE_LIMIT,
                     RATE_LIMIT_BACKOFF, RATE_LIMIT_RETRIES, REQUEST_THROTTLE)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def api_key() -> str:
    """Read the Coinglass API key from 1Password (vault 'local')."""
    return subprocess.check_output(
        ["op", "item", "get", COINGLASS_OP_ITEM, "--vault", COINGLASS_OP_VAULT,
         "--fields", "credential", "--reveal"], text=True).strip()


def _symbol_to_pair(symbol: str) -> str:
    return f"{symbol}USDT"


def _is_rate_limit(status_code: int, msg: str) -> bool:
    return status_code == 429 or "too many requests" in msg.lower() or "rate limit" in msg.lower()


_last_call = [0.0]


def _throttle() -> None:
    """Enforce a minimum interval between live API calls."""
    elapsed = time.time() - _last_call[0]
    if elapsed < REQUEST_THROTTLE:
        time.sleep(REQUEST_THROTTLE - elapsed)
    _last_call[0] = time.time()


def _get_json(url: str, params: dict, label: str) -> dict:
    """GET Coinglass JSON, retrying with exponential backoff on rate-limiting only.

    Returns the parsed payload (code == "0"). Raises RuntimeError on genuine API errors
    or if still rate-limited after RATE_LIMIT_RETRIES attempts.
    """
    headers = {"CG-API-KEY": api_key()}
    attempts = RATE_LIMIT_RETRIES + 1          # 1 initial call + RATE_LIMIT_RETRIES retries
    for attempt in range(attempts):
        _throttle()
        r = httpx.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        status = r.status_code
        msg = ""
        if status != 429:
            r.raise_for_status()
            payload = r.json()
            if str(payload.get("code")) == "0":
                return payload
            msg = str(payload.get("msg") or "")
            if not _is_rate_limit(status, msg):
                raise RuntimeError(f"Coinglass error for {label}: {msg}")
        if attempt == attempts - 1:
            break                              # final attempt: don't sleep, fall through to raise
        wait = RATE_LIMIT_BACKOFF * (2 ** attempt)
        _log(f"  {label}: rate limited ({msg or 'HTTP 429'}); "
             f"backoff {wait:.0f}s (attempt {attempt + 1}/{attempts})")
        time.sleep(wait)
    raise RuntimeError(f"Coinglass still rate-limited for {label} after "
                       f"{RATE_LIMIT_RETRIES} retries")


def price_history(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """Daily close price for a token's Binance USDT perp. Returns columns [date, close] (UTC).

    Empty DataFrame (same columns) if the pair has no data — caller skips the constituent.
    """
    cache = PRICE_CACHE_DIR / f"{symbol}_{PRICE_INTERVAL}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    pair = _symbol_to_pair(symbol)
    url = f"{COINGLASS_BASE}/api/futures/price/history"
    params = {"exchange": EXCHANGE, "symbol": pair,
              "interval": PRICE_INTERVAL, "limit": PRICE_LIMIT}
    payload = _get_json(url, params, label=pair)

    rows = payload.get("data") or []
    if not rows:
        _log(f"  no price history for {symbol} ({pair}); skipping")
        empty = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns, UTC]"),
                              "close": pd.Series([], dtype="float64")})
        return empty

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)

    if use_cache:
        PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df
