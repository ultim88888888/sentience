"""Async twitterapi.io client: auth, rate-limited GET, advanced_search."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import (
    ADVANCED_SEARCH_PATH, BASE, OP_ITEM, OP_VAULT,
    _BACKOFF_BASE, _BACKOFF_FACTOR, _BACKOFF_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def api_key() -> str:
    """Retrieve the twitterapi.io API key from 1Password (never hardcoded)."""
    result = subprocess.run(
        ["op", "item", "get", OP_ITEM, "--vault", OP_VAULT,
         "--fields", "credential", "--reveal"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


class RateLimiter:
    """Paces async requests to at most `per_second` issuances, shared across callers."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
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


@dataclass
class SearchPage:
    tweets: list[dict[str, Any]]
    has_next_page: bool
    next_cursor: str


def _is_rate_limited(status_code: int, payload: dict[str, Any]) -> bool:
    if status_code == 429 or status_code >= 500:
        return True
    msg = str(payload.get("msg", "")).lower()
    return "rate" in msg or "too many requests" in msg


class TwitterAPI:
    """Async twitterapi.io client."""

    def __init__(self, http: httpx.AsyncClient, limiter: RateLimiter,
                 semaphore: asyncio.Semaphore) -> None:
        self._http = http
        self._limiter = limiter
        self._semaphore = semaphore

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE}{path}"
        for attempt in range(_BACKOFF_MAX_RETRIES + 1):
            await self._limiter.acquire()
            async with self._semaphore:
                resp = await self._http.get(url, params=params)
                payload: dict[str, Any] = resp.json()

            if _is_rate_limited(resp.status_code, payload):
                if attempt < _BACKOFF_MAX_RETRIES:
                    sleep_secs = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                    logger.warning("Rate/5xx on %s (attempt %d/%d), sleeping %.1fs",
                                   path, attempt + 1, _BACKOFF_MAX_RETRIES, sleep_secs)
                    await asyncio.sleep(sleep_secs)
                    continue
                raise RuntimeError(f"Rate limited after {_BACKOFF_MAX_RETRIES} retries: {payload}")

            if payload.get("status") == "error":
                raise RuntimeError(f"twitterapi.io error: {payload.get('msg', payload)}")

            return payload
        raise RuntimeError("Unexpected: exhausted retry loop")  # pragma: no cover

    async def advanced_search(self, query: str, cursor: str = "") -> SearchPage:
        """One page of advanced_search results (queryType=Latest, newest→oldest)."""
        payload = await self._get(ADVANCED_SEARCH_PATH, params={
            "query": query, "queryType": "Latest", "cursor": cursor,
        })
        return SearchPage(
            tweets=payload.get("tweets", []),
            has_next_page=bool(payload.get("has_next_page", False)),
            next_cursor=payload.get("next_cursor", "") or "",
        )
