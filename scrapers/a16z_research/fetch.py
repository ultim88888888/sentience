"""Fetch article HTML through scrape.do at high concurrency."""
import asyncio
import subprocess
import urllib.parse
from dataclasses import dataclass

import httpx

from .config import (CONCURRENCY, FETCH_RETRIES, FETCH_TIMEOUT, SCRAPEDO_BASE,
                     SCRAPEDO_OP_ITEM, SCRAPEDO_OP_VAULT, UA)


def scrapedo_token() -> str:
    """Read the scrape.do API token from 1Password (vault 'local')."""
    return subprocess.check_output(
        ["op", "item", "get", SCRAPEDO_OP_ITEM, "--vault", SCRAPEDO_OP_VAULT,
         "--fields", "credential", "--reveal"], text=True).strip()


@dataclass
class FetchResult:
    permalink: str       # original metadata permalink (join key)
    status: int          # upstream HTTP status (origin), or 0 on hard failure
    html: str
    error: str | None = None
    fetched_url: str | None = None  # URL actually fetched (may differ via host fallback)


def _scrapedo_url(token: str, target: str) -> str:
    q = urllib.parse.quote(target, safe="")
    # Article pages are server-rendered; no render=true needed. customHeaders forwards UA.
    return f"{SCRAPEDO_BASE}?token={token}&url={q}&customHeaders=true"


def _alt_host(url: str) -> str | None:
    """Some posts exist only on the public host, others only on the cms backend host.
    Return the same path on the other host, or None if neither host applies."""
    if "://a16zcrypto.com/" in url:
        return url.replace("://a16zcrypto.com/", "://cms.a16zcrypto.com/", 1)
    if "://cms.a16zcrypto.com/" in url:
        return url.replace("://cms.a16zcrypto.com/", "://a16zcrypto.com/", 1)
    return None


async def _get(client, token, url):
    """One scrape.do GET with transient-retry. Returns (status, text, err)."""
    last_err = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            r = await client.get(_scrapedo_url(token, url),
                                 headers={"User-Agent": UA}, timeout=FETCH_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < FETCH_RETRIES:
                last_err = f"scrapedo HTTP {r.status_code}"
                await asyncio.sleep(2 * attempt)
                continue
            return r.status_code, r.text or "", (None if r.status_code == 200
                                                 else f"scrapedo HTTP {r.status_code}")
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = repr(e)
            if attempt < FETCH_RETRIES:
                await asyncio.sleep(2 * attempt)
    return 0, "", last_err


async def _fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                     token: str, permalink: str) -> FetchResult:
    async with sem:
        status, html, err = await _get(client, token, permalink)
        if status == 200:
            return FetchResult(permalink, 200, html, fetched_url=permalink)
        # On 404, the post may live only on the other host (cms <-> public) -> try alt.
        if status == 404:
            alt = _alt_host(permalink)
            if alt:
                a_status, a_html, a_err = await _get(client, token, alt)
                if a_status == 200:
                    return FetchResult(permalink, 200, a_html, fetched_url=alt)
        return FetchResult(permalink, status, html, error=err, fetched_url=permalink)


async def fetch_all(permalinks: list[str], token: str | None = None,
                    progress=None) -> list[FetchResult]:
    token = token or scrapedo_token()
    sem = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10,
                          max_keepalive_connections=CONCURRENCY)
    results: list[FetchResult] = []
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        tasks = [asyncio.create_task(_fetch_one(client, sem, token, u)) for u in permalinks]
        done = 0
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
            done += 1
            if progress:
                progress(done, len(permalinks))
    return results
