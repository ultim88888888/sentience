"""Fetch public LinkedIn profile HTML through scrape.do (sequential, low-volume)."""
from __future__ import annotations

import asyncio
import subprocess
import urllib.parse
from dataclasses import dataclass

import httpx

from .config import (FETCH_RETRIES, FETCH_TIMEOUT, OP_VAULT, PROFILE_URL,
                     SCRAPEDO_BASE, SCRAPEDO_GEO, SCRAPEDO_OP_ITEM, UA)


@dataclass
class FetchResult:
    slug: str
    status: int
    html: str = ""
    error: str | None = None


def scrapedo_token() -> str:
    """Read the scrape.do API token from 1Password (vault `local`)."""
    return subprocess.check_output(
        ["op", "read", f"op://{OP_VAULT}/{SCRAPEDO_OP_ITEM}/credential"],
        text=True,
    ).strip()


def scrapedo_url(token: str, target: str) -> str:
    q = urllib.parse.quote(target, safe="")
    # super=true -> residential proxy (clears LinkedIn's Cloudflare WAF); geoCode
    # pins region. No render/cookies: we want the logged-out public SSR template.
    return f"{SCRAPEDO_BASE}?token={token}&url={q}&super=true&geoCode={SCRAPEDO_GEO}"


async def fetch_profile(client: httpx.AsyncClient, token: str, slug: str) -> FetchResult:
    url = scrapedo_url(token, PROFILE_URL.format(slug=slug))
    headers = {"User-Agent": UA}
    last_err = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            r = await client.get(url, headers=headers, timeout=FETCH_TIMEOUT)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = repr(e)
            if attempt < FETCH_RETRIES:
                await asyncio.sleep(2 * attempt)
            continue

        if r.status_code in (429, 500, 502, 503, 504) and attempt < FETCH_RETRIES:
            last_err = f"upstream HTTP {r.status_code}"
            await asyncio.sleep(2 * attempt)
            continue

        if r.status_code != 200:
            return FetchResult(slug, r.status_code, error=f"HTTP {r.status_code}")

        return FetchResult(slug, 200, html=r.text or "")

    return FetchResult(slug, 0, error=last_err)
