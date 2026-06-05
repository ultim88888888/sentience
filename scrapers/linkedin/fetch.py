"""Fetch profiles from LinkedIn Voyager through scrape.do (sequential, low-volume)."""
from __future__ import annotations

import asyncio
import json
import urllib.parse
from dataclasses import dataclass

import httpx

from .auth import Auth
from .config import (FETCH_RETRIES, FETCH_TIMEOUT, SCRAPEDO_BASE, SCRAPEDO_GEO,
                     SCRAPEDO_SESSION_ID, VOYAGER_PROFILE_VIEW)


class AuthExpiredError(RuntimeError):
    """LinkedIn rejected the session — cookies need refreshing. Fatal for the run."""


@dataclass
class FetchResult:
    slug: str
    status: int
    payload: dict | None = None    # parsed Voyager JSON on success
    error: str | None = None


def scrapedo_url(token: str, target: str) -> str:
    q = urllib.parse.quote(target, safe="")
    # customHeaders=true forwards our auth headers; super=true = residential proxy;
    # sessionId pins one IP; geoCode matches the account's region.
    return (f"{SCRAPEDO_BASE}?token={token}&url={q}"
            f"&customHeaders=true&super=true"
            f"&geoCode={SCRAPEDO_GEO}&sessionId={SCRAPEDO_SESSION_ID}")


async def fetch_profile(client: httpx.AsyncClient, auth: Auth, slug: str) -> FetchResult:
    url = scrapedo_url(auth.scrapedo_token, VOYAGER_PROFILE_VIEW.format(slug=slug))
    headers = auth.voyager_headers()
    last_err = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            r = await client.get(url, headers=headers, timeout=FETCH_TIMEOUT)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = repr(e)
            if attempt < FETCH_RETRIES:
                await asyncio.sleep(2 * attempt)
            continue

        if r.status_code in (401, 403):
            raise AuthExpiredError(
                f"LinkedIn rejected the session (HTTP {r.status_code}) on '{slug}'. "
                "Refresh linkedin_li_at / linkedin_jsessionid in 1Password.")

        if r.status_code in (429, 500, 502, 503, 504) and attempt < FETCH_RETRIES:
            last_err = f"upstream HTTP {r.status_code}"
            await asyncio.sleep(2 * attempt)
            continue

        if r.status_code != 200:
            return FetchResult(slug, r.status_code, error=f"HTTP {r.status_code}")

        try:
            return FetchResult(slug, 200, payload=r.json())
        except json.JSONDecodeError:
            # A 200 that isn't JSON is a login wall served to a dead session.
            raise AuthExpiredError(
                f"Got non-JSON 200 for '{slug}' (likely a login wall). "
                "Refresh linkedin_li_at / linkedin_jsessionid in 1Password.")

    return FetchResult(slug, 0, error=last_err)
