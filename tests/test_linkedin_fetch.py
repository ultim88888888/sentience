"""Fetch logic against a mocked scrape.do/Voyager transport (no network)."""
import urllib.parse

import httpx
import pytest

from scrapers.linkedin.auth import Auth
from scrapers.linkedin.fetch import (AuthExpiredError, fetch_profile,
                                     scrapedo_url)

AUTH = Auth(scrapedo_token="TOK", li_at="LIAT", jsessionid="ajax:123")


def test_scrapedo_url_has_all_guard_params():
    url = scrapedo_url("TOK", "https://www.linkedin.com/voyager/x")
    assert url.startswith("https://api.scrape.do/?token=TOK&url=")
    assert "customHeaders=true" in url
    assert "super=true" in url
    assert "geoCode=us" in url
    assert "sessionId=778899" in url
    # target must be percent-encoded
    assert urllib.parse.quote("https://www.linkedin.com/voyager/x", safe="") in url


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_success_returns_payload():
    def handler(request):
        return httpx.Response(200, json={"profile": {"firstName": "Ada"}})

    async with _client(handler) as client:
        result = await fetch_profile(client, AUTH, "ada")
    assert result.status == 200
    assert result.payload == {"profile": {"firstName": "Ada"}}
    assert result.error is None


async def test_fetch_401_raises_auth_expired():
    def handler(request):
        return httpx.Response(401, text="unauthorized")

    async with _client(handler) as client:
        with pytest.raises(AuthExpiredError):
            await fetch_profile(client, AUTH, "ada")


async def test_fetch_non_json_login_wall_raises_auth_expired():
    def handler(request):
        return httpx.Response(200, text="<html>Sign in</html>")

    async with _client(handler) as client:
        with pytest.raises(AuthExpiredError):
            await fetch_profile(client, AUTH, "ada")


async def test_fetch_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setattr("scrapers.linkedin.fetch.asyncio.sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"profile": {}})

    async with _client(handler) as client:
        result = await fetch_profile(client, AUTH, "ada")
    assert calls["n"] == 2
    assert result.status == 200


async def test_fetch_non_200_non_auth_returns_error_result(monkeypatch):
    monkeypatch.setattr("scrapers.linkedin.fetch.asyncio.sleep", _no_sleep)

    def handler(request):
        return httpx.Response(404, text="not found")

    async with _client(handler) as client:
        result = await fetch_profile(client, AUTH, "ghost")
    assert result.status == 404
    assert result.payload is None
    assert result.error is not None


async def _no_sleep(_seconds):
    return None
