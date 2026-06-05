"""Fetch logic against a mocked scrape.do transport (no network)."""
import urllib.parse

import httpx

from scrapers.linkedin.fetch import fetch_profile, scrapedo_url


def test_scrapedo_url_public_page_params():
    url = scrapedo_url("TOK", "https://www.linkedin.com/in/ada/")
    assert url.startswith("https://api.scrape.do/?token=TOK&url=")
    assert "super=true" in url          # residential -> clears Cloudflare
    assert "geoCode=us" in url
    assert "render=true" not in url     # public page is SSR; no JS render
    assert "setCookies" not in url      # logged-out: we want the public template
    assert urllib.parse.quote("https://www.linkedin.com/in/ada/", safe="") in url


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_success_returns_html():
    def handler(request):
        return httpx.Response(200, text="<html>hi</html>")

    async with _client(handler) as client:
        result = await fetch_profile(client, "TOK", "ada")
    assert result.status == 200
    assert result.html == "<html>hi</html>"
    assert result.error is None


async def test_fetch_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setattr("scrapers.linkedin.fetch.asyncio.sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text="<html>ok</html>")

    async with _client(handler) as client:
        result = await fetch_profile(client, "TOK", "ada")
    assert calls["n"] == 2
    assert result.status == 200


async def test_fetch_non_200_returns_error(monkeypatch):
    monkeypatch.setattr("scrapers.linkedin.fetch.asyncio.sleep", _no_sleep)

    def handler(request):
        return httpx.Response(404, text="nope")

    async with _client(handler) as client:
        result = await fetch_profile(client, "TOK", "ghost")
    assert result.status == 404
    assert result.html == ""
    assert result.error is not None


async def _no_sleep(_seconds):
    return None
