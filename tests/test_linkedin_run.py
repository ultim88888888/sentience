"""slug normalization + orchestration smoke (fetch/auth mocked, tmp output dirs)."""
import asyncio

from scrapers.linkedin.auth import Auth
from scrapers.linkedin.fetch import FetchResult


def test_normalize_slug_from_url():
    from scrapers.linkedin.run import normalize_slug
    assert normalize_slug("https://www.linkedin.com/in/ada-lovelace/") == "ada-lovelace"
    assert normalize_slug("https://linkedin.com/in/grace?foo=bar") == "grace"
    assert normalize_slug("ada-lovelace") == "ada-lovelace"


def test_read_slugs_single_value_when_not_a_file():
    from scrapers.linkedin.run import read_slugs
    assert read_slugs("ada-lovelace") == ["ada-lovelace"]


def test_read_slugs_from_file(tmp_path):
    from scrapers.linkedin.run import read_slugs
    f = tmp_path / "people.txt"
    f.write_text("ada-lovelace\nhttps://www.linkedin.com/in/grace/\n\n")
    assert read_slugs(str(f)) == ["ada-lovelace", "grace"]


def test_main_writes_raw_and_parsed(tmp_path, monkeypatch):
    import scrapers.linkedin.run as run

    raw, parsed = tmp_path / "raw", tmp_path / "parsed"
    monkeypatch.setattr(run, "RAW_DIR", raw)
    monkeypatch.setattr(run, "PARSED_DIR", parsed)
    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)

    payload = {"profile": {"firstName": "Ada", "lastName": "Lovelace",
                           "summary": "Pioneer."}}

    async def fake_fetch(client, auth, slug):
        return FetchResult(slug, 200, payload=payload)

    monkeypatch.setattr(run, "load_auth",
                        lambda: Auth("TOK", "LIAT", "ajax:1"))
    monkeypatch.setattr(run, "fetch_profile", fake_fetch)

    asyncio.run(run.main(["ada-lovelace"]))

    assert (raw / "ada-lovelace.json").exists()
    parsed_file = parsed / "ada-lovelace.json"
    assert parsed_file.exists()
    assert "Ada Lovelace" in parsed_file.read_text()


def test_main_skips_failed_profile_without_aborting(tmp_path, monkeypatch):
    import scrapers.linkedin.run as run

    raw, parsed = tmp_path / "raw", tmp_path / "parsed"
    monkeypatch.setattr(run, "RAW_DIR", raw)
    monkeypatch.setattr(run, "PARSED_DIR", parsed)
    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(run, "load_auth", lambda: Auth("T", "L", "j"))

    async def fake_fetch(client, auth, slug):
        if slug == "ghost":
            return FetchResult(slug, 404, error="HTTP 404")
        return FetchResult(slug, 200, payload={"profile": {"firstName": "Ada"}})

    monkeypatch.setattr(run, "fetch_profile", fake_fetch)

    asyncio.run(run.main(["ghost", "ada"]))

    assert not (parsed / "ghost.json").exists()   # failed one skipped
    assert (parsed / "ada.json").exists()          # batch continued


async def _no_sleep(_seconds):
    return None
