"""slug normalization + orchestration smoke (fetch mocked, tmp output dirs)."""
import asyncio

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


_PUBLIC_HTML = """<html><head>
<script type="application/ld+json">
{"@type":"Person","name":"Ada Lovelace","description":"Analyst.",
 "alumniOf":[{"@type":"EducationalOrganization","name":"Home","member":{"startDate":1832}}]}
</script></head><body></body></html>"""


def _wire(monkeypatch, tmp_path, fetch_impl):
    import scrapers.linkedin.run as run
    monkeypatch.setattr(run, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(run, "PARSED_DIR", tmp_path / "parsed")
    monkeypatch.setattr(run, "RESTRICTED_LIST", tmp_path / "_restricted.txt")
    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(run, "scrapedo_token", lambda: "TOK")
    monkeypatch.setattr(run, "fetch_profile", fetch_impl)
    return run


def test_main_writes_raw_html_and_parsed_json(tmp_path, monkeypatch):
    async def fake_fetch(client, token, slug):
        return FetchResult(slug, 200, html=_PUBLIC_HTML)

    run = _wire(monkeypatch, tmp_path, fake_fetch)
    asyncio.run(run.main(["ada-lovelace"]))

    assert (tmp_path / "raw" / "ada-lovelace.html").exists()
    parsed = tmp_path / "parsed" / "ada-lovelace.json"
    assert parsed.exists()
    assert "Ada Lovelace" in parsed.read_text()
    assert not (tmp_path / "_restricted.txt").exists()   # full profile, not flagged


def test_main_flags_thin_and_failed_profiles(tmp_path, monkeypatch):
    async def fake_fetch(client, token, slug):
        if slug == "ghost":          # fetch failure
            return FetchResult(slug, 404, error="HTTP 404")
        if slug == "private":        # 200 but restricted -> thin parse
            return FetchResult(slug, 200,
                               html="<html><head><title>Sign Up | LinkedIn</title></head><body></body></html>")
        return FetchResult(slug, 200, html=_PUBLIC_HTML)

    run = _wire(monkeypatch, tmp_path, fake_fetch)
    asyncio.run(run.main(["ghost", "private", "ada"]))

    assert (tmp_path / "parsed" / "ada.json").exists()       # full one written
    restricted = (tmp_path / "_restricted.txt").read_text().split()
    assert "ghost" in restricted and "private" in restricted
    assert "ada" not in restricted


async def _no_sleep(_seconds):
    return None
