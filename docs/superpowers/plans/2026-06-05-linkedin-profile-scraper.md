# LinkedIn Profile Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-volume client that scrapes structured work experience, education, and bio for a hand-supplied list of LinkedIn people, via LinkedIn's Voyager API tunneled through scrape.do with an authenticated session.

**Architecture:** Sequential client. For each profile slug, build an authenticated Voyager `profileView` request, send it through scrape.do (residential + sticky session + geo to dodge LinkedIn's new-location lockout), save the raw JSON, then parse it into a pydantic `Profile` and save that too. Secrets (scrape.do token, `li_at`, `JSESSIONID`) come from 1Password via `op`. Raw-JSON-first so a parser change never forces a re-scrape.

**Tech Stack:** Python, httpx (async client, used sequentially), pydantic, pytest (`asyncio_mode=auto`). Run tests with `.venv/bin/python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scrapers/linkedin/__init__.py` | Package marker. |
| `scrapers/linkedin/config.py` | scrape.do + Voyager constants, sticky-session/geo, `op` item names, paths, delays. |
| `scrapers/linkedin/models.py` | pydantic `Profile`, `Experience`, `Education`. |
| `scrapers/linkedin/parse.py` | Voyager `profileView` JSON → `Profile`. |
| `scrapers/linkedin/auth.py` | Read secrets from 1Password; build Voyager request headers. |
| `scrapers/linkedin/fetch.py` | Build scrape.do URL, GET with retry + auth-expiry detection. |
| `scrapers/linkedin/run.py` | CLI: read slug list → fetch → save raw → parse → save parsed. |
| `scrapers/linkedin/requirements.txt` | `httpx`, `pydantic`. |
| `scrapers/linkedin/README.md` | Usage + cookie/secret setup + refresh steps. |
| `tests/test_linkedin_models.py` | Model defaults. |
| `tests/test_linkedin_parse.py` | Parser against an inline Voyager fixture + edge cases. |
| `tests/test_linkedin_auth.py` | Header/cookie construction (subprocess mocked). |
| `tests/test_linkedin_fetch.py` | URL construction, success, auth-expiry, retry (httpx MockTransport). |
| `tests/test_linkedin_run.py` | slug normalization + orchestration smoke (mocked fetch/auth). |

---

### Task 1: Package scaffold + config

**Files:**
- Create: `scrapers/linkedin/__init__.py`
- Create: `scrapers/linkedin/requirements.txt`
- Create: `scrapers/linkedin/config.py`

- [ ] **Step 1: Create the package marker**

`scrapers/linkedin/__init__.py`:

```python
"""LinkedIn profile scraper — authenticated Voyager API via scrape.do."""
```

- [ ] **Step 2: Create requirements.txt**

`scrapers/linkedin/requirements.txt`:

```
httpx
pydantic
```

- [ ] **Step 3: Create config.py**

`scrapers/linkedin/config.py`:

```python
"""Configuration for the LinkedIn profile scraper."""
from pathlib import Path

# scrape.do transport
SCRAPEDO_BASE = "https://api.scrape.do/"
SCRAPEDO_OP_ITEM = "scrape.do"
OP_VAULT = "local"

# LinkedIn auth secrets — 1Password items in vault `local` (Jax creates these).
LI_AT_OP_ITEM = "linkedin_li_at"
JSESSIONID_OP_ITEM = "linkedin_jsessionid"

# Voyager API — LinkedIn's own internal JSON endpoint for full profile data.
VOYAGER_PROFILE_VIEW = (
    "https://www.linkedin.com/voyager/api/identity/profiles/{slug}/profileView"
)

# Session-security guard: a fixed sticky session pins ONE residential IP for the
# whole run, geo-matched to where the account normally logs in. This — not request
# volume — is what trips LinkedIn's "sign-in from a new location" lockout.
SCRAPEDO_SESSION_ID = "778899"   # any fixed value -> same upstream IP across the run
SCRAPEDO_GEO = "us"               # match the account's usual login region

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")

# Politeness / resilience
REQUEST_DELAY = 4.0    # seconds between profiles
FETCH_RETRIES = 3
FETCH_TIMEOUT = 60.0

# Output
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "linkedin"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
```

- [ ] **Step 4: Verify config imports cleanly**

Run: `.venv/bin/python -c "from scrapers.linkedin import config; print(config.RAW_DIR)"`
Expected: prints a path ending in `data/linkedin/raw`.

- [ ] **Step 5: Commit**

```bash
git add scrapers/linkedin/__init__.py scrapers/linkedin/requirements.txt scrapers/linkedin/config.py
git commit -m "feat(linkedin): package scaffold + config"
```

---

### Task 2: Profile models

**Files:**
- Create: `scrapers/linkedin/models.py`
- Test: `tests/test_linkedin_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_linkedin_models.py`:

```python
"""Schema defaults for the LinkedIn profile models — pure, no network."""
from scrapers.linkedin.models import Education, Experience, Profile


def test_experience_defaults_all_optional():
    exp = Experience()
    assert exp.title is None
    assert exp.company is None
    assert exp.start is None
    assert exp.end is None
    assert exp.description is None


def test_education_defaults_all_optional():
    edu = Education()
    assert edu.school is None
    assert edu.degree is None
    assert edu.field is None


def test_profile_requires_slug_and_defaults_lists():
    p = Profile(slug="ada-lovelace")
    assert p.slug == "ada-lovelace"
    assert p.name is None
    assert p.experience == []
    assert p.education == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linkedin_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scrapers.linkedin.models'`.

- [ ] **Step 3: Write the implementation**

`scrapers/linkedin/models.py`:

```python
"""Structured LinkedIn profile schema."""
from __future__ import annotations

from pydantic import BaseModel


class Experience(BaseModel):
    title: str | None = None
    company: str | None = None
    start: str | None = None        # "YYYY-MM" or "YYYY"
    end: str | None = None          # None == present
    description: str | None = None


class Education(BaseModel):
    school: str | None = None
    degree: str | None = None
    field: str | None = None
    start: str | None = None
    end: str | None = None


class Profile(BaseModel):
    slug: str
    name: str | None = None
    headline: str | None = None
    location: str | None = None
    bio: str | None = None
    experience: list[Experience] = []
    education: list[Education] = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_linkedin_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scrapers/linkedin/models.py tests/test_linkedin_models.py
git commit -m "feat(linkedin): profile/experience/education models"
```

---

### Task 3: Parser (the core, and the brittle part)

**Files:**
- Create: `scrapers/linkedin/parse.py`
- Test: `tests/test_linkedin_parse.py`

- [ ] **Step 1: Write the failing test (with inline Voyager fixture)**

`tests/test_linkedin_parse.py`:

```python
"""Parse Voyager profileView JSON -> Profile. Pure, no network.

The fixture mirrors the real profileView decoration: top-level `profile`,
`positionView.elements`, `educationView.elements`, with Voyager date shapes.
"""
from scrapers.linkedin.parse import parse_profile

FIXTURE = {
    "profile": {
        "firstName": "Ada",
        "lastName": "Lovelace",
        "headline": "Mathematician | First Programmer",
        "locationName": "London, England, United Kingdom",
        "summary": "Pioneer of computing.",
    },
    "positionView": {
        "elements": [
            {
                "title": "Analyst",
                "companyName": "Analytical Engine Project",
                "description": "Wrote the first algorithm.",
                "timePeriod": {
                    "startDate": {"month": 6, "year": 1842},
                    "endDate": {"month": 8, "year": 1843},
                },
            },
            {
                "title": "Correspondent",
                "companyName": "Royal Society",
                "timePeriod": {"startDate": {"year": 1840}},
            },
        ]
    },
    "educationView": {
        "elements": [
            {
                "schoolName": "Self-taught",
                "degreeName": "Mathematics",
                "fieldOfStudy": "Analysis",
                "timePeriod": {
                    "startDate": {"year": 1832},
                    "endDate": {"year": 1835},
                },
            }
        ]
    },
}


def test_parse_identity_and_bio():
    p = parse_profile("ada-lovelace", FIXTURE)
    assert p.slug == "ada-lovelace"
    assert p.name == "Ada Lovelace"
    assert p.headline == "Mathematician | First Programmer"
    assert p.location == "London, England, United Kingdom"
    assert p.bio == "Pioneer of computing."


def test_parse_experience_with_date_formats():
    p = parse_profile("ada-lovelace", FIXTURE)
    assert len(p.experience) == 2
    first = p.experience[0]
    assert first.title == "Analyst"
    assert first.company == "Analytical Engine Project"
    assert first.start == "1842-06"
    assert first.end == "1843-08"
    assert first.description == "Wrote the first algorithm."
    # year-only start, no end -> present
    second = p.experience[1]
    assert second.start == "1840"
    assert second.end is None


def test_parse_education():
    p = parse_profile("ada-lovelace", FIXTURE)
    assert len(p.education) == 1
    edu = p.education[0]
    assert edu.school == "Self-taught"
    assert edu.degree == "Mathematics"
    assert edu.field == "Analysis"
    assert edu.start == "1832"
    assert edu.end == "1835"


def test_parse_missing_summary_is_none():
    payload = {"profile": {"firstName": "Grace", "lastName": "Hopper"}}
    p = parse_profile("grace-hopper", payload)
    assert p.name == "Grace Hopper"
    assert p.bio is None
    assert p.experience == []
    assert p.education == []


def test_parse_empty_payload():
    p = parse_profile("nobody", {})
    assert p.slug == "nobody"
    assert p.name is None
    assert p.experience == []
    assert p.education == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linkedin_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scrapers.linkedin.parse'`.

- [ ] **Step 3: Write the implementation**

`scrapers/linkedin/parse.py`:

```python
"""Parse a Voyager profileView payload into a structured Profile."""
from __future__ import annotations

from .models import Education, Experience, Profile


def _fmt_date(date: dict | None) -> str | None:
    """Voyager date {month, year} -> 'YYYY-MM' or 'YYYY'. None if no year."""
    if not date:
        return None
    year = date.get("year")
    if not year:
        return None
    month = date.get("month")
    return f"{year:04d}-{month:02d}" if month else f"{year:04d}"


def _period(node: dict) -> tuple[str | None, str | None]:
    tp = node.get("timePeriod") or {}
    return _fmt_date(tp.get("startDate")), _fmt_date(tp.get("endDate"))


def parse_profile(slug: str, payload: dict) -> Profile:
    profile = payload.get("profile") or {}
    name = " ".join(
        p for p in (profile.get("firstName"), profile.get("lastName")) if p
    ) or None

    experience: list[Experience] = []
    for el in (payload.get("positionView") or {}).get("elements", []):
        start, end = _period(el)
        experience.append(Experience(
            title=el.get("title"),
            company=el.get("companyName"),
            start=start,
            end=end,
            description=el.get("description"),
        ))

    education: list[Education] = []
    for el in (payload.get("educationView") or {}).get("elements", []):
        start, end = _period(el)
        education.append(Education(
            school=el.get("schoolName"),
            degree=el.get("degreeName"),
            field=el.get("fieldOfStudy"),
            start=start,
            end=end,
        ))

    return Profile(
        slug=slug,
        name=name,
        headline=profile.get("headline"),
        location=profile.get("locationName"),
        bio=profile.get("summary"),
        experience=experience,
        education=education,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_linkedin_parse.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scrapers/linkedin/parse.py tests/test_linkedin_parse.py
git commit -m "feat(linkedin): parse Voyager profileView into Profile"
```

---

### Task 4: Auth — secrets + request headers

**Files:**
- Create: `scrapers/linkedin/auth.py`
- Test: `tests/test_linkedin_auth.py`

- [ ] **Step 1: Write the failing test**

`tests/test_linkedin_auth.py`:

```python
"""Auth header/cookie construction. 1Password (`op`) subprocess is mocked."""
from unittest.mock import patch

from scrapers.linkedin.auth import Auth, load_auth


def test_voyager_headers_carry_cookie_and_csrf():
    auth = Auth(scrapedo_token="TOK", li_at="LIAT", jsessionid="ajax:123")
    h = auth.voyager_headers()
    assert h["Csrf-Token"] == "ajax:123"
    assert h["Cookie"] == 'li_at=LIAT; JSESSIONID="ajax:123"'
    assert h["X-RestLi-Protocol-Version"] == "2.0.0"
    assert "User-Agent" in h


def test_load_auth_reads_three_secrets_and_strips_jsessionid_quotes():
    # `op` returns each item's credential in order; JSESSIONID may come quoted.
    with patch("scrapers.linkedin.auth.subprocess.check_output",
               side_effect=["TOK\n", "LIAT\n", '"ajax:123"\n']) as m:
        auth = load_auth()
    assert auth.scrapedo_token == "TOK"
    assert auth.li_at == "LIAT"
    assert auth.jsessionid == "ajax:123"   # surrounding quotes stripped
    assert m.call_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linkedin_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scrapers.linkedin.auth'`.

- [ ] **Step 3: Write the implementation**

`scrapers/linkedin/auth.py`:

```python
"""Read LinkedIn auth secrets from 1Password and build Voyager request headers."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .config import (JSESSIONID_OP_ITEM, LI_AT_OP_ITEM, OP_VAULT,
                     SCRAPEDO_OP_ITEM, UA)


def _op_read(item: str) -> str:
    return subprocess.check_output(
        ["op", "item", "get", item, "--vault", OP_VAULT,
         "--fields", "credential", "--reveal"],
        text=True,
    ).strip()


@dataclass
class Auth:
    scrapedo_token: str
    li_at: str
    jsessionid: str   # raw value, no surrounding quotes

    def voyager_headers(self) -> dict[str, str]:
        # scrape.do forwards these upstream because we pass customHeaders=true.
        # LinkedIn requires Csrf-Token == the JSESSIONID cookie value.
        return {
            "User-Agent": UA,
            "Accept": "application/json",
            "Csrf-Token": self.jsessionid,
            "X-RestLi-Protocol-Version": "2.0.0",
            "Cookie": f'li_at={self.li_at}; JSESSIONID="{self.jsessionid}"',
        }


def load_auth() -> Auth:
    return Auth(
        scrapedo_token=_op_read(SCRAPEDO_OP_ITEM),
        li_at=_op_read(LI_AT_OP_ITEM),
        jsessionid=_op_read(JSESSIONID_OP_ITEM).strip('"'),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_linkedin_auth.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scrapers/linkedin/auth.py tests/test_linkedin_auth.py
git commit -m "feat(linkedin): 1Password secrets + Voyager auth headers"
```

---

### Task 5: Fetch — scrape.do URL, retry, auth-expiry detection

**Files:**
- Create: `scrapers/linkedin/fetch.py`
- Test: `tests/test_linkedin_fetch.py`

- [ ] **Step 1: Write the failing test**

`tests/test_linkedin_fetch.py`:

```python
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
    monkeypatch.setattr("scrapers.linkedin.fetch.asyncio.sleep",
                        _no_sleep)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linkedin_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scrapers.linkedin.fetch'`.

- [ ] **Step 3: Write the implementation**

`scrapers/linkedin/fetch.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_linkedin_fetch.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scrapers/linkedin/fetch.py tests/test_linkedin_fetch.py
git commit -m "feat(linkedin): scrape.do fetch with retry + auth-expiry detection"
```

---

### Task 6: Run — CLI orchestration

**Files:**
- Create: `scrapers/linkedin/run.py`
- Test: `tests/test_linkedin_run.py`

- [ ] **Step 1: Write the failing test**

`tests/test_linkedin_run.py`:

```python
"""slug normalization + orchestration smoke (fetch/auth mocked, tmp output dirs)."""
import asyncio
from unittest.mock import patch

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linkedin_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scrapers.linkedin.run'`.

- [ ] **Step 3: Write the implementation**

`scrapers/linkedin/run.py`:

```python
"""CLI: scrape a list of LinkedIn profiles to data/linkedin/{raw,parsed}/."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

from .auth import load_auth
from .config import PARSED_DIR, RAW_DIR, REQUEST_DELAY
from .fetch import AuthExpiredError, fetch_profile
from .parse import parse_profile


def normalize_slug(value: str) -> str:
    """A vanity slug or a full /in/ URL -> bare slug."""
    value = value.strip().rstrip("/")
    if "linkedin.com/in/" in value:
        value = value.split("linkedin.com/in/", 1)[1].split("/")[0].split("?")[0]
    return value


def read_slugs(arg: str) -> list[str]:
    """arg is a file path (one slug/URL per line) or a single slug/URL."""
    p = Path(arg)
    lines = p.read_text().splitlines() if p.exists() else [arg]
    return [normalize_slug(x) for x in lines if x.strip()]


async def main(slugs: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    auth = load_auth()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for i, slug in enumerate(slugs, 1):
            print(f"[{i}/{len(slugs)}] {slug}", file=sys.stderr)
            try:
                result = await fetch_profile(client, auth, slug)
            except AuthExpiredError as e:
                print(f"FATAL: {e}", file=sys.stderr)
                raise SystemExit(2)
            if result.status != 200 or result.payload is None:
                print(f"  skip ({result.error})", file=sys.stderr)
                continue
            (RAW_DIR / f"{slug}.json").write_text(json.dumps(result.payload, indent=2))
            profile = parse_profile(slug, result.payload)
            (PARSED_DIR / f"{slug}.json").write_text(profile.model_dump_json(indent=2))
            print(f"  ok ({len(profile.experience)} exp, {len(profile.education)} edu)",
                  file=sys.stderr)
            if i < len(slugs):
                await asyncio.sleep(REQUEST_DELAY)


def cli() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m scrapers.linkedin.run <slug|url|file>", file=sys.stderr)
        raise SystemExit(1)
    asyncio.run(main(read_slugs(sys.argv[1])))


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_linkedin_run.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the full linkedin suite**

Run: `.venv/bin/python -m pytest tests/test_linkedin_*.py -v`
Expected: all linkedin tests pass (21 total: 3 models + 5 parse + 2 auth + 6 fetch + 5 run).

- [ ] **Step 6: Commit**

```bash
git add scrapers/linkedin/run.py tests/test_linkedin_run.py
git commit -m "feat(linkedin): CLI orchestration (fetch->save raw->parse->save)"
```

---

### Task 7: README + data dir + gitignore check

**Files:**
- Create: `scrapers/linkedin/README.md`
- Create: `data/linkedin/.gitkeep` (or confirm output dir handling)

- [ ] **Step 1: Decide whether scraped profile data is committed**

Check `.gitignore` for any `data/` rule:

Run: `grep -n "data" .gitignore || echo "no data ignore rule"`

LinkedIn profile data is personal and should NOT be committed. Add an ignore rule
for the scraped output while keeping the directory tracked:

Append to `.gitignore`:

```
# LinkedIn scraped profile data (personal data — do not commit)
data/linkedin/raw/
data/linkedin/parsed/
```

- [ ] **Step 2: Write the README**

`scrapers/linkedin/README.md`:

```markdown
# LinkedIn Profile Scraper

Low-volume client (<20 profiles) that pulls work experience, education, and bio
for a supplied list of people, via LinkedIn's Voyager API through scrape.do with
an authenticated session.

## Secrets (1Password vault `local`)

Create two items (the scrape.do token item `scrape.do` already exists):

- `linkedin_li_at` — the `li_at` cookie value.
- `linkedin_jsessionid` — the `JSESSIONID` cookie value (looks like `ajax:12345...`).

### Getting the cookies

1. Log into linkedin.com in a browser (use a burner account — LinkedIn bans
   automation).
2. Open DevTools → Application → Cookies → `https://www.linkedin.com`.
3. Copy the `li_at` value → store as `linkedin_li_at` (field `credential`).
4. Copy the `JSESSIONID` value (strip the surrounding quotes) → store as
   `linkedin_jsessionid` (field `credential`).

Cookies expire periodically. When the run fails with an auth-expiry error,
re-copy both and update the 1Password items.

## Run

```bash
# single profile (slug or full URL)
.venv/bin/python -m scrapers.linkedin.run ada-lovelace
.venv/bin/python -m scrapers.linkedin.run https://www.linkedin.com/in/ada-lovelace/

# a list, one slug/URL per line
.venv/bin/python -m scrapers.linkedin.run people.txt
```

Output:

- `data/linkedin/raw/{slug}.json` — untouched Voyager response (durable; re-parse
  from here, never re-scrape).
- `data/linkedin/parsed/{slug}.json` — structured profile.

## Session-security note

The run pins ONE residential IP (`sessionId` in `config.py`) geo-matched to the
account (`SCRAPEDO_GEO`). This avoids LinkedIn's "new location" lockout — the real
risk, not request volume. If the account normally logs in outside the US, change
`SCRAPEDO_GEO`.
```

- [ ] **Step 3: Commit**

```bash
git add scrapers/linkedin/README.md .gitignore
git commit -m "docs(linkedin): README + ignore scraped personal data"
```

---

## Self-Review

- **Spec coverage:** Voyager approach (Task 5), auth + secrets + session guard (Tasks 1, 4, 5), module layout (all tasks match the spec table), data flow + JSON-only output (Task 6), extraction targets (Tasks 2–3), per-profile isolation + auth-expiry-loud (Tasks 5–6), fixture-based tests no live calls (Tasks 3–6). No parquet index (correctly omitted — deferred). Covered.
- **Type consistency:** `Auth(scrapedo_token, li_at, jsessionid)`, `FetchResult(slug, status, payload, error)`, `parse_profile(slug, payload)`, `Profile`/`Experience`/`Education` field names — consistent across auth/fetch/run/parse tasks and their tests.
- **Placeholder scan:** none — every step carries full code and exact commands.

---

## Manual end-to-end (after secrets exist, not part of TDD)

Once Jax has created the two 1Password items and supplied a real slug:

```bash
.venv/bin/python -m scrapers.linkedin.run <real-slug>
cat data/linkedin/parsed/<real-slug>.json
```

Confirm experience/education/bio populate. This is the only live-network check and
validates the Voyager payload shape against the parser; if LinkedIn's shape differs
from the fixture, adjust `parse.py` (the raw JSON is saved, so no re-scrape needed).
