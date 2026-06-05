---
date: 2026-06-05
project: sentience
status: design
tags: [scraper, linkedin, scrape.do, voyager]
---

# LinkedIn Profile Scraper — Design

## Purpose

A low-volume client that scrapes structured profile data — **work experience,
education, and bio** — for a hand-supplied list of LinkedIn people (realistically
<20, capped well under 100). Feeds the `sentience` research corpus alongside the
a16z and market-data datasets.

This is a **simple sequential client**. Tiny volume removes any need for
concurrency, batching, or aggressive rate-limit machinery.

## Approach

**Authenticated Voyager API via scrape.do.**

LinkedIn gates all profile substance (full experience, education, bio) behind
authentication — logged-out scraping returns only a teaser plus an auth wall. So
we carry an authenticated session.

For extraction we call LinkedIn's own internal JSON API — **Voyager**, the same
endpoint the linkedin.com web app calls — rather than parsing rendered HTML:

```
GET https://www.linkedin.com/voyager/api/identity/profiles/{publicId}/profileView
```

It returns `positions` (experience), `educations`, and `summary` (bio) as clean
structured JSON. No HTML parsing, smaller payloads, maps 1:1 to the target fields.
Voyager is free (no API fee, no signup) but unofficial — same ToS gray zone as the
overall scrape.

scrape.do remains the proxy/transport layer; we feed it an authenticated request.

### Rejected alternatives

- **Rendered-HTML parse.** Fetch the profile page, extract data from embedded
  `<code>` JSON blocks. Same underlying data but brittle against constant layout
  changes, larger payloads, no upside at our scale.
- **LinkedIn-specialized API (Proxycurl etc.).** Clean structured JSON, sidesteps
  all auth/cookie handling — but it isn't scrape.do, and costs money per profile.
  Out of scope by request.

## Authentication & secrets

The client needs, from a logged-in (burner-recommended) LinkedIn account:

- `li_at` cookie — the auth session token.
- `JSESSIONID` cookie — also the value of the required `csrf-token` request header.

Both are read at runtime from **1Password vault `local`** via the `op` CLI (same
pattern as the existing scrape.do token). scrape.do token item is `scrape.do`;
new items `linkedin_li_at` and `linkedin_jsessionid` to be added by Jax.

Requests set `customHeaders=true` on scrape.do (which forwards our headers
upstream) and send:

```
Cookie: li_at=<li_at>; JSESSIONID="<jsessionid>"
Csrf-Token: <jsessionid>
```

plus the standard Voyager headers (`x-restli-protocol-version: 2.0.0`, UA, etc.).

### Session-security guard (the real risk)

The failure mode at this scale is **not** request volume — it's LinkedIn's
"sign-in from a new location" lockout, triggered by carrying an auth cookie across
*rotating* residential IPs. Mitigation, baked into config:

- **Sticky scrape.do session** (`super=true&sessionId=<fixed>`) so the entire run
  shares one stable residential IP.
- **Geo targeting** (`geoCode`) matched to where the account normally logs in.
- Sequential requests with a polite delay (a few seconds) between profiles.

## Module layout

`scrapers/linkedin/` — mirrors `scrapers/a16z_research/`:

| File | Responsibility |
|---|---|
| `config.py` | scrape.do base/params, sticky-session + geo settings, `op` item names, UA, Voyager endpoint template, paths, delay/retry/timeout. |
| `auth.py` | Read `li_at`, `JSESSIONID`, scrape.do token from 1Password; build the Voyager request headers/cookies. |
| `fetch.py` | `fetch_profile(slug) -> FetchResult`: build the scrape.do+Voyager URL, GET with transient retry, return raw JSON text + status. Sequential `fetch_all(slugs)`. Detect auth-expiry (401/challenge) and raise a clear "refresh cookies" error. |
| `parse.py` | Parse raw Voyager JSON into the structured `Profile` schema (pydantic). |
| `models.py` | pydantic models: `Profile`, `Experience`, `Education`. |
| `run.py` | CLI entry: read slug list (arg/file), fetch → save raw → parse → save parsed, per-profile isolation + progress. |
| `requirements.txt` | `httpx`, `pydantic`. |
| `README.md` | usage, secret setup, cookie-refresh instructions. |

## Data flow

```
slug list ──► fetch.py (scrape.do → Voyager) ──► raw JSON ──► save raw
                                                     │
                                                     ▼
                                              parse.py ──► Profile ──► save parsed
```

Sequential, one profile at a time, polite delay between each.

## Extraction targets

- **Identity:** name, headline, location.
- **Bio:** `summary`.
- **Experience** (list): title, company, start/end dates, description.
- **Education** (list): school, degree, field of study, start/end dates.

Schema is intentionally easy to extend (skills, certifications) later.

## Output

JSON-only (nested profile data; flat parquet would mangle multi-job/multi-school
records). No parquet summary index now — deferred (YAGNI); trivially generated
from the JSON if cross-querying is ever needed.

- `data/linkedin/raw/{slug}.json` — untouched Voyager response. Durable,
  re-parseable asset; parser changes never force a re-fetch.
- `data/linkedin/parsed/{slug}.json` — structured `Profile`.

## Error handling

- **Per-profile isolation.** A withheld/private/failed profile logs and skips;
  the batch never aborts (same pattern as the Coinglass client).
- **Auth expiry.** Voyager 401 / login-challenge response → fail loud with an
  explicit "refresh your `li_at` / `JSESSIONID` in 1Password" message, never
  silently emit empty profiles.
- **Transient errors** (429/5xx/timeout) → bounded retry with backoff, mirroring
  `a16z_research/fetch.py`.

## Testing (TDD)

- `pytest`, `asyncio_mode=auto` (repo convention), tests in root `tests/`.
- **Fixture-based, no live calls.** Record one real Voyager `profileView` JSON
  payload as a fixture; tests drive `parse.py` against it (and edge variants:
  missing summary, no education, multiple positions, private/withheld).
- `fetch.py` tested against a mocked scrape.do response (httpx mock): asserts
  correct URL construction (sticky session, geo, customHeaders), auth-header
  injection, retry on 5xx, auth-expiry detection.

## Out of scope

- Profile **discovery** / search (Jax supplies the list).
- Connections / posts / activity scraping.
- Concurrency, scheduling, incremental re-runs.
- parquet summary index (deferred).
