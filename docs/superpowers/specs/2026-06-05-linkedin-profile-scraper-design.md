---
date: 2026-06-05
project: sentience
status: shipped
tags: [scraper, linkedin, scrape.do, public-profile, ld-json]
---

# LinkedIn Profile Scraper — Design

> **Revision (2026-06-05, post-implementation).** The original Voyager-API design
> below was **invalidated by live testing** and replaced with a public-page HTML
> parse. Evidence gathered against scrape.do:
> - **Voyager API is dead via scrape.do** — every variant (cookie-header auth,
>   `setCookies` auth, super/no-super, render) returns `ROTATION_FAILED`
>   ("cannot connect target url"). LinkedIn blocks proxied access to `/voyager/`
>   at the edge, regardless of authentication.
> - The **authenticated** profile page works via scrape.do's `setCookies`
>   param, but its data lives in **obfuscated, per-deploy-rotating** React DOM
>   classes — brittle to parse.
> - The **public** profile page is a clean, crawler-facing SSR template with a
>   schema.org `ld+json` Person node + stable section markup, and needs **no
>   auth**. This is the shipped approach. Completeness is bounded by each
>   target's public-profile visibility; restricted targets are flagged for an
>   authenticated (Chrome) fallback.
>
> The **Approach** and **Coverage & secrets** sections below are updated to the
> shipped design. The remaining lower sections (module layout, data flow, output,
> testing) describe the original Voyager plan and are **superseded** — see
> `scrapers/linkedin/README.md` and the code for the authoritative shipped detail.

## Purpose

A low-volume client that scrapes structured profile data — **work experience,
education, and bio** — for a hand-supplied list of LinkedIn people (realistically
<20, capped well under 100). Feeds the `sentience` research corpus alongside the
a16z and market-data datasets.

This is a **simple sequential client**. Tiny volume removes any need for
concurrency, batching, or aggressive rate-limit machinery.

## Approach

**Public profile page via scrape.do, parse `ld+json` + stable HTML sections.**

Fetch `https://www.linkedin.com/in/{slug}/` through scrape.do with `super=true`
(residential proxy, to clear LinkedIn's Cloudflare WAF) and `geoCode=us` — no auth
cookie, no JS render. The logged-out public page is a server-rendered template
carrying:

- a schema.org **`ld+json` Person node** — `name`, `description` (bio),
  `jobTitle`, `worksFor` (orgs + start years), `alumniOf` (schools + years); and
- **stable, non-obfuscated HTML sections** — `li.experience-item`
  (`.experience-item__title`/`__subtitle`, `.date-range`) and
  `li.education__list-item` — for per-entry detail (title, dates, degree).

The parser takes name/bio from the `ld+json` and per-entry detail from the HTML
sections, falling back to the `ld+json` worksFor/alumniOf when the HTML sections
are absent (itself a signal the profile is restricted).

### Rejected / failed alternatives

- **Voyager API (original plan).** Blocked at the edge for proxied requests —
  `ROTATION_FAILED` on every scrape.do variant. Dead.
- **Authenticated HTML via scrape.do `setCookies`.** Works, but data is in
  obfuscated per-deploy-rotating React DOM classes — brittle. Not used.
- **LinkedIn-specialized API (Proxycurl etc.).** Clean JSON, all-profiles, but
  paid + abandons scrape.do. Considered as fallback; deferred.

## Coverage & secrets

Only the **scrape.do token** is needed (1Password item `scrape.do`, vault
`local`). No LinkedIn cookies — the public-page route is logged-out.

Completeness is bounded by each target's public-profile visibility. Targets that
come back thin (restricted/authwall) are flagged in `data/linkedin/_restricted.txt`
for an authenticated **Chrome fallback** (pulled manually via the logged-in
session, where the data is complete regardless of public setting).

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
