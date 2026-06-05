# a16z crypto team scraper

Scrapes the **team roster** at [a16zcrypto.com/our-team](https://a16zcrypto.com/our-team/)
to parquet — one row per person, with name, title, bio, and social accounts.

Sibling of the `a16z_research` scraper: it **reuses that scraper's scrape.do fetch
layer** (`a16z_research/fetch.py` — token from 1Password, transient-retry, concurrency)
so every request goes through the same proxy. This module adds only the team-roster
parsing.

## How it works

Both the roster and the profile pages are **server-rendered** — a plain scrape.do GET
returns the full markup, so no headless browser / JS render is needed (unlike the
research scraper, which mints an Algolia key in a browser).

Pipeline (two network stages, both via scrape.do):

1. **Roster** (`listing.py`) — GET `/our-team/`, parse the member grid. Members are
   grouped under section headers (Investing, Engineering, Research, … Advisors); a
   person can appear under more than one section (e.g. Eddy Lazzarin in Engineering
   **and** Research), so members are **deduped by slug** and their sections merged.
   Yields `slug, name, listing_title, sections`. ~89 members.
2. **Profiles** (`fetch.py`, reused) — GET each `/team/<slug>` page through scrape.do.
3. **Extract** (`extract.py`) — from each profile, pull the authoritative `title`,
   the `bio` (the prose paragraphs in the bio column), and `socials`. Extraction is
   **scoped to the profile `<section>`** so the site-wide footer socials (a16zcrypto,
   a16z) never leak in. The socials list is the `flex-wrap` `<ul>`; a plain `<ul>` in
   the same column is a *publications* list (research members) and is deliberately
   excluded. Each social is normalized to a platform key by link label then host
   (`x, linkedin, farcaster, github, website, …`).
4. **Write** (`run.py`) — one parquet (below).

## Run

```bash
. .venv/bin/activate                              # shared sentience venv
pip install -r scrapers/a16z_team/requirements.txt
python -m scrapers.a16z_team.run                  # full scrape (~14s, 89 members)
python -m scrapers.a16z_team.run --limit 5        # smoke test
python -m scrapers.a16z_team.run --roster-only    # parse roster, print, don't fetch profiles
```

The scrape.do API token is read from 1Password (`op item get scrape.do --vault local`);
`op` must be signed in.

## Output (`data/a16z_team/team.parquet`)

One row per person (~89):

`slug, name, title, listing_title, sections, bio, bio_len, socials_json,
socials_count, x_url, linkedin_url, farcaster_url, github_url, website_url,
profile_url, fetch_status, fetch_error, fetched_at, raw_html, raw_html_len`

- **`slug`** — `/team/<slug>` id; the join key to the research corpus's `author_slugs`.
- **`title`** — authoritative title from the profile page; falls back to the roster
  card title if the profile omits it.
- **`sections`** — list; departments the person is listed under on the roster.
- **`socials_json`** — full list as JSON: `[{platform, label, url}, …]`. The broken-out
  `*_url` columns are convenience projections for the common platforms (the JSON is the
  complete source — e.g. OpenSea links live only there).
- **`raw_html`** — kept so bios/socials can be re-extracted without re-fetching.

## Known limitations

- **Bios:** ~17 members have no published bio on the site (mostly operations / EA /
  some stub partner pages). Those rows carry `bio = NULL`, `bio_len = 0` — verified
  against the source markup (the bio column is literally empty), not an extraction miss.
- **Socials:** ~31 of 89 members list any socials; the rest publish none.
- Social URLs are stored **verbatim** from the site, including a16z's own quirks (e.g.
  a stray `@` in some `twitter.com/@handle` links).
