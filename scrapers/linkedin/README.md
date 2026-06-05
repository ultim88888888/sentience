# LinkedIn Profile Scraper

Low-volume client (<20 profiles) that pulls work experience, education, and bio
for a supplied list of people, by parsing LinkedIn's **public** profile page
fetched through scrape.do.

## Why the public page

LinkedIn's internal Voyager API is blocked at the edge for proxied requests
(scrape.do returns `ROTATION_FAILED`), and the *authenticated* profile page is an
obfuscated React DOM that rotates class names per deploy — brittle to parse.

The **public** profile page, by contrast, is a crawler-facing server-rendered
template carrying a clean schema.org `ld+json` Person node (name, bio, jobTitle,
worksFor, alumniOf) plus stable, non-obfuscated section markup
(`li.experience-item`, `li.education__list-item`). We fetch it via scrape.do with
a residential proxy (`super=true`, to clear LinkedIn's Cloudflare WAF) — **no auth
cookie, no JS render** — and parse those.

### Completeness depends on the target's public visibility

What the logged-out page exposes is controlled by each person's *public profile*
setting. Public targets parse fully. Targets that have restricted their public
profile come back thin (name/headline only or an authwall); the run **flags
these** in `data/linkedin/_restricted.txt` so you can pull them via an
authenticated browser (Chrome) fallback.

## Secrets (1Password vault `local`)

Only the scrape.do token is needed (item `scrape.do`, already present). No
LinkedIn cookies required for the public-page route.

## Run

```bash
# single profile (slug or full URL)
.venv/bin/python -m scrapers.linkedin.run williamhgates
.venv/bin/python -m scrapers.linkedin.run https://www.linkedin.com/in/williamhgates/

# a list, one slug/URL per line
.venv/bin/python -m scrapers.linkedin.run people.txt
```

Output (all under `data/linkedin/`, gitignored — profile data is personal):

- `raw/{slug}.html` — untouched public page (durable; re-parse from here, never
  re-scrape).
- `parsed/{slug}.json` — structured profile (name, headline, location, bio,
  experience[], education[]).
- `_restricted.txt` — slugs that came back thin, for the Chrome fallback.

## Maintenance note

The parser anchors on the public template's stable section classes and the
`ld+json` block, not on the rotating obfuscated classes. If LinkedIn redesigns the
public profile page, update `parse.py`; the saved raw HTML means re-parsing never
requires re-scraping.
