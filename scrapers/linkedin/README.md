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

Capture both in the **same** DevTools sitting — a `li_at` and `JSESSIONID` from
different logins fail LinkedIn's CSRF check. Cookies expire periodically; when a
run fails with an auth-expiry error, re-copy both and update the 1Password items.

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

Both output dirs are gitignored — scraped profile data is personal and is not
committed.

## Session-security note

The run pins ONE residential IP (`SCRAPEDO_SESSION_ID` in `config.py`) geo-matched
to the account (`SCRAPEDO_GEO`). This avoids LinkedIn's "new location" lockout —
the real risk, not request volume. If the account normally logs in outside the US,
change `SCRAPEDO_GEO`.
