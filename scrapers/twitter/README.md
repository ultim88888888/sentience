# twitter — twitterapi.io scraper

Scrapes **all posts for a user since a start date** (originals, replies, quotes,
retweets) via twitterapi.io's `advanced_search`. One parquet per user, one row per tweet.

## Usage

```bash
# CLI
python -m scrapers.twitter.run --user eddylazzarin --since 2024-01-01
python -m scrapers.twitter.run --user a --user b --since 2024-01-01 --until 2024-06-01

# Programmatic
from scrapers.twitter.run import pull
frames = pull(["eddylazzarin"], since="2024-01-01")
```

Output: `data/twitter/<username>.parquet`. Each run **overwrites** (no incremental/dedup in v1).

## Auth

API key in 1Password: item `twitterapi-io`, vault `local`, field `credential`.
Pulled at runtime via `op` — never hardcoded.

## Schema

Curated columns + a `raw_json` column holding the full tweet object. Type tag precedence:
`retweet` > `quote` > `reply` > `original`. `created_at` is tz-aware UTC (`datetime64[ns, UTC]`).

## Retweet coverage

`advanced_search`'s `from:<user>` query excludes native retweets. To capture them,
`fetch_user` runs a **second pass** with `include:nativeretweets filter:nativeretweets`
and unions the results (deduped by tweet id). On by default; opt out with `--no-retweets`
(CLI) or `include_retweets=False` (`pull`). Verified live 2026-06-05 against
`@eddylazzarin` (replies + quotes + retweets all present).

## Escape hatch

If a single query's cursor proves depth-capped, chunk `[since, until]` into windows
(e.g. monthly) and concatenate. Not built — added only if we hit the wall.

## Cost

Negligible for normal corpora (~723k credits available as of 2026-06-05).
