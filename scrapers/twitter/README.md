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

## Known limitation (verify per corpus)

`advanced_search` runs on Twitter's search index. `from:<user>` reliably returns the
user's originals, replies, and quotes; **pure retweets may be under-represented**.
If a corpus needs guaranteed retweet coverage, supplement with `user/last_tweets`
(capped ~3,200 historical tweets). Not built — added only if needed.

## Escape hatch

If a single query's cursor proves depth-capped, chunk `[since, until]` into windows
(e.g. monthly) and concatenate. Not built — added only if we hit the wall.

## Cost

Negligible for normal corpora (~723k credits available as of 2026-06-05).
