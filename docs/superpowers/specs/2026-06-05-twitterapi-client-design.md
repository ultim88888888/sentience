# twitterapi.io client — design

- **Date:** 2026-06-05
- **Project:** sentience
- **Status:** approved (design); pending implementation plan
- **Author:** Fushi

## Goal

Scrape **all posts for a given user since a designated start date** from
[twitterapi.io](https://twitterapi.io), into the sentience corpus. Replies, quotes,
and retweets are in scope. Output is one parquet per user, one row per tweet.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Collection strategy | **Depth-first** via `tweet/advanced_search` | Date-windowed queries reach arbitrarily far back regardless of account volume. The `user/last_tweets` timeline is capped at ~3,200 historical tweets — unacceptable for a date floor on prolific accounts. |
| Row content | **Curated columns + `raw_json`** | Flat curated schema for analysis; full object retained so nothing is lost without re-scraping. Matches the a16z corpus pattern. |
| Incremental / resume | **Not built (v1)** | Re-scrape is idempotent and cheap. Each run overwrites `<username>.parquet`. |
| Cross-run dedup | **Not built (v1)** | Single-run overwrite, no merge. |

## Open risk — RESOLVED (sprint 1, 2026-06-05)

`advanced_search`'s `from:<user>` query **excludes native retweets** (confirmed live:
0 retweets in the base pass vs. retweets present once `filter:nativeretweets` was added).
Replies and quotes surface fine. **Cure (shipped, not deferred):** `fetch_user` runs a
second pass with `include:nativeretweets filter:nativeretweets`, unioned + deduped by id —
complete coverage stays inside `advanced_search`; the `last_tweets` supplement was not
needed. Live-verified against `@eddylazzarin`: 631 tweets since 2025-01-01
(407 reply / 141 quote / 82 original / 1 retweet), no dupes, date floor respected.

## API reference (as of 2026-06-05)

- **Endpoint:** `GET https://api.twitterapi.io/twitter/tweet/advanced_search`
- **Auth:** `X-API-Key` header. Key in 1Password: item `twitterapi-io`, vault `local`,
  field `credential`. Pulled at runtime via `op` (never hardcoded), same as
  `market_data.coinglass.api_key()`.
- **Params:** `query` (required), `queryType` = `Latest` (newest→oldest), `cursor`
  (empty string for first page).
- **Query syntax:** `from:<username> since_time:<unix> until_time:<unix>`.
  (Per API docs, the `since:`/`until:` date forms are unsupported — use `*_time` unix.)
- **Pagination:** response has `tweets[]`, `has_next_page` (bool), `next_cursor`
  (string). ~20 tweets/page.
- **Tweet fields used:** `id, url, text, source, createdAt, lang, retweetCount,
  replyCount, likeCount, quoteCount, viewCount, bookmarkCount, isReply, inReplyToId,
  inReplyToUserId, inReplyToUsername, conversationId, author{...}, quoted_tweet,
  retweeted_tweet`.
- **Credits:** ~723k remaining as of 2026-06-05; advanced_search cost is negligible
  for this corpus. Test account: `@eddylazzarin`.

## Architecture

Lives in `scrapers/twitter/` (data-collection taxonomy, like `a16z_research/`) but
shaped like the `coinglass` client (pure async REST, no Playwright). Mirrors existing
conventions: `httpx.AsyncClient`, `RateLimiter` + `asyncio.Semaphore`, sync `pull()`
entrypoint, tz-aware-UTC datetimes, parquet output.

### Files

- **`config.py`** — `BASE`, `OP_ITEM`/`OP_VAULT`, rate/concurrency/backoff constants,
  `DATA_DIR`, curated column list.
- **`client.py`** — `api_key()` (1Password subprocess). `TwitterAPI` async client:
  `_get(path, params)` does one rate-limited, `X-API-Key`-authed GET with exponential
  backoff on 429/5xx and raises on `status:"error"` payloads. `advanced_search(query,
  cursor)` → `SearchPage(tweets, has_next_page, next_cursor)`.
- **`collect.py`** —
  - `fetch_user(client, username, since, until=now)`: builds the `from:` query, walks
    the cursor newest→oldest until `has_next_page` is false, with a `max_pages` safety
    guard and a belt-and-suspenders `createdAt < since` break. Returns list of raw
    tweet dicts.
  - `normalize(tweet)` → one flat row dict. **Type tag:** `retweet` (has
    `retweeted_tweet`) / `quote` (has `quoted_tweet`) / `reply` (`isReply`) /
    `original`. Parses Twitter `createdAt` ("Wed Oct 10 20:19:24 +0000 2018") → UTC.
    Serializes the full object into `raw_json`.
- **`run.py`** — CLI:
  `python -m scrapers.twitter.run --user eddylazzarin --since 2024-01-01 [--until …]
  [--max-pages N]`. Sync `pull(users: list[str], since, ...)` parallelizes across users
  under the shared limiter (per-user failures logged and skipped). Writes
  `data/twitter/<username>.parquet`.
- **`requirements.txt`** — `httpx`, `pandas`, `pyarrow`.
- **`README.md`** — usage, schema, cost note, the RT-coverage caveat, the time-windowing
  escape hatch.

### Curated schema (per row)

```
id, created_at (UTC), type, text, lang,
author_id, author_username, author_name,
reply_count, retweet_count, like_count, quote_count, view_count, bookmark_count,
conversation_id, in_reply_to_id, in_reply_to_user_id, in_reply_to_username,
quoted_id, retweeted_id, url, raw_json
```

## Error handling

- 429 / 5xx → exponential backoff, capped retries (mirror `Coinglass._get`).
- `status:"error"` payload → raise `RuntimeError` with the API message.
- Per-user failure in `pull()` → log and skip; never kill the batch.
- Empty result set → write an empty (typed) frame, log it.

## Escape hatch (not built yet)

If a single query's cursor proves depth-capped, chunk `[since, until]` into windows
(e.g. monthly) and concatenate. Documented in README; added only if we hit the wall.

## Testing (TDD, mocked httpx — like `test_market_data_coinglass`)

No live API calls in the suite. Mock `op` key retrieval and httpx responses.

- Multi-page cursor walk terminates on `has_next_page:false`.
- Stop-at-`since` break (tweet older than floor).
- All four type tags (`original`/`reply`/`quote`/`retweet`).
- `createdAt` → tz-aware UTC parse.
- `raw_json` retains the full object.
- 429 → backoff → success.
- Empty result → typed empty frame.
