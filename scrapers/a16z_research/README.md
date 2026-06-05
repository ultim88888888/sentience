# a16z crypto research scraper

Scrapes every article in the **research** category of
[a16zcrypto.com](https://a16zcrypto.com/posts/focus-areas/research/) to parquet.

## How it works

The research listing is powered by an Algolia index, fronted by an AWS WAF challenge
and a **short-lived (300s) secured API key** minted per-visitor at `POST /api/generate-key`
(pinned to the caller's IP). The cURL key a16z's site hands out expires within ~24h, so
it can't be reused — the key must be freshly minted.

Pipeline:

1. **Mint key** (`keys.py`) — a headless Chromium loads the research page, lets the AWS WAF
   `challenge.js` solve, then calls `/api/generate-key` from page context. The returned key
   is pinned to this machine's IP, so Algolia queries must run from the same machine.
2. **Metadata** (`algolia.py`) — paginate the Algolia index (`taxonomies.category:research`,
   `post_type:post`); ~235 posts over 3 pages of 100. Each hit carries rich metadata **plus
   `acf_content`** (Algolia's own plain-text body — kept as a bonus/fallback field).
3. **Content** (`fetch.py`) — fetch each article page through **scrape.do** at concurrency 90
   (no JS render needed — pages are server-rendered). Algolia permalinks point at the
   `cms.a16zcrypto.com` backend host; some posts only resolve on the public host and vice
   versa, so a 404 triggers an automatic host-swap fallback.
4. **Extract** (`extract.py`) — `trafilatura` pulls clean main-article text + markdown.
5. **Write** (`run.py`) — two parquet files (below).

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r scrapers/a16z_research/requirements.txt
python -m playwright install chromium          # one-time browser download
python -m scrapers.a16z_research.run            # full scrape (~45s)
python -m scrapers.a16z_research.run --limit 5  # smoke test
python -m scrapers.a16z_research.run --metadata-only
```

The scrape.do API token is read from 1Password (`op item get scrape.do --vault local`);
`op` must be signed in.

## Output (`data/a16z_research/`)

**`metadata.parquet`** — one row per post (235), Algolia-sourced:
`object_id, post_id, title, permalink, permalink_raw, post_date, post_modified,
post_date_formatted, categories, tags, formats, author_slugs, poster_display_name,
description, excerpt, meta_description, acf_content, acf_content_len, images`

**`articles.parquet`** — metadata joined with fetched content:
adds `fetched_url, fetch_status, fetch_error, fetched_at, raw_html, raw_html_len,
extracted_text, extracted_markdown, extracted_text_len, meta_title, meta_author, meta_date`

Body strategy: prefer `extracted_text` (full, authoritative — recovers long articles that
Algolia truncates), fall back to `acf_content` (often richer for videos/podcasts, whose
pages carry little prose). 234/235 posts have a body; the lone exception is a whitepaper
PDF link with no prose anywhere.

## Known limitations

- Dates: `post_date`/`post_modified` normalized to UTC ISO-8601.
- `extracted_markdown` can carry a little page chrome (e.g. the search widget) because
  extraction favors recall; `extracted_text` is the clean field.
- The corpus spans 4 formats: ~139 videos, ~85 articles, ~10 podcasts, 1 paper.
