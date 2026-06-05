# twitterapi.io Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scrape all posts (originals, replies, quotes, retweets) for a given user since a designated start date from twitterapi.io into one parquet per user.

**Architecture:** Pure async REST client (`httpx`) over twitterapi.io's `tweet/advanced_search`, shaped like the existing `market_data.coinglass` client. Walk the `from:<user> since_time:… until_time:…` query newest→oldest via cursor pagination until exhausted, normalize each tweet to a flat row with type tag + retained `raw_json`, write parquet.

**Tech Stack:** Python, `httpx` (async), `pandas` + `pyarrow`, `op` (1Password CLI) for the API key, `pytest` + `unittest.mock` for TDD.

**Conventions mirrored from `market_data/coinglass.py`:** `api_key()` via `op` subprocess, `RateLimiter` + `asyncio.Semaphore`, exponential backoff on rate-limit/5xx, tz-aware-UTC datetimes, sync `pull()` convenience entrypoint, parquet to `data/`.

**Run tests with:** `.venv/bin/python -m pytest tests/test_twitter_client.py -v`

---

### Task 1: Package scaffold + config

**Files:**
- Create: `scrapers/twitter/__init__.py`
- Create: `scrapers/twitter/config.py`
- Create: `scrapers/twitter/requirements.txt`

- [ ] **Step 1: Create `scrapers/twitter/__init__.py`** (empty package marker)

```python
```

- [ ] **Step 2: Create `scrapers/twitter/requirements.txt`**

```
httpx>=0.27
pandas>=2.0
pyarrow>=15.0
```

- [ ] **Step 3: Create `scrapers/twitter/config.py`**

```python
"""Configuration for the twitterapi.io scraper."""
from pathlib import Path

# twitterapi.io REST API
BASE = "https://api.twitterapi.io"
ADVANCED_SEARCH_PATH = "/twitter/tweet/advanced_search"

# API key in 1Password (pulled at runtime via `op`, never hardcoded).
OP_ITEM = "twitterapi-io"
OP_VAULT = "local"

HTTP_TIMEOUT = 60.0

# Self-imposed pacing. twitterapi.io tolerates high QPS, but we pace to be polite
# and keep cost predictable. Sequential cursor walk per user; concurrency is across users.
RATE_LIMIT_PER_SEC = 10.0     # max request issuances/sec, shared across all users
MAX_CONCURRENCY = 5           # max users fetched simultaneously

# Backoff for 429/5xx retries
_BACKOFF_BASE = 4.0           # seconds
_BACKOFF_FACTOR = 2.0
_BACKOFF_MAX_RETRIES = 4

# Safety guard: cap pages per user (20 tweets/page). 5000 pages = 100k tweets.
MAX_PAGES = 5000

# Output
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "twitter"

# Curated columns, in order. `raw_json` holds the full tweet object.
COLUMNS = [
    "id", "created_at", "type", "text", "lang",
    "author_id", "author_username", "author_name",
    "reply_count", "retweet_count", "like_count", "quote_count",
    "view_count", "bookmark_count",
    "conversation_id", "in_reply_to_id", "in_reply_to_user_id", "in_reply_to_username",
    "quoted_id", "retweeted_id", "url", "raw_json",
]
```

- [ ] **Step 4: Commit**

```bash
git add scrapers/twitter/__init__.py scrapers/twitter/config.py scrapers/twitter/requirements.txt
git commit -m "feat(twitter): package scaffold + config"
```

---

### Task 2: `normalize()` — flatten one tweet to a row

**Files:**
- Create: `scrapers/twitter/collect.py`
- Test: `tests/test_twitter_client.py`

`normalize()` is pure (no I/O), so build it first. Type-tag precedence: **retweet > quote > reply > original** (a repost is not original content; a quote-reply is tagged quote).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_twitter_client.py`:

```python
"""TDD tests for scrapers.twitter — twitterapi.io client.

Execution: .venv/bin/python -m pytest tests/test_twitter_client.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from scrapers.twitter import collect


def _tweet(**over):
    base = {
        "id": "1", "url": "https://x.com/u/status/1", "text": "hello",
        "lang": "en", "createdAt": "Wed Oct 10 20:19:24 +0000 2018",
        "replyCount": 1, "retweetCount": 2, "likeCount": 3, "quoteCount": 4,
        "viewCount": 5, "bookmarkCount": 6, "conversationId": "1",
        "isReply": False, "inReplyToId": None, "inReplyToUserId": None,
        "inReplyToUsername": None, "quoted_tweet": None, "retweeted_tweet": None,
        "author": {"id": "99", "userName": "eddy", "name": "Eddy L"},
    }
    base.update(over)
    return base


def test_normalize_original():
    row = collect.normalize(_tweet())
    assert row["id"] == "1"
    assert row["type"] == "original"
    assert row["text"] == "hello"
    assert row["author_id"] == "99"
    assert row["author_username"] == "eddy"
    assert row["author_name"] == "Eddy L"
    assert row["like_count"] == 3
    assert row["conversation_id"] == "1"


def test_normalize_created_at_is_utc():
    row = collect.normalize(_tweet())
    assert isinstance(row["created_at"], datetime)
    assert row["created_at"] == datetime(2018, 10, 10, 20, 19, 24, tzinfo=timezone.utc)


def test_normalize_type_reply():
    row = collect.normalize(_tweet(isReply=True, inReplyToId="7",
                                   inReplyToUserId="8", inReplyToUsername="bob"))
    assert row["type"] == "reply"
    assert row["in_reply_to_id"] == "7"
    assert row["in_reply_to_username"] == "bob"


def test_normalize_type_quote():
    row = collect.normalize(_tweet(quoted_tweet={"id": "55"}))
    assert row["type"] == "quote"
    assert row["quoted_id"] == "55"


def test_normalize_type_retweet():
    # retweet takes precedence even if other flags are set
    row = collect.normalize(_tweet(isReply=True, retweeted_tweet={"id": "77"}))
    assert row["type"] == "retweet"
    assert row["retweeted_id"] == "77"


def test_normalize_raw_json_roundtrips():
    t = _tweet()
    row = collect.normalize(t)
    assert json.loads(row["raw_json"])["id"] == "1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'scrapers.twitter.collect' has no attribute 'normalize'`

- [ ] **Step 3: Write `scrapers/twitter/collect.py` with `normalize()`**

```python
"""Collect + normalize tweets from twitterapi.io advanced_search."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Twitter's createdAt format, e.g. "Wed Oct 10 20:19:24 +0000 2018"
_TWITTER_TS_FMT = "%a %b %d %H:%M:%S %z %Y"


def parse_created_at(value: str) -> datetime:
    """Parse Twitter's createdAt string to a tz-aware UTC datetime."""
    dt = datetime.strptime(value, _TWITTER_TS_FMT)
    return dt.astimezone(timezone.utc)


def _tweet_type(tweet: dict[str, Any]) -> str:
    """Tag tweet type. Precedence: retweet > quote > reply > original."""
    if tweet.get("retweeted_tweet"):
        return "retweet"
    if tweet.get("quoted_tweet"):
        return "quote"
    if tweet.get("isReply"):
        return "reply"
    return "original"


def normalize(tweet: dict[str, Any]) -> dict[str, Any]:
    """Flatten one twitterapi.io tweet object to a curated row dict (+ raw_json)."""
    author = tweet.get("author") or {}
    quoted = tweet.get("quoted_tweet") or {}
    retweeted = tweet.get("retweeted_tweet") or {}
    return {
        "id": tweet.get("id"),
        "created_at": parse_created_at(tweet["createdAt"]),
        "type": _tweet_type(tweet),
        "text": tweet.get("text"),
        "lang": tweet.get("lang"),
        "author_id": author.get("id"),
        "author_username": author.get("userName"),
        "author_name": author.get("name"),
        "reply_count": tweet.get("replyCount"),
        "retweet_count": tweet.get("retweetCount"),
        "like_count": tweet.get("likeCount"),
        "quote_count": tweet.get("quoteCount"),
        "view_count": tweet.get("viewCount"),
        "bookmark_count": tweet.get("bookmarkCount"),
        "conversation_id": tweet.get("conversationId"),
        "in_reply_to_id": tweet.get("inReplyToId"),
        "in_reply_to_user_id": tweet.get("inReplyToUserId"),
        "in_reply_to_username": tweet.get("inReplyToUsername"),
        "quoted_id": quoted.get("id"),
        "retweeted_id": retweeted.get("id"),
        "url": tweet.get("url"),
        "raw_json": json.dumps(tweet, ensure_ascii=False),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scrapers/twitter/collect.py tests/test_twitter_client.py
git commit -m "feat(twitter): normalize() with type tagging + raw_json"
```

---

### Task 3: `TwitterAPI` client — auth, rate-limited GET, `advanced_search`

**Files:**
- Create: `scrapers/twitter/client.py`
- Test: `tests/test_twitter_client.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_twitter_client.py`)

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from scrapers.twitter import client as twclient


def _resp(status_code=200, payload=None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = payload if payload is not None else {}
    return r


def test_api_key_reads_from_1password():
    fake = MagicMock()
    fake.stdout = "secret-key\n"
    with patch("scrapers.twitter.client.subprocess.run", return_value=fake) as run:
        assert twclient.api_key() == "secret-key"
    assert run.call_args[0][0][:3] == ["op", "item", "get"]


def test_advanced_search_returns_page():
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(payload={
        "tweets": [{"id": "1"}],
        "has_next_page": True,
        "next_cursor": "CURSOR2",
        "status": "success",
    }))
    limiter = twclient.RateLimiter(1000.0)
    sem = asyncio.Semaphore(1)
    c = twclient.TwitterAPI(http=http, limiter=limiter, semaphore=sem)
    page = asyncio.run(c.advanced_search("from:eddy", cursor=""))
    assert page.tweets == [{"id": "1"}]
    assert page.has_next_page is True
    assert page.next_cursor == "CURSOR2"
    # query + queryType + cursor passed as params
    params = http.get.call_args.kwargs["params"]
    assert params["query"] == "from:eddy"
    assert params["queryType"] == "Latest"


def test_get_retries_on_429_then_succeeds():
    http = MagicMock()
    http.get = AsyncMock(side_effect=[
        _resp(status_code=429, payload={"status": "error", "msg": "rate"}),
        _resp(payload={"tweets": [], "has_next_page": False, "next_cursor": "", "status": "success"}),
    ])
    limiter = twclient.RateLimiter(1000.0)
    sem = asyncio.Semaphore(1)
    c = twclient.TwitterAPI(http=http, limiter=limiter, semaphore=sem)
    with patch("scrapers.twitter.client.asyncio.sleep", new=AsyncMock()):
        page = asyncio.run(c.advanced_search("from:eddy", cursor=""))
    assert page.has_next_page is False
    assert http.get.call_count == 2


def test_get_raises_on_error_status():
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(payload={"status": "error", "msg": "bad query"}))
    limiter = twclient.RateLimiter(1000.0)
    sem = asyncio.Semaphore(1)
    c = twclient.TwitterAPI(http=http, limiter=limiter, semaphore=sem)
    with pytest.raises(RuntimeError, match="bad query"):
        asyncio.run(c.advanced_search("from:eddy", cursor=""))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: FAIL — `ModuleNotFoundError: scrapers.twitter.client`

- [ ] **Step 3: Write `scrapers/twitter/client.py`**

```python
"""Async twitterapi.io client: auth, rate-limited GET, advanced_search."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import (
    ADVANCED_SEARCH_PATH, BASE, OP_ITEM, OP_VAULT,
    _BACKOFF_BASE, _BACKOFF_FACTOR, _BACKOFF_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def api_key() -> str:
    """Retrieve the twitterapi.io API key from 1Password (never hardcoded)."""
    result = subprocess.run(
        ["op", "item", "get", OP_ITEM, "--vault", OP_VAULT,
         "--fields", "credential", "--reveal"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


class RateLimiter:
    """Paces async requests to at most `per_second` issuances, shared across callers."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait < 0:
                wait = 0.0
                self._next_at = now
            self._next_at += self._interval
        if wait > 0:
            await asyncio.sleep(wait)


@dataclass
class SearchPage:
    tweets: list[dict[str, Any]]
    has_next_page: bool
    next_cursor: str


def _is_rate_limited(status_code: int, payload: dict[str, Any]) -> bool:
    if status_code == 429 or status_code >= 500:
        return True
    msg = str(payload.get("msg", "")).lower()
    return "rate" in msg or "too many requests" in msg


class TwitterAPI:
    """Async twitterapi.io client."""

    def __init__(self, http: httpx.AsyncClient, limiter: RateLimiter,
                 semaphore: asyncio.Semaphore) -> None:
        self._http = http
        self._limiter = limiter
        self._semaphore = semaphore

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE}{path}"
        for attempt in range(_BACKOFF_MAX_RETRIES + 1):
            await self._limiter.acquire()
            async with self._semaphore:
                resp = await self._http.get(url, params=params)
                payload: dict[str, Any] = resp.json()

            if _is_rate_limited(resp.status_code, payload):
                if attempt < _BACKOFF_MAX_RETRIES:
                    sleep_secs = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                    logger.warning("Rate/5xx on %s (attempt %d/%d), sleeping %.1fs",
                                   path, attempt + 1, _BACKOFF_MAX_RETRIES, sleep_secs)
                    await asyncio.sleep(sleep_secs)
                    continue
                raise RuntimeError(f"Rate limited after {_BACKOFF_MAX_RETRIES} retries: {payload}")

            if payload.get("status") == "error":
                raise RuntimeError(f"twitterapi.io error: {payload.get('msg', payload)}")

            return payload
        raise RuntimeError("Unexpected: exhausted retry loop")  # pragma: no cover

    async def advanced_search(self, query: str, cursor: str = "") -> SearchPage:
        """One page of advanced_search results (queryType=Latest, newest→oldest)."""
        payload = await self._get(ADVANCED_SEARCH_PATH, params={
            "query": query, "queryType": "Latest", "cursor": cursor,
        })
        return SearchPage(
            tweets=payload.get("tweets", []),
            has_next_page=bool(payload.get("has_next_page", False)),
            next_cursor=payload.get("next_cursor", "") or "",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: PASS (10 tests total)

- [ ] **Step 5: Commit**

```bash
git add scrapers/twitter/client.py tests/test_twitter_client.py
git commit -m "feat(twitter): TwitterAPI client — auth, rate-limited GET, advanced_search"
```

---

### Task 4: `fetch_user()` — cursor walk with stop-at-since

**Files:**
- Modify: `scrapers/twitter/collect.py`
- Test: `tests/test_twitter_client.py` (append)

`fetch_user` builds the query, walks the cursor until `has_next_page` is false, and stops early once a page's oldest tweet predates `since` (belt-and-suspenders; `since_time` already filters server-side). Guarded by `MAX_PAGES`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_twitter_client.py`)

```python
def test_to_unix_utc():
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert collect.to_unix(dt) == 1704067200


def test_build_query():
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2024, 2, 1, tzinfo=timezone.utc)
    q = collect.build_query("eddylazzarin", since, until)
    assert q == "from:eddylazzarin since_time:1704067200 until_time:1706745600"


def test_fetch_user_walks_pages_until_exhausted():
    pages = [
        collect_page([{"id": "3", "createdAt": "Wed Jan 10 00:00:00 +0000 2024"}], True, "C2"),
        collect_page([{"id": "2", "createdAt": "Wed Jan 05 00:00:00 +0000 2024"}], False, ""),
    ]
    client = MagicMock()
    client.advanced_search = AsyncMock(side_effect=pages)
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tweets = asyncio.run(collect.fetch_user(client, "eddy", since))
    assert [t["id"] for t in tweets] == ["3", "2"]
    assert client.advanced_search.call_count == 2


def test_fetch_user_stops_when_page_predates_since():
    pages = [
        collect_page([{"id": "3", "createdAt": "Wed Jan 10 00:00:00 +0000 2024"}], True, "C2"),
        # oldest tweet here is before `since` -> stop, do not request page 3
        collect_page([{"id": "old", "createdAt": "Wed Dec 10 00:00:00 +0000 2023"}], True, "C3"),
    ]
    client = MagicMock()
    client.advanced_search = AsyncMock(side_effect=pages)
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tweets = asyncio.run(collect.fetch_user(client, "eddy", since))
    assert client.advanced_search.call_count == 2  # stopped after page 2, no page 3
    assert "old" not in [t["id"] for t in tweets]  # pre-since tweet dropped
```

Also add this helper near the top of the test file (after `_tweet`):

```python
def collect_page(tweets, has_next, cursor):
    from scrapers.twitter.client import SearchPage
    return SearchPage(tweets=tweets, has_next_page=has_next, next_cursor=cursor)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: FAIL — `AttributeError: module 'scrapers.twitter.collect' has no attribute 'to_unix'`

- [ ] **Step 3: Add `to_unix`, `build_query`, `fetch_user` to `collect.py`**

Add imports at top of `collect.py`:

```python
from .config import MAX_PAGES
```

Append to `collect.py`:

```python
def to_unix(dt: datetime) -> int:
    """UTC datetime -> unix seconds."""
    return int(dt.astimezone(timezone.utc).timestamp())


def build_query(username: str, since: datetime, until: datetime) -> str:
    return f"from:{username} since_time:{to_unix(since)} until_time:{to_unix(until)}"


async def fetch_user(client, username: str, since: datetime,
                     until: datetime | None = None) -> list[dict[str, Any]]:
    """Walk advanced_search newest→oldest until exhausted or past `since`.

    Returns raw tweet dicts whose createdAt >= since. `client` is a TwitterAPI.
    """
    if until is None:
        until = datetime.now(timezone.utc)
    query = build_query(username, since, until)
    out: list[dict[str, Any]] = []
    cursor = ""
    for page_num in range(MAX_PAGES):
        page = await client.advanced_search(query, cursor=cursor)
        passed_floor = False
        for t in page.tweets:
            if parse_created_at(t["createdAt"]) < since:
                passed_floor = True
                continue
            out.append(t)
        if passed_floor or not page.has_next_page or not page.next_cursor:
            break
        cursor = page.next_cursor
    else:
        logger.warning("fetch_user(%s): hit MAX_PAGES=%d guard", username, MAX_PAGES)
    logger.info("fetch_user(%s): %d tweets since %s", username, len(out), since.date())
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: PASS (14 tests total)

- [ ] **Step 5: Commit**

```bash
git add scrapers/twitter/collect.py tests/test_twitter_client.py
git commit -m "feat(twitter): fetch_user() cursor walk with stop-at-since"
```

---

### Task 5: `run.py` — DataFrame build, parquet write, `pull()`, CLI

**Files:**
- Create: `scrapers/twitter/run.py`
- Test: `tests/test_twitter_client.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_twitter_client.py`)

```python
from scrapers.twitter import run as twrun


def test_to_frame_has_columns_and_order():
    from scrapers.twitter.config import COLUMNS
    rows = [collect.normalize(_tweet()), collect.normalize(_tweet(id="2"))]
    df = twrun.to_frame(rows)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert str(df["created_at"].dtype) == "datetime64[ns, UTC]"


def test_to_frame_empty_is_typed():
    from scrapers.twitter.config import COLUMNS
    df = twrun.to_frame([])
    assert list(df.columns) == COLUMNS
    assert len(df) == 0


def test_save_writes_parquet(tmp_path):
    rows = [collect.normalize(_tweet())]
    df = twrun.to_frame(rows)
    path = twrun.save("eddy", df, out_dir=tmp_path)
    assert path.exists()
    back = pd.read_parquet(path)
    assert back.iloc[0]["id"] == "1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: FAIL — `ModuleNotFoundError: scrapers.twitter.run`

- [ ] **Step 3: Write `scrapers/twitter/run.py`**

```python
"""Orchestrate a twitterapi.io scrape: advanced_search -> normalize -> parquet.

CLI:
    python -m scrapers.twitter.run --user eddylazzarin --since 2024-01-01

Programmatic:
    from scrapers.twitter.run import pull
    pull(["eddylazzarin"], since="2024-01-01")
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from . import collect
from .client import RateLimiter, TwitterAPI, api_key
from .config import (
    COLUMNS, DATA_DIR, HTTP_TIMEOUT, MAX_CONCURRENCY, RATE_LIMIT_PER_SEC,
)

logger = logging.getLogger(__name__)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Build the curated DataFrame (typed columns, UTC created_at) from rows."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def save(username: str, df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{username}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))
    return path


def _parse_date(s: str) -> datetime:
    """Parse a YYYY-MM-DD (or ISO) date as UTC."""
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def pull(users: list[str], since: str | datetime, until: str | datetime | None = None,
         persist: bool = True, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Fetch all tweets since `since` for each user; optionally persist parquet each.

    Per-user failures are logged and skipped. Returns {username: DataFrame}.
    """
    since_dt = _parse_date(since) if isinstance(since, str) else since
    until_dt = (_parse_date(until) if isinstance(until, str) else until)
    key = api_key()
    limiter = RateLimiter(RATE_LIMIT_PER_SEC)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _one(client: TwitterAPI, user: str) -> tuple[str, pd.DataFrame]:
        raw = await collect.fetch_user(client, user, since_dt, until_dt)
        return user, to_frame([collect.normalize(t) for t in raw])

    async def _run() -> dict[str, pd.DataFrame]:
        async with httpx.AsyncClient(headers={"X-API-Key": key}, timeout=HTTP_TIMEOUT) as http:
            client = TwitterAPI(http=http, limiter=limiter, semaphore=semaphore)
            results = await asyncio.gather(*[_one(client, u) for u in users],
                                           return_exceptions=True)
        out: dict[str, pd.DataFrame] = {}
        for user, res in zip(users, results):
            if isinstance(res, Exception):
                logger.warning("%s: pull failed (%r); skipping", user, res)
            else:
                out[res[0]] = res[1]
        return out

    frames = asyncio.run(_run())
    if persist:
        for user, df in frames.items():
            save(user, df, out_dir=out_dir)
    return frames


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Scrape a user's tweets since a date to parquet.")
    ap.add_argument("--user", action="append", required=True,
                    help="Username (repeatable for multiple users).")
    ap.add_argument("--since", required=True, help="Start date, YYYY-MM-DD (UTC).")
    ap.add_argument("--until", default=None, help="End date, YYYY-MM-DD (UTC). Default: now.")
    args = ap.parse_args()
    frames = pull(args.user, since=args.since, until=args.until)
    for user, df in frames.items():
        print(f"{user}: {len(df)} tweets", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_twitter_client.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 5: Commit**

```bash
git add scrapers/twitter/run.py tests/test_twitter_client.py
git commit -m "feat(twitter): run.py — to_frame/save/pull + CLI"
```

---

### Task 6: README

**Files:**
- Create: `scrapers/twitter/README.md`

- [ ] **Step 1: Write `scrapers/twitter/README.md`**

```markdown
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
`retweet` > `quote` > `reply` > `original`. `created_at` is tz-aware UTC.

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
```

- [ ] **Step 2: Commit**

```bash
git add scrapers/twitter/README.md
git commit -m "docs(twitter): README"
```

---

### Task 7: Live smoke test + RT/reply coverage check

This is the sprint-1 verification from the spec. **Requires network + the 1Password key.** Not part of the pytest suite.

**Files:**
- None (manual verification; results recorded in commit message / STATUS)

- [ ] **Step 1: Install deps** (if not already in `.venv`)

```bash
.venv/bin/pip install -r scrapers/twitter/requirements.txt
```

- [ ] **Step 2: Small live pull against the test account**

Run:
```bash
.venv/bin/python -m scrapers.twitter.run --user eddylazzarin --since 2025-01-01
```
Expected: writes `data/twitter/eddylazzarin.parquet`, logs a tweet count.

- [ ] **Step 3: Inspect coverage**

Run:
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/twitter/eddylazzarin.parquet')
print('total:', len(df))
print(df['type'].value_counts())
print('date range:', df['created_at'].min(), '->', df['created_at'].max())
print('null created_at:', df['created_at'].isna().sum())
"
```
Expected: nonzero `reply` count confirms replies surface. Record the `retweet` count —
**if 0**, retweets are not captured by search; note it and decide on a `last_tweets`
supplement (do NOT build preemptively). Confirm `created_at` range respects `--since`.

- [ ] **Step 4: Record the finding**

Update `docs/STATUS.md` (or a `vault/knowledge/` note) with: total tweets, type
breakdown, and the retweet-coverage verdict. Commit.

```bash
git add docs/STATUS.md
git commit -m "test(twitter): live smoke vs @eddylazzarin — coverage verdict recorded"
```

---

## Self-Review

- **Spec coverage:** depth-first advanced_search ✓ (Task 3–4); curated + raw_json ✓ (Task 2, COLUMNS in Task 1); type tagging ✓ (Task 2); 1Password auth ✓ (Task 3); RateLimiter/Semaphore/backoff ✓ (Tasks 1, 3); per-user skip in pull ✓ (Task 5); empty typed frame ✓ (Task 5); stop-at-since + MAX_PAGES ✓ (Task 4); parquet to data/twitter ✓ (Tasks 1, 5); RT-coverage verification ✓ (Task 7); README incl. escape hatch ✓ (Task 6). No gaps.
- **Type consistency:** `TwitterAPI.advanced_search` returns `SearchPage(tweets, has_next_page, next_cursor)` — used consistently in Tasks 3–5. `normalize()` row keys match `COLUMNS` exactly. `fetch_user(client, username, since, until=None)` signature consistent across Tasks 4–5.
- **Placeholder scan:** none — all steps carry full code and exact commands.
```
