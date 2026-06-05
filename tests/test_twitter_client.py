"""TDD tests for scrapers.twitter — twitterapi.io client.

Execution: .venv/bin/python -m pytest tests/test_twitter_client.py -v
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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


def collect_page(tweets, has_next, cursor):
    from scrapers.twitter.client import SearchPage
    return SearchPage(tweets=tweets, has_next_page=has_next, next_cursor=cursor)


# ── normalize ────────────────────────────────────────────────────────────────

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


# ── client ───────────────────────────────────────────────────────────────────

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


# ── fetch_user ───────────────────────────────────────────────────────────────

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


# ── run ──────────────────────────────────────────────────────────────────────

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
