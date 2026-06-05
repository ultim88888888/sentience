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
