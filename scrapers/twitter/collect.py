"""Collect + normalize tweets from twitterapi.io advanced_search."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .config import MAX_PAGES

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
