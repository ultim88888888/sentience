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
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).astype("datetime64[ns, UTC]")
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
         include_retweets: bool = True, persist: bool = True,
         out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Fetch all tweets since `since` for each user; optionally persist parquet each.

    Per-user failures are logged and skipped. Returns {username: DataFrame}.
    """
    since_dt = _parse_date(since) if isinstance(since, str) else since
    until_dt = (_parse_date(until) if isinstance(until, str) else until)
    key = api_key()
    limiter = RateLimiter(RATE_LIMIT_PER_SEC)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _one(client: TwitterAPI, user: str) -> tuple[str, pd.DataFrame]:
        raw = await collect.fetch_user(client, user, since_dt, until_dt,
                                       include_retweets=include_retweets)
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
    ap.add_argument("--no-retweets", action="store_true",
                    help="Skip the second pass that captures native retweets.")
    args = ap.parse_args()
    frames = pull(args.user, since=args.since, until=args.until,
                  include_retweets=not args.no_retweets)
    for user, df in frames.items():
        print(f"{user}: {len(df)} tweets", file=sys.stderr)


if __name__ == "__main__":
    main()
