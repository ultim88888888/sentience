"""Orchestrate the a16z team scrape: roster page -> per-member profile fetch -> extract -> parquet.

Reuses the research scraper's scrape.do fetch layer (token, retry, concurrency) so the
team scrape goes through the same proxy. Two network stages, both via scrape.do:
  1. GET the roster page, parse it into members (slug, name, listing title, sections).
  2. GET each /team/<slug> profile, extract the authoritative title, bio, and socials.
"""
import argparse
import asyncio
import sys
from datetime import datetime, timezone

import pandas as pd

from scrapers.a16z_research import fetch  # shared scrape.do fetch layer
from . import extract, listing
from .config import DATA_DIR, PROFILE_BASE, TEAM_PARQUET, TEAM_URL

# Normalized platform columns broken out for easy querying / joins.
_PLATFORM_COLS = ["x", "linkedin", "farcaster", "github", "website"]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def collect_roster(token: str) -> list[listing.Member]:
    _log(f"Fetching team roster {TEAM_URL} via scrape.do ...")
    res = asyncio.run(fetch.fetch_all([TEAM_URL], token=token))[0]
    if res.status != 200 or not res.html:
        raise RuntimeError(f"roster fetch failed: HTTP {res.status} {res.error or ''}")
    members = listing.parse_roster(res.html)
    _log(f"  {len(members)} unique members on the roster.")
    return members


def collect_profiles(members: list[listing.Member], token: str) -> pd.DataFrame:
    urls = [PROFILE_BASE + m.slug for m in members]
    _log(f"Fetching {len(urls)} profile pages via scrape.do (concurrency {fetch.CONCURRENCY})...")

    def progress(done, total):
        if done % 20 == 0 or done == total:
            _log(f"  fetched {done}/{total}")

    results = asyncio.run(fetch.fetch_all(urls, token=token, progress=progress))
    by_url = {r.permalink: r for r in results}
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for m in members:
        url = PROFILE_BASE + m.slug
        res = by_url[url]
        prof = extract.extract(res.html)
        # First URL per platform (members rarely list two of the same).
        platform_url = {}
        for s in prof.socials:
            platform_url.setdefault(s["platform"], s["url"])
        rows.append({
            "slug": m.slug,
            "name": prof.name or m.name,
            "title": prof.title or m.listing_title,
            "listing_title": m.listing_title,
            "sections": m.sections,
            "bio": prof.bio,
            "bio_len": len(prof.bio or ""),
            "socials_json": extract.socials_json(prof.socials),
            "socials_count": len(prof.socials),
            **{f"{p}_url": platform_url.get(p) for p in _PLATFORM_COLS},
            "profile_url": url,
            "fetch_status": res.status,
            "fetch_error": res.error,
            "fetched_at": now,
            "raw_html": res.html,
            "raw_html_len": len(res.html),
        })
    df = pd.DataFrame.from_records(rows)
    ok = (df["fetch_status"] == 200).sum()
    nobio = ((df["fetch_status"] == 200) & (df["bio_len"] == 0)).sum()
    _log(f"  {ok}/{len(df)} fetched OK; {nobio} with no bio; "
         f"{int(df['socials_count'].gt(0).sum())} have socials.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape the a16zcrypto team roster to parquet.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N members (smoke test).")
    ap.add_argument("--roster-only", action="store_true",
                    help="Parse the roster and stop before fetching profiles.")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    token = fetch.scrapedo_token()

    members = collect_roster(token)
    if args.limit:
        members = members[:args.limit]
    if args.roster_only:
        for m in members:
            _log(f"  {m.slug:32} {m.name:28} {m.listing_title!r} {m.sections}")
        return

    df = collect_profiles(members, token)
    df.to_parquet(TEAM_PARQUET, index=False)
    _log(f"Wrote {TEAM_PARQUET} ({len(df)} rows)")


if __name__ == "__main__":
    main()
