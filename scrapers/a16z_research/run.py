"""Orchestrate the a16z research scrape: Algolia metadata -> article fetch -> extract -> parquet."""
import argparse
import asyncio
import sys
from datetime import datetime, timezone

import pandas as pd

from . import algolia, fetch, extract
from .config import ARTICLES_PARQUET, DATA_DIR, METADATA_PARQUET


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _merge_split_records(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per permalink, unioning multi-valued category/tag fields.

    Algolia indexes one record per (post, category) — the same post appears under multiple
    objectIDs (e.g. 2922-0/2922-1), each carrying a *different* category. Deduping naively
    would drop categories, so we union `categories` and `tags` across a permalink's records
    and keep the first of every other (identical) field.
    """
    union: dict[str, dict[str, list]] = {}
    for permalink, cats, tags in zip(df["permalink"], df["categories"], df["tags"]):
        acc = union.setdefault(permalink, {"categories": [], "tags": []})
        for col, vals in (("categories", cats), ("tags", tags)):
            for x in (vals if vals is not None else []):
                if x not in acc[col]:
                    acc[col].append(x)
    out = df.drop_duplicates(subset="permalink", keep="first").reset_index(drop=True).copy()
    out["categories"] = out["permalink"].map(lambda p: union[p]["categories"])
    out["tags"] = out["permalink"].map(lambda p: union[p]["tags"])
    return out


def collect_metadata() -> pd.DataFrame:
    _log("Minting Algolia key and paginating all post categories...")
    raw = algolia.fetch_raw_hits()
    records = [algolia.normalize_hit(h) for h in raw]
    df = pd.DataFrame.from_records(records)
    df = _merge_split_records(df)
    _log(f"  {len(df)} unique posts collected (across all categories).")
    return df


def collect_articles(permalinks: list[str]) -> pd.DataFrame:
    token = fetch.scrapedo_token()
    _log(f"Fetching {len(permalinks)} article pages via scrape.do (concurrency {fetch.CONCURRENCY})...")

    def progress(done, total):
        if done % 25 == 0 or done == total:
            _log(f"  fetched {done}/{total}")

    results = asyncio.run(fetch.fetch_all(permalinks, token=token, progress=progress))
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for res in results:
        ex = extract.extract(res.html, url=res.permalink)
        rows.append({
            "permalink": res.permalink,
            "fetched_url": res.fetched_url,
            "fetch_status": res.status,
            "fetch_error": res.error,
            "fetched_at": now,
            "raw_html": res.html,
            "raw_html_len": len(res.html),
            **ex,
        })
    df = pd.DataFrame.from_records(rows)
    ok = (df["fetch_status"] == 200).sum()
    empty = ((df["fetch_status"] == 200) & (df["extracted_text_len"] == 0)).sum()
    _log(f"  {ok}/{len(df)} fetched OK; {empty} fetched-but-empty extraction.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape a16zcrypto research articles to parquet.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N articles (smoke test).")
    ap.add_argument("--metadata-only", action="store_true",
                    help="Skip the article fetch/extract pass.")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meta = collect_metadata()
    if args.limit:
        meta = meta.head(args.limit).copy()
    meta.to_parquet(METADATA_PARQUET, index=False)
    _log(f"Wrote {METADATA_PARQUET} ({len(meta)} rows)")

    if args.metadata_only:
        return

    permalinks = [p for p in meta["permalink"].tolist() if p]
    articles = collect_articles(permalinks)
    # Join Algolia metadata with fetched/extracted content on permalink.
    joined = meta.merge(articles, on="permalink", how="left")
    joined.to_parquet(ARTICLES_PARQUET, index=False)
    _log(f"Wrote {ARTICLES_PARQUET} ({len(joined)} rows)")


if __name__ == "__main__":
    main()
