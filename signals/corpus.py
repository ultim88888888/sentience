"""Assemble the as-of-T blended corpus over a trailing holding-period window.
Uniform window across sources (spec stage 2). Tweets + articles verbatim;
transcripts come from the distillate cache (signals/distill.py)."""
from __future__ import annotations
import re
from datetime import date
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta


def is_substantive_tweet(text: str) -> bool:
    """Drop low-substance tweets: strip URLs + @mentions, require >= 50 chars left.
    Mirrors the doppelganger twitter adapter's reply filter."""
    t = re.sub(r"https?://\S+", "", str(text))
    t = re.sub(r"@\w+", "", t)
    return len(t.strip()) >= 50


def in_window(d: str | date, t: date, window_months: int) -> bool:
    dd = pd.to_datetime(d).date() if not isinstance(d, date) else d
    start = t - relativedelta(months=window_months)
    return start < dd <= t


def assemble_corpus(*, t: date, window_months: int, twitter_paths: list[Path],
                    articles, distillates: dict[str, list[dict]],
                    article_distillates: dict[str, list[dict]] | None = None,
                    tweet_distillates: list | None = None) -> str:
    """Return one chronological, source-tagged text block of all in-window evidence.

    `distillates` = transcript distillates (object_id -> passages). `article_distillates`,
    when provided, makes research articles contribute their DISTILLED passages instead of
    full bodies (token control for the blended A1 corpus); when None, full `extracted_text`
    is used (small corpora / tests). `tweet_distillates`, when provided (A1 path), is a
    flat member-agnostic list of (iso_date, text) used INSTEAD of reading raw tweets from
    `twitter_paths`; when None the existing raw-tweet loop runs unchanged (A2 path)."""
    rows: list[tuple[str, str, str]] = []  # (iso_date, source_tag, text)

    if tweet_distillates is not None:
        # A1 path: use pre-distilled tweets (member-agnostic, AUP-clean)
        for d, text in tweet_distillates:
            if in_window(d, t, window_months):
                rows.append((d if isinstance(d, str) else d.isoformat(), "x", str(text).strip()))
    else:
        # A2 path (and default): raw tweets, drop retweets + low-substance
        for p in twitter_paths:
            tw = pd.read_parquet(p)
            tw = tw[tw["type"] != "retweet"]
            for _, r in tw.iterrows():
                d = r["created_at"].date()
                if in_window(d, t, window_months):
                    if not is_substantive_tweet(r["text"]):
                        continue
                    rows.append((d.isoformat(), "x", str(r["text"]).strip()))

    # Research articles: distilled passages (A1) or full extracted_text (default)
    arts = articles if isinstance(articles, pd.DataFrame) else (
        pd.read_parquet(articles) if articles is not None else pd.DataFrame())
    for _, r in arts.iterrows():
        if not in_window(r["post_date"], t, window_months):
            continue
        oid = str(r.get("object_id") or "")
        if article_distillates is not None:
            for passage in article_distillates.get(oid, []):
                if in_window(passage["date"], t, window_months):
                    rows.append((passage["date"], "research", passage["passage"].strip()))
        else:
            body = str(r.get("extracted_text") or "").strip()
            if body:
                rows.append((pd.to_datetime(r["post_date"]).date().isoformat(), "research", body))

    # Distilled transcript passages (keyed by article object_id, dated by passage)
    for _, r in arts.iterrows():
        oid = str(r.get("object_id") or "")
        for passage in distillates.get(oid, []):
            if in_window(passage["date"], t, window_months):
                rows.append((passage["date"], "transcript", passage["passage"].strip()))

    rows.sort(key=lambda x: x[0])
    return "\n".join(f"[{d}] ({tag}) {txt}" for d, tag, txt in rows)
