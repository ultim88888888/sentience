"""One-time extractive distillation of transcripts → verbatim, dated, stance-bearing
passages. EXTRACTIVE not abstractive: every passage must be a verbatim substring of
the source so the leakage audit (signals/audit.py) still works. Cached + resumable."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from doppelganger.llm import run_claude
from signals import config

_SYSTEM = """You extract VERBATIM stance-bearing passages from a transcript.

Rules:
- Output ONLY passages that express a view/stance on a crypto sector, token, or market
  regime (bullish/bearish/concerned/excited, a thesis, a prediction, a risk).
- Each passage MUST be copied VERBATIM from the transcript — do NOT paraphrase,
  summarize, or rewrite. Preserve enough surrounding context that the stance is
  interpretable on its own (what "it"/"that" refers to).
- Be CONSERVATIVE on dropping: when unsure whether something is a view, keep it.
  Be strict on verbatim: never alter wording.
- Skip pure filler, logistics, pleasantries, and off-topic chatter.

Output JSON only: {"passages": [{"date": "<YYYY-MM-DD>", "passage": "<verbatim text>"}]}
Use the provided publish date for every passage's date."""


def distill_one(*, object_id: str, title: str, transcript: str, post_date: str) -> list[dict]:
    user = (f"PUBLISH DATE: {post_date}\nTITLE: {title}\n\nTRANSCRIPT:\n{transcript}")
    raw = run_claude(_SYSTEM, user)
    obj = _extract_json(raw)
    passages = obj.get("passages", [])
    # Force the publish date (single-date doc) and keep only non-empty passages.
    return [{"date": post_date, "passage": p["passage"]}
            for p in passages if p.get("passage", "").strip()]


def build_distillate_cache(transcripts_path, articles, *, cache_path: Path | None = None,
                           post_dates: dict[str, str] | None = None) -> Path:
    """Distill each ok transcript once; append to a JSONL cache. Resumable: rows whose
    object_id already appears in the cache are skipped."""
    cache_path = Path(cache_path or config.DISTILLATE_CACHE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache_path.exists():
        done = {json.loads(l)["object_id"] for l in cache_path.read_text().splitlines() if l.strip()}

    tx = pd.read_parquet(transcripts_path)
    post_dates = post_dates or _join_post_dates(tx, articles)

    with cache_path.open("a") as fh:
        for _, row in tx.iterrows():
            oid = row["object_id"]
            if oid in done:
                continue
            if row.get("status") != "ok" or not str(row.get("transcript", "")).strip():
                continue
            try:
                passages = distill_one(object_id=oid, title=row.get("title", ""),
                                       transcript=row["transcript"],
                                       post_date=post_dates.get(oid, ""))
            except Exception as e:  # malformed LLM JSON etc. — skip this doc, don't crash the batch
                print(f"[distill skip] {oid}: {e}")
                passages = []
            fh.write(json.dumps({"object_id": oid, "passages": passages}) + "\n")
            fh.flush()
    return cache_path


def build_article_distillate_cache(articles_path, *, cache_path: Path | None = None,
                                   since: str = "2021-01-01", min_chars: int = 500) -> Path:
    """Distill each in-range article's extracted_text once (extractive, verbatim) → JSONL cache.
    Same firewall-preserving mechanism as transcripts; articles in aggregate are too large to
    feed verbatim, so A1 reads these distillates instead of full bodies. Resumable."""
    cache_path = Path(cache_path or (config.SIGNAL_OUT_DIR / "article_distillates.jsonl"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache_path.exists():
        done = {json.loads(l)["object_id"] for l in cache_path.read_text().splitlines() if l.strip()}

    arts = articles_path if isinstance(articles_path, pd.DataFrame) else pd.read_parquet(articles_path)
    arts = arts.copy()
    arts["_pd"] = pd.to_datetime(arts["post_date"], utc=True, errors="coerce")
    floor = pd.Timestamp(since, tz="UTC")
    with cache_path.open("a") as fh:
        for _, row in arts.iterrows():
            oid = str(row["object_id"])
            if oid in done:
                continue
            body = str(row.get("extracted_text") or "")
            if pd.isnull(row["_pd"]) or row["_pd"] < floor or len(body.strip()) < min_chars:
                continue
            pdate = row["_pd"].date().isoformat()
            try:
                passages = distill_one(object_id=oid, title=str(row.get("title", "")),
                                       transcript=body, post_date=pdate)
            except Exception as e:
                print(f"[distill skip] {oid}: {e}")
                passages = []
            fh.write(json.dumps({"object_id": oid, "passages": passages}) + "\n")
            fh.flush()
    return cache_path


def load_distillates(cache_path: Path | None = None) -> dict[str, list[dict]]:
    """object_id -> list of {date, passage}."""
    cache_path = Path(cache_path or config.DISTILLATE_CACHE)
    if not cache_path.exists():
        return {}
    out = {}
    for line in cache_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["object_id"]] = d["passages"]
    return out


_TWEET_SYS = ('You are filtering tweets from crypto investors to ONLY those that express a substantive view '
 'on a crypto SECTOR, TOKEN, or the MARKET (bullish/bearish/concern/thesis/prediction/risk). '
 'Drop personal chatter, jokes, logistics, replies-about-nothing, generic hype with no specific subject. '
 'Keep each kept tweet VERBATIM with its date. This is content filtering, not advice. '
 'Output JSON only: {"kept":[{"date":"YYYY-MM-DD","text":"<verbatim>"}]}')


def distill_tweet_batch(tweets: list) -> list[dict]:
    """tweets: list of (iso_date, text). Returns kept [{date,text}] (verbatim, trade-relevant)."""
    payload = "\n".join(f"[{d}] {t}" for d, t in tweets)
    out = _extract_json(run_claude(_TWEET_SYS, payload)).get("kept", [])
    return [{"date": k["date"], "text": k["text"]} for k in out
            if isinstance(k, dict) and k.get("date") and k.get("text")]


def build_tweet_distillate_cache(twitter_paths, *, cache_path=None, since: str = "2021-01-01",
                                 batch_chars: int = 130000) -> Path:
    """Distill each handle's substantive, in-range tweets in batches; cache to JSONL. Resumable
    at the (handle, batch) level. Cache line: {"handle":..., "batch":i, "kept":[{date,text}]}."""
    import pandas as pd
    from signals.corpus import is_substantive_tweet
    cache_path = Path(cache_path or config.TWEET_DISTILLATE_CACHE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done.add((d["handle"], d["batch"]))
    floor = pd.Timestamp(since, tz="UTC")
    with cache_path.open("a") as fh:
        for p in twitter_paths:
            handle = Path(p).stem
            df = pd.read_parquet(p, columns=["created_at", "type", "text"])
            df = df[df["type"] != "retweet"]
            rows = []
            for _, r in df.iterrows():
                if r["created_at"] >= floor and is_substantive_tweet(r["text"]):
                    rows.append((r["created_at"].date().isoformat(), str(r["text"]).strip()))
            rows.sort()
            # batch by char budget
            batches, cur, n = [], [], 0
            for row in rows:
                if n + len(row[1]) > batch_chars and cur:
                    batches.append(cur); cur, n = [], 0
                cur.append(row); n += len(row[1])
            if cur: batches.append(cur)
            for i, b in enumerate(batches):
                if (handle, i) in done:
                    continue
                try:
                    kept = distill_tweet_batch(b)
                except Exception as e:
                    print(f"[tweet-distill skip] {handle} batch {i}: {e}"); kept = []
                fh.write(json.dumps({"handle": handle, "batch": i, "kept": kept}) + "\n"); fh.flush()
    return cache_path


def load_tweet_distillates(cache_path=None) -> list:
    """Flat MEMBER-AGNOSTIC list of (iso_date, text) across all handles (for A1)."""
    cache_path = Path(cache_path or config.TWEET_DISTILLATE_CACHE)
    if not cache_path.exists():
        return []
    out = []
    for l in cache_path.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            out += [(k["date"], k["text"]) for k in d["kept"]]
    return out


def _join_post_dates(tx: pd.DataFrame, articles) -> dict[str, str]:
    if articles is None:
        return {}
    arts = articles if isinstance(articles, pd.DataFrame) else pd.read_parquet(articles)
    # articles join key may be object_id or object_id_join depending on source
    key = "object_id" if "object_id" in arts.columns else "object_id_join"
    return dict(zip(arts[key].astype(str), arts["post_date"].astype(str)))


def _extract_json(raw: str) -> dict:
    """Mirror doppelganger/respond.py: pull the first JSON object, tolerate ``` fences."""
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return {"passages": []}
    return json.loads(s[start:end + 1])
