"""Priority + checkpointed transcribe.

Processes posts by given authors FIRST, and writes transcripts.parquet after EVERY post so
an unattended/overnight run is crash-safe and resumable (the base `run` writes only at the
end). Reuses the same caption/whisper/scrape.do machinery.

  python -m scrapers.a16z_transcripts.priority --priority-author eddy-lazzarin
  python -m scrapers.a16z_transcripts.priority --priority-author eddy-lazzarin --limit 2
"""
import argparse
import datetime as dt

import pandas as pd

from .config import AUDIO_CACHE, CORPUS_PARQUET, TRANSCRIPTS_PARQUET
from .sources import route_corpus, WHISPER, YOUTUBE


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _log(m: str) -> None:
    print(m, flush=True)


def _has_author(a, who) -> bool:
    try:
        return who in list(a)
    except TypeError:
        return False


def _checkpoint(rows_by_id: dict) -> None:
    """Atomic write of the full transcript table after each post."""
    df = pd.DataFrame(list(rows_by_id.values()))
    tmp = TRANSCRIPTS_PARQUET.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(TRANSCRIPTS_PARQUET)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority-author", action="append", default=[],
                    help="Roster slug whose posts transcribe first (repeatable).")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N to-do posts.")
    args = ap.parse_args()

    corpus = pd.read_parquet(CORPUS_PARQUET)
    prio_ids = set()
    for who in args.priority_author:
        prio_ids |= set(corpus[corpus["author_slugs"].apply(lambda a: _has_author(a, who))]["object_id"])

    routes = [r for r in route_corpus(corpus) if r.source in (YOUTUBE, WHISPER)]
    routes.sort(key=lambda r: 0 if r.object_id in prio_ids else 1)  # priority first; stable

    rows = {}
    if TRANSCRIPTS_PARQUET.exists():
        rows = {r["object_id"]: r.to_dict() for _, r in pd.read_parquet(TRANSCRIPTS_PARQUET).iterrows()}
    done_ok = {oid for oid, r in rows.items() if r.get("status") == "ok"}
    todo = [r for r in routes if r.object_id not in done_ok]
    if args.limit:
        todo = todo[:args.limit]
    n_prio = sum(1 for r in todo if r.object_id in prio_ids)
    _log(f"{len(routes)} transcribable, {len(done_ok)} already ok, {len(todo)} to do "
         f"({n_prio} priority first).")

    yt_api, token = None, None
    for i, r in enumerate(todo, 1):
        prio = "PRIO " if r.object_id in prio_ids else ""
        try:
            if r.source == YOUTUBE:
                if yt_api is None:
                    from youtube_transcript_api import YouTubeTranscriptApi
                    from .proxy import proxied_session
                    yt_api = YouTubeTranscriptApi(http_client=proxied_session(residential=True))
                from .youtube import fetch_one
                res = fetch_one(yt_api, r.media_id)
            else:  # WHISPER
                from .audio import resolve_audio_url, download, transcribe_file
                from .proxy import scrapedo_token
                if token is None:
                    token = scrapedo_token()
                url = resolve_audio_url(r.media_id, token=token)
                if not url:
                    res = {"text": "", "lang": None, "status": "no_audio", "error": "no enclosure_url"}
                else:
                    dest = AUDIO_CACHE / f"{r.media_id}.mp3"
                    download(url, dest, token=token)
                    res = transcribe_file(dest)
        except Exception as e:
            res = {"text": "", "lang": None, "status": "error", "error": repr(e)}
        text = res.get("text", "")
        rows[r.object_id] = {
            "object_id": r.object_id, "title": r.title, "format": r.fmt, "source": r.source,
            "media_id": r.media_id, "transcript": text, "transcript_len": len(text),
            "lang": res.get("lang"), "status": res.get("status", "skipped"),
            "error": res.get("error"), "fetched_at": _now(),
        }
        _checkpoint(rows)  # crash-safe: persist after every post
        _log(f"  [{i}/{len(todo)}] {prio}{r.source:7} {res.get('status'):11} "
             f"({len(text):>6} chars) {r.title[:42]!r}")
    _log("priority transcribe complete.")


if __name__ == "__main__":
    main()
