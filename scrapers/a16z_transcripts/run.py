"""Collect full transcripts for every a16z video/podcast post -> transcripts.parquet.

  python -m scrapers.a16z_transcripts.run                 # full run
  python -m scrapers.a16z_transcripts.run --youtube-only  # skip audio/whisper leg
  python -m scrapers.a16z_transcripts.run --limit 5       # smoke test (per source)

Output ``data/a16z_research/transcripts.parquet`` is a join table keyed on ``object_id``;
the read-only corpus parquet is never mutated.
"""
import argparse
import datetime as dt

import pandas as pd

from .config import CORPUS_PARQUET, TRANSCRIPTS_PARQUET
from .sources import NONE, WHISPER, YOUTUBE, route_corpus


def _log(msg: str) -> None:
    print(msg, flush=True)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _existing_ok() -> dict[str, dict]:
    """Map object_id -> prior row dict for transcripts already fetched OK (resume support)."""
    if not TRANSCRIPTS_PARQUET.exists():
        return {}
    prev = pd.read_parquet(TRANSCRIPTS_PARQUET)
    ok = prev[prev["status"] == "ok"]
    return {row["object_id"]: row.to_dict() for _, row in ok.iterrows()}


def build(limit: int | None = None, youtube_only: bool = False, resume: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(CORPUS_PARQUET)
    routes = route_corpus(df)
    n_yt = sum(r.source == YOUTUBE for r in routes)
    n_wh = sum(r.source == WHISPER for r in routes)
    n_no = sum(r.source == NONE for r in routes)
    _log(f"Routed {len(routes)} posts: {n_yt} youtube, {n_wh} whisper, {n_no} unroutable.")

    done = _existing_ok() if resume else {}
    if done:
        _log(f"Resume: {len(done)} transcripts already OK — skipping those.")

    yt_routes = [r for r in routes if r.source == YOUTUBE and r.object_id not in done]
    wh_routes = [r for r in routes if r.source == WHISPER and r.object_id not in done]
    if limit:
        yt_routes, wh_routes = yt_routes[:limit], wh_routes[:limit]

    # ── YouTube caption leg ──────────────────────────────────────────────────────────
    from .youtube import fetch_many
    _log(f"Fetching YouTube captions for {len(yt_routes)} posts...")
    yt_by_id = fetch_many([r.media_id for r in yt_routes], log=_log)

    # ── Audio / whisper leg ──────────────────────────────────────────────────────────
    wh_by_oid: dict[str, dict] = {}
    if youtube_only:
        _log("Skipping audio/whisper leg (--youtube-only).")
    else:
        from .audio import transcribe_podcasts
        _log(f"Transcribing {len(wh_routes)} audio-only podcasts (mlx-whisper)...")
        wh_by_oid = transcribe_podcasts(wh_routes, log=_log)

    # ── Assemble rows ────────────────────────────────────────────────────────────────
    rows = []
    fetched_at = _now()
    for r in routes:
        if r.object_id in done:
            rows.append(done[r.object_id])
            continue
        if r.source == YOUTUBE:
            res = yt_by_id.get(r.media_id, {})
        elif r.source == WHISPER:
            res = wh_by_oid.get(r.object_id, {})
        else:
            res = {"text": "", "lang": None, "status": "unroutable", "error": r.note}
        text = res.get("text", "")
        rows.append({
            "object_id": r.object_id,
            "title": r.title,
            "format": r.fmt,
            "source": r.source,
            "media_id": r.media_id,
            "transcript": text,
            "transcript_len": len(text),
            "lang": res.get("lang"),
            "status": res.get("status", "skipped"),
            "error": res.get("error"),
            "fetched_at": fetched_at,
        })
    out = pd.DataFrame.from_records(rows)
    ok = (out["status"] == "ok").sum()
    chars = int(out["transcript_len"].sum())
    _log(f"Done: {ok}/{len(out)} transcripts OK, {chars:,} chars total.")
    # Surface anything that didn't land cleanly — no silent gaps.
    bad = out[out["status"] != "ok"]
    if len(bad):
        _log(f"Non-OK ({len(bad)}):")
        for _, b in bad.iterrows():
            _log(f"  - {b['status']:12} {b['format']:8} {b['title'][:50]!r} {b['error'] or ''}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect a16z video/podcast transcripts.")
    ap.add_argument("--limit", type=int, default=None, help="First N per source (smoke test).")
    ap.add_argument("--youtube-only", action="store_true", help="Skip the audio/whisper leg.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-fetch everything, ignoring already-OK rows in the output.")
    args = ap.parse_args()

    out = build(limit=args.limit, youtube_only=args.youtube_only, resume=not args.no_resume)
    TRANSCRIPTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TRANSCRIPTS_PARQUET, index=False)
    _log(f"Wrote {TRANSCRIPTS_PARQUET} ({len(out)} rows)")


if __name__ == "__main__":
    main()
