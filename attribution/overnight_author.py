"""Overnight, overlap-friendly attribution for a single author (e.g. eddy-lazzarin).

Attributes each of the author's posts AS its transcript becomes available (the transcribe
driver runs concurrently), checkpointing after every post and rebuilding the author's corpus.
Self-terminates when every transcribable post for the author is either attributed or has a
terminal transcript failure. Crash-safe and resumable: re-running skips already-attributed
posts.

  python -m attribution.overnight_author --author eddy-lazzarin
"""
import argparse
import datetime as dt
import time

import pandas as pd

from .config import CORPUS, REPORT_OUT, SEGMENTS_OUT, TRANSCRIPTS, HOST_PRIOR
from .gate import gate_segment
from .roster import Roster
from .route import route_post, CONVERSATIONAL
from .run import _participants, _segments_for_post

POLL_SECONDS = 240


def _log(m: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {m}", flush=True)


def _author_post_ids(author: str) -> set:
    corpus = pd.read_parquet(CORPUS)
    def has(a):
        try:
            return author in list(a)
        except TypeError:
            return False
    vp = corpus[corpus["formats"].astype(str).str.contains("video|podcast", na=False)]
    return set(vp[vp["author_slugs"].apply(has)]["object_id"])


def _load_attributed_ids() -> set:
    if SEGMENTS_OUT.exists():
        return set(pd.read_parquet(SEGMENTS_OUT)["post_id"].unique())
    return set()


def _append_segments(new_rows: list[dict]) -> None:
    prev = pd.read_parquet(SEGMENTS_OUT) if SEGMENTS_OUT.exists() else pd.DataFrame()
    out = pd.concat([prev, pd.DataFrame(new_rows)], ignore_index=True)
    tmp = SEGMENTS_OUT.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.rename(SEGMENTS_OUT)


def _rebuild_outputs(author: str) -> None:
    """The ONLY deliverable is the per-post tagged-transcript dataset (jsonl) + a stats report.
    No per-person files — the doppelganger engine matches/assembles from the jsonl itself."""
    seg = pd.read_parquet(SEGMENTS_OUT)
    from .records import export as export_records
    n = export_records(seg)  # canonical: attributed_transcripts.jsonl
    kept = int(seg["kept"].sum())
    a16z_kept = (seg[(seg["is_a16z"] == True) & (seg["kept"] == True)]  # noqa: E712
                 .groupby("slug").size().sort_values(ascending=False))
    with open(REPORT_OUT, "w") as f:
        f.write(f"# Attribution report ({dt.datetime.now(dt.timezone.utc).isoformat()})\n\n")
        f.write(f"priority author: {author}\n")
        f.write(f"posts: {seg['post_id'].nunique()} | segments: {len(seg)} | "
                f"kept: {kept} | dropped: {len(seg)-kept}\n")
        f.write(f"canonical deliverable: attributed_transcripts.jsonl ({n} post records)\n\n")
        f.write("## a16z kept segments by person (who shows up, how much)\n")
        for slug, c in a16z_kept.items():
            f.write(f"- {slug}: {c}\n")


def _attribute_post(row, roster) -> list[dict]:
    mode = route_post(row).mode
    prior = HOST_PRIOR["podcast"] if mode == CONVERSATIONAL else HOST_PRIOR["research-seminar"]
    segs = _segments_for_post(row, row["transcript"], roster, mode)
    rows = []
    for i, s in enumerate(segs):
        m = roster.resolve(s.get("speaker"), prefer=prior)
        kept, reason = gate_segment(s)
        rows.append({
            "post_id": row["object_id"], "segment_idx": i, "method": s["method"],
            "speaker": s.get("speaker"), "slug": m.slug, "is_a16z": m.is_a16z,
            "confidence": s.get("confidence"), "agreed": s.get("agreed"),
            "kept": kept, "dropped_reason": reason, "text": s["text"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", required=True, help="Roster slug to attribute (e.g. eddy-lazzarin).")
    args = ap.parse_args()
    roster = Roster.load()
    target = _author_post_ids(args.author)
    corpus = pd.read_parquet(CORPUS)[["object_id", "formats", "author_slugs", "title"]]
    _log(f"author {args.author}: {len(target)} transcribable posts to attribute.")

    fail_count: dict[str, int] = {}
    give_up: set = set()           # posts that failed attribution past the cap — stop retrying
    MAX_ATTR_RETRIES = 3

    while True:
        attributed = _load_attributed_ids()
        tx = pd.read_parquet(TRANSCRIPTS)
        tx_ok = tx[tx["status"] == "ok"]
        ready = tx_ok[tx_ok["object_id"].isin(target - attributed - give_up)]
        terminal = (set(tx[tx["status"].isin(["error", "unroutable", "no_audio", "unavailable",
                                             "no_captions"])]["object_id"]) & target) | give_up
        if len(ready) == 0:
            remaining = target - attributed - terminal
            if not remaining:
                _log(f"all {len(target)} posts done (attributed or terminal). finishing.")
                break
            _log(f"waiting for transcripts: {len(remaining)} of {len(target)} not ready yet.")
            time.sleep(POLL_SECONDS)
            continue
        merged = ready.merge(corpus, on="object_id", how="left", suffixes=("", "_c"))
        for _, row in merged.iterrows():
            try:
                rows = _attribute_post(row, roster)
                _append_segments(rows)
                _rebuild_outputs(args.author)
                kept = sum(r["kept"] for r in rows)
                _log(f"  attributed {row['object_id']} ({len(rows)} segs, {kept} kept) "
                     f"{str(row['title'])[:42]!r}")
            except Exception as e:
                oid = row["object_id"]
                fail_count[oid] = fail_count.get(oid, 0) + 1
                if fail_count[oid] >= MAX_ATTR_RETRIES:
                    give_up.add(oid)
                    _log(f"  GAVE UP on {oid} after {fail_count[oid]} attempts: {type(e).__name__}")
                else:
                    _log(f"  FAILED {oid} (attempt {fail_count[oid]}/{MAX_ATTR_RETRIES}): "
                         f"{type(e).__name__}: {str(e)[:120]}")
    _log("overnight author attribution complete.")


if __name__ == "__main__":
    main()
