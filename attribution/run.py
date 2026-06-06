"""Run the attribution pipeline end-to-end -> segments parquet, per-person corpora, report.

  python -m attribution.run --pilot          # seminar host + cached podcasts only
  python -m attribution.run                   # full corpus (deferred until pilot clears)
"""
import argparse
import datetime as dt
import glob

import pandas as pd

from .config import AUDIO_CACHE, CORPUS, HOST_PRIOR, REPORT_OUT, SEGMENTS_OUT, TRANSCRIPTS
from .route import route_post, CONVERSATIONAL
from .roster import Roster
from .gate import gate_segment

def _log(m): print(m, flush=True)

def _participants(row, roster: Roster) -> list[str]:
    """Known participant names for the LLM prompt: guest author(s) by slug."""
    names = []
    raw = row.get("author_slugs")  # may be a numpy array — avoid `or` (ambiguous truthiness)
    try:
        slugs = [s for s in list(raw) if s] if raw is not None else []
    except TypeError:
        slugs = []
    for slug in slugs:
        hit = roster._df[roster._df["slug"] == slug]
        if len(hit):
            names.append(hit.iloc[0]["name"])
    return names

def _both_engines(row, mp3, parts, roster) -> list[dict]:
    """Corrected both-engines path (index-aligned fusion). Diarization fixes the segmentation,
    the LLM names each segment, fuse_by_voice resolves voice->identity by majority with
    intra-voice consistency as confidence. See docs/.../both-engines-fusion-redesign.md."""
    from .diarize import diarize_audio, transcribe_timestamped, assign_segments_to_turns
    from .attribute_text import attribute_segments
    from .fuse import fuse_by_voice
    turns = diarize_audio(mp3)
    voiced = assign_segments_to_turns(transcribe_timestamped(mp3), turns)
    names = attribute_segments(voiced, parts)
    prior = HOST_PRIOR["podcast"]
    fused, _, _ = fuse_by_voice(voiced, names, lambda n: roster.resolve(n, prefer=prior).slug)
    out = []
    for s in fused:
        ident = s["identity"]
        speaker = roster.name_for(ident) or (ident.title() if ident else "UNKNOWN")
        out.append({"text": s["text"], "speaker": speaker, "method": "fused",
                    "agreed": s["agree"], "confidence": s["confidence"]})
    return out


def _segments_for_post(row, transcript, roster, mode, diarize: bool = False) -> list[dict]:
    from .attribute_text import attribute
    parts = _participants(row, roster)
    media_id = row.get("media_id")
    mp3 = str(AUDIO_CACHE / f"{media_id}.mp3") if media_id else None
    if diarize and mode == CONVERSATIONAL and mp3 and glob.glob(mp3):
        try:
            return _both_engines(row, mp3, parts, roster)
        except Exception as e:  # gated model / decode error -> degrade to text-only, logged
            _log(f"  diarization failed for {row.get('object_id')} ({type(e).__name__}) -> text-only")
    llm = attribute(transcript, parts)  # text-LLM path (default + fallback)
    for s in llm:
        s["method"] = "text"
    return llm

def build(pilot: bool, limit: int | None = None, diarize: bool = False) -> pd.DataFrame:
    corpus = pd.read_parquet(CORPUS)
    tx = pd.read_parquet(TRANSCRIPTS)
    tx = tx[tx["status"] == "ok"]
    roster = Roster.load()
    merged = tx.merge(corpus[["object_id", "formats", "author_slugs", "title"]],
                      on="object_id", how="left", suffixes=("", "_corpus"))
    if pilot:
        # videos first (the seminar-host target; no audio needed, faster), then podcasts.
        merged = pd.concat([
            merged[merged["formats"].astype(str).str.contains("video", na=False)].head(20),
            merged[merged["formats"].astype(str).str.contains("podcast", na=False)],
        ])
    if limit:
        merged = merged.head(limit)
    rows = []
    for _, row in merged.iterrows():
        mode = route_post(row).mode
        # Host-prior: which a16z members host this format (disambiguates first-name thanks).
        prior = HOST_PRIOR["podcast"] if mode == CONVERSATIONAL else HOST_PRIOR["research-seminar"]
        try:
            segs = _segments_for_post(row, row["transcript"], roster, mode, diarize=diarize)
        except Exception as e:
            _log(f"  attribution_failed {row['object_id']}: {e!r}")
            continue
        for i, s in enumerate(segs):
            m = roster.resolve(s.get("speaker"), prefer=prior)
            kept, reason = gate_segment(s)
            rows.append({
                "post_id": row["object_id"], "segment_idx": i, "method": s["method"],
                "speaker": s.get("speaker"), "slug": m.slug, "is_a16z": m.is_a16z,
                "confidence": s.get("confidence"), "agreed": s.get("agreed"),
                "kept": kept, "dropped_reason": reason, "text": s["text"],
            })
    return pd.DataFrame.from_records(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Cap to first N posts (fast pilot).")
    ap.add_argument("--diarize", action="store_true",
                    help="Both-engines for podcasts (pyannote+LLM fusion). Default off (text-LLM).")
    args = ap.parse_args()
    df = build(pilot=args.pilot, limit=args.limit, diarize=args.diarize)
    SEGMENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SEGMENTS_OUT, index=False)
    # Canonical deliverable: per-post tagged-transcript records + metadata (jsonl). No
    # per-person files — the doppelganger engine matches/assembles from the jsonl itself.
    from .records import export as export_records
    n = export_records(df) if len(df) else 0
    kept = int(df["kept"].sum()) if len(df) else 0
    a16z_kept = (df[(df["is_a16z"] == True) & (df["kept"] == True)].groupby("slug").size()  # noqa: E712
                 .sort_values(ascending=False)) if len(df) else {}
    with open(REPORT_OUT, "w") as f:
        f.write(f"# Attribution report ({dt.datetime.now(dt.timezone.utc).isoformat()})\n\n")
        f.write(f"posts: {df['post_id'].nunique() if len(df) else 0} | segments: {len(df)} | "
                f"kept: {kept} | dropped: {len(df)-kept}\n")
        f.write(f"canonical deliverable: attributed_transcripts.jsonl ({n} post records)\n\n")
        f.write("## a16z kept segments by person\n")
        for slug, c in (a16z_kept.items() if len(df) else []):
            f.write(f"- {slug}: {c}\n")
    _log(f"Wrote {SEGMENTS_OUT} ({len(df)} segs, {kept} kept) + attributed_transcripts.jsonl ({n})")

if __name__ == "__main__":
    main()
