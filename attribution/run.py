"""Run the attribution pipeline end-to-end -> segments parquet, per-person corpora, report.

  python -m attribution.run --pilot          # seminar host + cached podcasts only
  python -m attribution.run                   # full corpus (deferred until pilot clears)
"""
import argparse
import datetime as dt
import glob

import pandas as pd

from .config import (AUDIO_CACHE, CORPUS, HOST_PRIOR, PERSONS_DIR, REPORT_OUT, SEGMENTS_OUT,
                     TRANSCRIPTS)
from .route import route_post, CONVERSATIONAL
from .roster import Roster
from .gate import gate_segment
from .assemble import build_person_corpus

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

def _segments_for_post(row, transcript, roster, mode) -> list[dict]:
    from .attribute_text import attribute
    parts = _participants(row, roster)
    llm = attribute(transcript, parts)  # [{speaker,text,confidence}]
    if mode != CONVERSATIONAL:
        for s in llm:
            s["method"] = "text"
        return llm
    # conversational: locate THIS post's cached mp3 by media_id, then diarize + fuse.
    media_id = row.get("media_id")
    mp3 = str(AUDIO_CACHE / f"{media_id}.mp3") if media_id else None
    if not mp3 or not glob.glob(mp3):
        _log(f"  no cached audio for {row.get('object_id')} (media_id={media_id}) -> text-only")
        for s in llm:
            s["method"] = "text"
        return llm
    try:
        from .diarize import diarize_audio, transcribe_timestamped, assign_segments_to_turns
        from .fuse import name_voices, fuse_segments
        turns = diarize_audio(mp3)
        tsegs = transcribe_timestamped(mp3)
        voiced = assign_segments_to_turns(tsegs, turns)
        llm_by_text = {s["text"]: s["speaker"] for s in llm}
        fused = fuse_segments(voiced, name_voices(voiced, llm_by_text), llm_by_text)
        for s in fused:
            s["method"] = "fused"
            s["confidence"] = 0.9 if s["agreed"] else 0.4
        return fused
    except Exception as e:
        # Per spec §5: diarization unavailable (gated model, decode error) -> degrade to
        # text-only rather than drop the post. Logged, not silent.
        _log(f"  diarization failed for {row.get('object_id')} ({type(e).__name__}) -> text-only")
        for s in llm:
            s["method"] = "text"
        return llm

def build(pilot: bool) -> pd.DataFrame:
    corpus = pd.read_parquet(CORPUS)
    tx = pd.read_parquet(TRANSCRIPTS)
    tx = tx[tx["status"] == "ok"]
    roster = Roster.load()
    merged = tx.merge(corpus[["object_id", "formats", "author_slugs", "title"]],
                      on="object_id", how="left", suffixes=("", "_corpus"))
    if pilot:
        merged = pd.concat([
            merged[merged["formats"].astype(str).str.contains("podcast", na=False)],
            merged[merged["formats"].astype(str).str.contains("video", na=False)].head(20),
        ])
    rows = []
    for _, row in merged.iterrows():
        mode = route_post(row).mode
        # Host-prior: which a16z members host this format (disambiguates first-name thanks).
        prior = HOST_PRIOR["podcast"] if mode == CONVERSATIONAL else HOST_PRIOR["research-seminar"]
        try:
            segs = _segments_for_post(row, row["transcript"], roster, mode)
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
    args = ap.parse_args()
    df = build(pilot=args.pilot)
    SEGMENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SEGMENTS_OUT, index=False)
    PERSONS_DIR.mkdir(parents=True, exist_ok=True)
    corp = build_person_corpus(df) if len(df) else {}
    for slug, text in corp.items():
        (PERSONS_DIR / f"{slug}.txt").write_text(text)
    kept = int(df["kept"].sum()) if len(df) else 0
    with open(REPORT_OUT, "w") as f:
        f.write(f"# Attribution report ({dt.datetime.now(dt.timezone.utc).isoformat()})\n\n")
        f.write(f"- segments: {len(df)} | kept: {kept} | dropped: {len(df)-kept}\n")
        f.write(f"- a16z people with corpus: {len(corp)}\n\n")
        if len(df):
            f.write("## Dropped reasons\n")
            f.write(df[~df['kept']]['dropped_reason'].value_counts().to_string())
            f.write("\n\n## Per-person kept chars\n")
            for slug, text in sorted(corp.items(), key=lambda x: -len(x[1])):
                f.write(f"- {slug}: {len(text):,}\n")
    _log(f"Wrote {SEGMENTS_OUT} ({len(df)} segs, {kept} kept), {len(corp)} person corpora")

if __name__ == "__main__":
    main()
