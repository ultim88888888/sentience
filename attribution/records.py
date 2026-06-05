"""Per-post attributed-transcript records — the canonical deliverable.

One record per post = full name-tagged transcript + metadata. The doppelganger engine
filters these (e.g. where a target appears in `a16z_participants`) and trains on the tagged
segments. Producer emits clean, attributed, metadata-rich transcripts; the consumer decides
how to assemble a given person's training set.
"""
import json

import pandas as pd

from .config import CORPUS, DATA, SEGMENTS_OUT

OUT_JSONL = DATA / "attributed_transcripts.jsonl"

_META_COLS = ["object_id", "title", "permalink", "post_date", "formats", "categories",
              "tags", "author_slugs", "poster_display_name"]


def _listify(v):
    try:
        return [str(x) for x in list(v)]
    except TypeError:
        return [] if v is None else [str(v)]


def build_records(seg: pd.DataFrame, corpus: pd.DataFrame) -> list[dict]:
    meta = corpus.set_index("object_id")
    records = []
    for pid, pdf in seg.sort_values(["post_id", "segment_idx"]).groupby("post_id"):
        m = meta.loc[pid] if pid in meta.index else None
        a16z = sorted({s for s in pdf.loc[pdf["is_a16z"] == True, "slug"].dropna()})  # noqa: E712
        speakers = list(dict.fromkeys(str(s) for s in pdf["speaker"] if s))
        segments = [{
            "idx": int(r.segment_idx), "speaker": (None if pd.isna(r.speaker) else str(r.speaker)),
            "slug": (None if pd.isna(r.slug) else str(r.slug)), "is_a16z": bool(r.is_a16z),
            "confidence": (None if pd.isna(r.confidence) else float(r.confidence)),
            "kept": bool(r.kept), "text": str(r.text),
        } for r in pdf.itertuples()]
        records.append({
            "object_id": pid,
            "title": None if m is None else str(m.get("title")),
            "post_date": None if m is None else str(m.get("post_date")),
            "format": None if m is None else (_listify(m.get("formats")) or [None])[0],
            "permalink": None if m is None else str(m.get("permalink")),
            "categories": [] if m is None else _listify(m.get("categories")),
            "tags": [] if m is None else _listify(m.get("tags")),
            "credited_authors": [] if m is None else _listify(m.get("author_slugs")),
            "a16z_participants": a16z,        # a16z people who actually SPEAK (matching key)
            "all_speakers": speakers,
            "segments": segments,
        })
    return records


def export(seg: pd.DataFrame | None = None) -> int:
    seg = pd.read_parquet(SEGMENTS_OUT) if seg is None else seg
    corpus = pd.read_parquet(CORPUS)
    records = build_records(seg, corpus)
    with open(OUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


if __name__ == "__main__":
    n = export()
    print(f"wrote {OUT_JSONL} ({n} post records)")
