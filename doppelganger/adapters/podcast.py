"""doppelganger.adapters.podcast — attributed_transcripts.jsonl -> EvidenceItems.

One item per subject turn (slug == subject, confidence >= threshold). The immediately
preceding non-subject turn is attached as context (the question he's answering).
Interlocutor turns are never attributed to the subject.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.schema import EvidenceItem


def load_podcast(jsonl_path: Path, subject_slug: str,
                 min_confidence: float = config.PODCAST_MIN_CONFIDENCE) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for line in Path(jsonl_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if subject_slug not in rec.get("a16z_participants", []):
            continue
        ts = pd.to_datetime(rec["post_date"], utc=True).to_pydatetime()
        segments = rec.get("segments", [])
        for i, seg in enumerate(segments):
            if seg.get("slug") != subject_slug or not seg.get("kept"):
                continue
            if float(seg.get("confidence", 0)) < min_confidence:
                continue
            prev = segments[i - 1] if i > 0 else None
            context = str(prev["text"]) if prev and prev.get("slug") != subject_slug else None
            items.append(EvidenceItem(
                id=f"{rec['object_id']}:{seg['idx']}", subject=subject_slug, timestamp=ts,
                source_type="podcast", text=str(seg["text"]), speaker_slug=subject_slug,
                attribution_confidence=float(seg["confidence"]), thread_id=str(rec["object_id"]),
                context=context,
            ))
    return items
