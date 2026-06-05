"""doppelganger.adapters.research — a16z research articles -> EvidenceItems.

Solo-authored posts are high-confidence the subject's voice; co-authored "firm"
posts are co-signed, lower confidence. One item per post (chunking is the memory
unit's concern, not ingestion's).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from doppelganger.schema import EvidenceItem


def _authors(val) -> list[str]:
    if isinstance(val, (list, np.ndarray)):
        return [str(s) for s in val]
    return [] if val is None else [str(val)]


def load_research(articles_path: Path, subject_slug: str) -> list[EvidenceItem]:
    df = pd.read_parquet(articles_path)
    items: list[EvidenceItem] = []
    for _, r in df.iterrows():
        authors = _authors(r["author_slugs"])
        if subject_slug not in authors:
            continue
        text = r.get("acf_content") or r.get("extracted_text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        solo = len(authors) == 1
        ts = pd.to_datetime(r["post_date"], utc=True).to_pydatetime()
        items.append(EvidenceItem(
            id=str(r["object_id"]), subject=subject_slug, timestamp=ts,
            source_type="research" if solo else "research_firm", text=text,
            speaker_slug=subject_slug, attribution_confidence=1.0 if solo else 0.5,
        ))
    return items
