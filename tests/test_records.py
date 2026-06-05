import numpy as np
import pandas as pd
from attribution.records import build_records

def test_record_carries_metadata_tagged_segments_and_participants():
    seg = pd.DataFrame([
        {"post_id": "p1", "segment_idx": 0, "speaker": "Eddy Lazzarin", "slug": "eddy-lazzarin",
         "is_a16z": True, "confidence": 0.9, "kept": True, "text": "tokens win"},
        {"post_id": "p1", "segment_idx": 1, "speaker": "GUEST_1", "slug": None,
         "is_a16z": False, "confidence": 0.3, "kept": False, "text": "maybe"},
    ])
    corpus = pd.DataFrame([{"object_id": "p1", "title": "T", "permalink": "u",
                            "post_date": "2025-01-01", "formats": np.array(["podcasts"], dtype=object),
                            "categories": np.array(["research"], dtype=object),
                            "tags": np.array(["tokens"], dtype=object),
                            "author_slugs": np.array(["eddy-lazzarin"], dtype=object),
                            "poster_display_name": "x"}])
    rec = build_records(seg, corpus)[0]
    assert rec["title"] == "T" and rec["format"] == "podcasts" and rec["permalink"] == "u"
    assert rec["a16z_participants"] == ["eddy-lazzarin"]      # speakers resolved to a16z
    assert rec["all_speakers"] == ["Eddy Lazzarin", "GUEST_1"]
    assert len(rec["segments"]) == 2
    # low-confidence turns are RETAINED (engine decides thresholds), tagged with confidence
    assert rec["segments"][1]["kept"] is False and rec["segments"][1]["confidence"] == 0.3

def test_record_handles_missing_metadata():
    seg = pd.DataFrame([{"post_id": "x9", "segment_idx": 0, "speaker": "A", "slug": None,
                         "is_a16z": False, "confidence": 0.5, "kept": True, "text": "hi"}])
    rec = build_records(seg, pd.DataFrame({"object_id": []}))[0]
    assert rec["object_id"] == "x9" and rec["title"] is None and rec["a16z_participants"] == []
