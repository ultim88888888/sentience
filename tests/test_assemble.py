import pandas as pd
from attribution.assemble import build_person_corpus

def test_groups_kept_a16z_segments_by_slug_with_guest_context():
    segs = pd.DataFrame([
        {"post_id": "p1", "segment_idx": 0, "slug": "sonal-chokshi", "is_a16z": True,
         "kept": True, "speaker": "Sonal Chokshi", "text": "Great question."},
        {"post_id": "p1", "segment_idx": 1, "slug": None, "is_a16z": False,
         "kept": True, "speaker": "GUEST_1", "text": "I think L2s win."},
        {"post_id": "p1", "segment_idx": 2, "slug": "sonal-chokshi", "is_a16z": True,
         "kept": True, "speaker": "Sonal Chokshi", "text": "Why?"},
        {"post_id": "p1", "segment_idx": 3, "slug": "sonal-chokshi", "is_a16z": True,
         "kept": False, "speaker": "Sonal Chokshi", "text": "dropped-uncertain"},
    ])
    corp = build_person_corpus(segs)
    assert "sonal-chokshi" in corp
    text = corp["sonal-chokshi"]
    assert "Great question." in text and "Why?" in text
    assert "dropped-uncertain" not in text          # gate respected
    assert "[GUEST]: I think L2s win." in text       # context retained for responses
