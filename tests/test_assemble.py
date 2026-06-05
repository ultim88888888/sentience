import pandas as pd
from attribution.assemble import build_person_corpus

def _seg(post, idx, slug, a16z, kept, speaker, text):
    return {"post_id": post, "segment_idx": idx, "slug": slug, "is_a16z": a16z,
            "kept": kept, "speaker": speaker, "text": text}

def test_person_corpus_scopes_to_own_turns_with_local_context():
    segs = pd.DataFrame([
        _seg("p1", 0, "sonal-chokshi", True, True, "Sonal Chokshi", "Great question."),
        _seg("p1", 1, None, False, True, "GUEST_1", "I think L2s win."),
        _seg("p1", 2, "sonal-chokshi", True, True, "Sonal Chokshi", "Why?"),
        _seg("p1", 3, "sonal-chokshi", True, False, "Sonal Chokshi", "dropped-uncertain"),
    ])
    corp = build_person_corpus(segs)
    text = corp["sonal-chokshi"]
    assert "Great question." in text and "Why?" in text
    assert "dropped-uncertain" not in text          # gate respected
    assert "[Q] I think L2s win." in text            # immediately-preceding prompt as context

def test_no_cross_contamination_between_people():
    # The bug we caught: guest words must NOT leak into an unrelated person's corpus.
    segs = pd.DataFrame([
        # post p1: Justin hosts, Ron (guest) speaks
        _seg("p1", 0, "justin-thaler", True, True, "Justin", "Introducing Ron."),
        _seg("p1", 1, None, False, True, "GUEST_1", "RON-SECRET-WORDS about SNARKs."),
        # post p2: Sonal hosts a totally different episode
        _seg("p2", 0, "sonal-chokshi", True, True, "Sonal", "Welcome to the show."),
    ])
    corp = build_person_corpus(segs)
    # Sonal (different post) must NOT contain Ron's guest words.
    assert "RON-SECRET-WORDS" not in corp["sonal-chokshi"]
    # Justin's corpus may carry Ron's line ONLY as local [Q] context to a Justin turn — but
    # here Ron speaks AFTER Justin's only turn, so it shouldn't be pulled in either.
    assert "RON-SECRET-WORDS" not in corp["justin-thaler"]

def test_min_segments_filters_spurious_people():
    # a single misattributed segment shouldn't mint a full person corpus.
    segs = pd.DataFrame([
        _seg("p1", 0, "justin-thaler", True, True, "Justin", "a"),
        _seg("p1", 1, "justin-thaler", True, True, "Justin", "b"),
        _seg("p1", 2, "tim-roughgarden", True, True, "Tim", "spurious-single"),
    ])
    corp = build_person_corpus(segs, min_segments=2)
    assert "justin-thaler" in corp
    assert "tim-roughgarden" not in corp     # only 1 segment -> filtered
