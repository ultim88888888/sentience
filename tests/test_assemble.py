import pandas as pd
from attribution.assemble import build_person_corpus, build_person_conversations

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

def test_conversations_render_both_sides_in_order_with_target_marked():
    segs = pd.DataFrame([
        _seg("p1", 0, "robert-hackett", True, True, "Robert Hackett", "What do you think?"),
        _seg("p1", 1, "eddy-lazzarin", True, True, "Eddy Lazzarin", "I think tokens win."),
        _seg("p1", 2, None, False, True, "GUEST_1", "Disagree."),
        _seg("p1", 3, "eddy-lazzarin", True, True, "Eddy Lazzarin", "Here's why."),
    ])
    conv = build_person_conversations(segs, "eddy-lazzarin")
    # both sides present, in order
    assert conv.index("What do you think?") < conv.index("I think tokens win.") < conv.index("Disagree.")
    # other speakers retained (the whole dialogue), labelled
    assert "Robert Hackett: What do you think?" in conv
    assert "GUEST_1: Disagree." in conv
    # eddy's own turns marked as the target
    assert "Eddy Lazzarin: I think tokens win.  ⟵ TARGET" in conv

def test_conversations_only_includes_posts_target_appears_in():
    segs = pd.DataFrame([
        _seg("p1", 0, "eddy-lazzarin", True, True, "Eddy Lazzarin", "eddy here"),
        _seg("p2", 0, "sonal-chokshi", True, True, "Sonal Chokshi", "no eddy in this post"),
    ])
    conv = build_person_conversations(segs, "eddy-lazzarin")
    assert "eddy here" in conv and "no eddy in this post" not in conv

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
