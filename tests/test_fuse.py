from attribution.fuse import name_voices, fuse_segments, fuse_by_voice
from attribution.attribute_text import parse_labels


def _resolve(name):
    # toy roster: Eddy/Eddy Lazzarin -> eddy-lazzarin; others external
    return "eddy-lazzarin" if name.lower().startswith("eddy") else None


def test_fuse_by_voice_majority_vote_collapses_name_wobble():
    voiced = [{"voice": "S0", "text": "a"}, {"voice": "S0", "text": "b"},
              {"voice": "S0", "text": "c"}, {"voice": "S1", "text": "d"}]
    # LLM wobbles "Eddy" vs "Eddy Lazzarin" for S0; both resolve to same slug -> one identity
    names = ["Eddy", "Eddy Lazzarin", "Eddy", "Sonal Chokshi"]
    fused, voice_id, cons = fuse_by_voice(voiced, names, _resolve)
    assert voice_id["S0"] == "eddy-lazzarin"
    assert cons["S0"] == 1.0                      # all three normalize to the same identity
    assert voice_id["S1"] == "sonal chokshi"      # external -> lowercased name
    assert all(f["confidence"] == 1.0 for f in fused if f["voice"] == "S0")


def test_fuse_by_voice_low_consistency_flags_disagreement():
    voiced = [{"voice": "S0", "text": "a"}, {"voice": "S0", "text": "b"},
              {"voice": "S0", "text": "c"}, {"voice": "S0", "text": "d"}]
    names = ["Eddy Lazzarin", "Eddy Lazzarin", "Chris Dixon", "Ben Horowitz"]  # 50% majority
    fused, voice_id, cons = fuse_by_voice(voiced, names, _resolve)
    assert voice_id["S0"] == "eddy-lazzarin" and cons["S0"] == 0.5
    # segments whose own name disagrees with the voice majority are not 'agree'
    assert [f["agree"] for f in fused] == [True, True, False, False]


def test_parse_labels():
    raw = '```json\n{"labels": {"0": "Eddy Lazzarin", "1": "HOST"}}\n```'
    assert parse_labels(raw) == {"0": "Eddy Lazzarin", "1": "HOST"}

def test_name_voices_maps_label_to_resolved_name():
    voice_segs = [{"voice": "S0", "text": "welcome"}, {"voice": "S0", "text": "next"},
                  {"voice": "S1", "text": "thanks"}]
    llm_by_text = {"welcome": "Sonal Chokshi", "next": "Sonal Chokshi", "thanks": "GUEST_1"}
    assert name_voices(voice_segs, llm_by_text) == {"S0": "Sonal Chokshi", "S1": "GUEST_1"}

def test_fuse_marks_agreement_high_conf():
    voice_segs = [{"voice": "S0", "text": "welcome", "start": 0, "end": 2}]
    voice_names = {"S0": "Sonal Chokshi"}
    llm_by_text = {"welcome": "Sonal Chokshi"}
    out = fuse_segments(voice_segs, voice_names, llm_by_text)
    assert out[0]["speaker"] == "Sonal Chokshi" and out[0]["agreed"] is True

def test_fuse_marks_disagreement_low_conf():
    voice_segs = [{"voice": "S0", "text": "welcome", "start": 0, "end": 2}]
    voice_names = {"S0": "Sonal Chokshi"}
    llm_by_text = {"welcome": "Chris Dixon"}
    out = fuse_segments(voice_segs, voice_names, llm_by_text)
    assert out[0]["agreed"] is False
