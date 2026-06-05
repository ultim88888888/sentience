from attribution.fuse import name_voices, fuse_segments

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
