import os
import glob
import pytest
from attribution.diarize import assign_segments_to_turns

def test_assign_segment_to_max_overlap_turn():
    turns = [(0.0, 5.0, "S0"), (5.0, 10.0, "S1")]
    segs = [{"start": 0.5, "end": 4.0, "text": "a"}, {"start": 6.0, "end": 9.0, "text": "b"}]
    out = assign_segments_to_turns(segs, turns)
    assert [s["voice"] for s in out] == ["S0", "S1"]

def test_segment_spanning_boundary_takes_larger_overlap():
    turns = [(0.0, 5.0, "S0"), (5.0, 10.0, "S1")]
    segs = [{"start": 4.0, "end": 9.0, "text": "x"}]  # 1s in S0, 4s in S1
    assert assign_segments_to_turns(segs, turns)[0]["voice"] == "S1"

@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF token")
def test_pyannote_runs_on_cached_podcast():
    from attribution.diarize import diarize_audio
    mp3 = glob.glob("data/a16z_research/audio_cache/*.mp3")
    assert mp3, "no cached podcast audio"
    turns = diarize_audio(mp3[0])
    assert len(turns) >= 2 and all(len(t) == 3 for t in turns)
