import pytest
from attribution.attribute_text import chunk_transcript, parse_segments, merge_chunks

def test_chunk_respects_size_and_overlap():
    text = "x" * 50000
    chunks = chunk_transcript(text, size=20000, overlap=1000)
    assert len(chunks) == 3
    assert chunks[0]["end"] == 20000
    assert chunks[1]["start"] == 19000  # overlap pulled back

def test_chunk_short_text_single_chunk():
    chunks = chunk_transcript("hello", size=20000, overlap=1000)
    assert len(chunks) == 1 and chunks[0]["text"] == "hello"

def test_parse_segments_reads_llm_json():
    raw = '```json\n{"segments":[{"speaker":"Tim Roughgarden","text":"Welcome.","confidence":0.95}]}\n```'
    segs = parse_segments(raw)
    assert segs[0]["speaker"] == "Tim Roughgarden" and segs[0]["confidence"] == 0.95

def test_parse_segments_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_segments("not json at all")

def test_merge_chunks_dedupes_overlap_by_text():
    a = [{"speaker": "A", "text": "hello world", "confidence": 0.9}]
    b = [{"speaker": "A", "text": "hello world", "confidence": 0.9},
         {"speaker": "B", "text": "goodbye", "confidence": 0.8}]
    merged = merge_chunks([a, b])
    assert [s["text"] for s in merged] == ["hello world", "goodbye"]
