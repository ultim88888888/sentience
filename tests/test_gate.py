from attribution.gate import gate_segment

def test_structured_kept_when_confident():
    kept, reason = gate_segment({"method": "text", "confidence": 0.8})
    assert kept is True and reason is None

def test_structured_dropped_when_unsure():
    kept, reason = gate_segment({"method": "text", "confidence": 0.5})
    assert kept is False and reason == "low_confidence"

def test_conversational_dropped_on_disagreement():
    kept, reason = gate_segment({"method": "fused", "confidence": 0.9, "agreed": False})
    assert kept is False and reason == "method_disagreement"

def test_conversational_kept_on_agreement():
    assert gate_segment({"method": "fused", "confidence": 0.9, "agreed": True})[0] is True
