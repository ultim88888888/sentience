"""Fuse diarization (boundaries+voices) with LLM attribution (names) and cross-check."""
from collections import Counter

def name_voices(voice_segs, llm_by_text):
    """Assign each anonymous voice label its majority LLM-resolved name."""
    votes = {}
    for s in voice_segs:
        nm = llm_by_text.get(s["text"])
        if nm:
            votes.setdefault(s["voice"], Counter())[nm] += 1
    return {v: c.most_common(1)[0][0] for v, c in votes.items()}

def fuse_segments(voice_segs, voice_names, llm_by_text):
    """Produce fused segments; `agreed` = diarization-voice name matches the LLM's per-segment
    name (the cross-validation / poison detector)."""
    out = []
    for s in voice_segs:
        diar_name = voice_names.get(s["voice"])
        llm_name = llm_by_text.get(s["text"])
        agreed = bool(diar_name) and diar_name == llm_name
        out.append({**s, "speaker": diar_name, "llm_speaker": llm_name, "agreed": agreed})
    return out
