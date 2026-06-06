"""Fuse diarization (boundaries+voices) with LLM attribution (names) and cross-check."""
from collections import Counter

from .attribute_text import _is_placeholder


def fuse_by_voice(voiced, names, resolve):
    """Corrected both-engines fusion (index-aligned, voice-majority).

    `voiced`  : segments [{text,start,end,voice}], one per whisper segment.
    `names`   : per-segment LLM name, aligned to `voiced` BY INDEX (same segmentation).
    `resolve` : fn(name) -> roster slug or None.

    Resolves each name to an identity (roster slug, else lowercased external name), takes a
    MAJORITY VOTE per diarization voice (so a voice = one identity even if the LLM wobbles or
    diarization over-splits), and reports intra-voice naming **consistency** as the per-segment
    confidence — that consistency IS the cross-check (diarization says "same voice"; how often
    does the LLM agree on who?). Returns (fused_segments, voice->identity, voice->consistency).
    """
    ident = []
    for n in names:
        if not n or _is_placeholder(n):
            ident.append(None)
        else:
            slug = resolve(n)
            ident.append(slug if slug else n.strip().lower())
    votes = {}
    for s, idv in zip(voiced, ident):
        if idv:
            votes.setdefault(s["voice"], Counter())[idv] += 1
    voice_id = {v: c.most_common(1)[0][0] for v, c in votes.items()}
    consistency = {v: c.most_common(1)[0][1] / sum(c.values()) for v, c in votes.items()}
    fused = []
    for s, idv in zip(voiced, ident):
        vid = voice_id.get(s["voice"])
        fused.append({**s, "identity": vid, "llm_identity": idv,
                      "agree": (idv == vid and idv is not None),
                      "confidence": round(consistency.get(s["voice"], 0.0), 3)})
    return fused, voice_id, consistency

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
