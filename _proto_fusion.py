"""PROTOTYPE (not committed) — corrected both-engines fusion: index-aligned, voice-majority.

Loads cached diarize+whisper (/tmp/fusion_cache.json), has the LLM label each NUMBERED whisper
segment (same segmentation as diarization), fuses by index, resolves voice->name by majority,
and uses intra-voice naming consistency as the confidence/cross-check signal.
"""
import json
import re
from collections import Counter

from attribution.attribute_text import _call_claude, _is_placeholder

SEG_PROMPT = """Label who speaks each numbered transcript segment of a podcast.
Known participants: {participants}.
Return ONLY JSON: {{"labels": {{"<index>": "<speaker>"}}}} for every index shown.
Use a real name once it's identifiable (introductions, "thanks X", the participant list) and
keep that same name for that speaker throughout. Use HOST/GUEST_1/GUEST_2 only if a speaker is
never identifiable.

SEGMENTS:
{numbered}"""


def parse_labels(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    obj = json.loads(m.group(0))
    return obj["labels"]


def attribute_segments(segments, participants, batch=60):
    names = [None] * len(segments)
    known = list(participants)
    for start in range(0, len(segments), batch):
        chunk = segments[start:start + batch]
        numbered = "\n".join(f"[{start+i}] {s['text']}" for i, s in enumerate(chunk))
        prompt = SEG_PROMPT.format(participants=", ".join(dict.fromkeys(known)) or "unknown",
                                   numbered=numbered)
        labels = parse_labels(_call_claude(prompt))
        for k, v in labels.items():
            i = int(k)
            if 0 <= i < len(names):
                names[i] = v
            if v and not _is_placeholder(v) and v not in known:
                known.append(v)
    return names


def fuse_by_voice(voiced, names):
    by_voice = {}
    for s, nm in zip(voiced, names):
        if nm and not _is_placeholder(nm):
            by_voice.setdefault(s["voice"], Counter())[nm] += 1
    voice_name = {v: c.most_common(1)[0][0] for v, c in by_voice.items()}
    consistency = {v: c.most_common(1)[0][1] / sum(c.values()) for v, c in by_voice.items()}
    fused = []
    for s, nm in zip(voiced, names):
        vn = voice_name.get(s["voice"])
        fused.append({**s, "speaker": vn, "llm_name": nm,
                      "agree": (nm == vn), "confidence": round(consistency.get(s["voice"], 0), 2)})
    return fused, voice_name, consistency


def normalize(names, roster, prior):
    """Resolve each raw LLM name to a roster slug (or a stable placeholder) so 'Eddy' and
    'Eddy Lazzarin' collapse to the same identity before voting."""
    out = []
    for n in names:
        if not n or _is_placeholder(n):
            out.append(None if not n else n.upper())
            continue
        m = roster.resolve(n, prefer=prior)
        out.append(m.slug if m.slug else n.strip().lower())  # slug, or normalized external name
    return out


if __name__ == "__main__":
    import os
    from attribution.roster import Roster
    from attribution.config import HOST_PRIOR
    cache = json.load(open("/tmp/fusion_cache.json"))
    voiced = cache["voiced"]
    print(f"loaded {len(voiced)} voiced segments")
    if os.path.exists("/tmp/fusion_names.json"):
        names = json.load(open("/tmp/fusion_names.json")); print("(loaded cached LLM names)")
    else:
        names = attribute_segments(voiced, ["Eddy Lazzarin"])
        json.dump(names, open("/tmp/fusion_names.json", "w"))
    roster = Roster.load()
    norm = normalize(names, roster, HOST_PRIOR["podcast"] + ["eddy-lazzarin"])
    fused, voice_id, cons = fuse_by_voice(voiced, norm)
    agree = sum(1 for f in fused if f["agree"])
    print(f"\nRAW voice->name: { {k: Counter(n for s,n in zip(voiced,names) if s['voice']==k).most_common(1)[0][0] for k in voice_id} }")
    print(f"NORMALIZED voice->slug: {voice_id}")
    print(f"voice consistency (normalized): { {k: round(v,2) for k,v in cons.items()} }")
    print(f"per-segment agreement w/ voice-majority (normalized): {agree}/{len(fused)} ({100*agree/len(fused):.0f}%)")
    # who is Eddy? how many segments resolve to eddy-lazzarin, at what mean voice-consistency
    eddy = [f for f in fused if f["speaker"] == "eddy-lazzarin"]
    print(f"\nEddy segments: {len(eddy)} | voices that resolved to Eddy: {sorted({f['voice'] for f in eddy})}")
