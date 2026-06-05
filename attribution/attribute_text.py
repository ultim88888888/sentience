"""LLM text attribution via `claude -p` (Opus, $0 on Max). Pure helpers are unit-tested;
the model call is validated in the pilot."""
import json
import re
import subprocess
import tempfile

from .config import CHUNK_CHARS, CHUNK_OVERLAP, LLM_EFFORT, LLM_MODEL

# Run `claude -p` from a clean dir so it doesn't inherit the project CLAUDE.md (the Fushi
# soul) — that would contaminate attribution output. Same guard the doppelganger evals use.
_CLEAN_CWD = tempfile.mkdtemp(prefix="attribution-llm-")

def chunk_transcript(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    """Split into overlapping char windows so a turn isn't blindly cut."""
    if len(text) <= size:
        return [{"start": 0, "end": len(text), "text": text}]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append({"start": start, "end": end, "text": text[start:end]})
        if end == len(text):
            break
        start = end - overlap
    return out

def parse_segments(raw: str):
    """Extract the JSON object from an LLM reply (tolerates ```json fences)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in LLM output")
    obj = json.loads(m.group(0))
    return obj["segments"]

def merge_chunks(chunk_results):
    """Concatenate per-chunk segments, dropping overlap duplicates (same speaker+text)."""
    merged, seen = [], set()
    for segs in chunk_results:
        for s in segs:
            key = (s["speaker"], s["text"].strip())
            if key in seen:
                continue
            seen.add(key)
            merged.append(s)
    return merged

PROMPT = """You are segmenting a transcript by speaker. Known participants: {participants}.
Return ONLY JSON: {{"segments":[{{"speaker": <name or "HOST"/"GUEST_1">, "text": <verbatim>,
"confidence": <0-1>}}]}}. Use a real name only when the transcript makes it unambiguous
(introductions, thank-yous). Otherwise use stable HOST/GUEST_n labels. Be conservative:
when unsure who is speaking, lower the confidence.

TRANSCRIPT:
{chunk}"""

def attribute(text: str, participants: list[str]) -> list[dict]:
    """Run the LLM over each chunk and merge. Returns segment dicts."""
    results = []
    for ch in chunk_transcript(text):
        prompt = PROMPT.format(participants=", ".join(participants) or "unknown", chunk=ch["text"])
        out = subprocess.run(
            ["claude", "-p", "--model", LLM_MODEL, "--effort", LLM_EFFORT,
             "--no-session-persistence"],
            input=prompt, text=True, capture_output=True, check=True,
            cwd=_CLEAN_CWD).stdout
        results.append(parse_segments(out))
    return merge_chunks(results)
