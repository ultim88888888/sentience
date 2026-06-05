"""LLM text attribution via `claude -p` (Opus, $0 on Max). Pure helpers are unit-tested;
the model call is validated in the pilot."""
import json
import re
import subprocess
import tempfile
import time

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

PROMPT = """You are segmenting a transcript into speaker turns. Known participants: {participants}.
Return ONLY JSON: {{"segments":[{{"speaker": <name-or-placeholder>, "text": <verbatim>,
"confidence": <0-1>}}]}}.

Rules:
- Identify who speaks each turn. Once a speaker's real name is established ANYWHERE (an intro
  like "please welcome X", a thank-you like "Thanks, Y", or the participant list above), use
  that SAME real name for ALL of that speaker's turns — earlier and later — never a role label.
- Use a stable placeholder (HOST, GUEST_1, GUEST_2, AUDIENCE) ONLY for a speaker whose real
  name is genuinely never identifiable.
- confidence = how sure you are of the identity for that turn; be conservative and lower it
  for brief/ambiguous turns (e.g. unidentified audience questions).

TRANSCRIPT:
{chunk}"""

_PLACEHOLDER = re.compile(r"^(HOST|GUEST(_\d+)?|AUDIENCE|SPEAKER(_?\d+)?|UNKNOWN)$", re.I)

def _is_placeholder(name) -> bool:
    return not name or bool(_PLACEHOLDER.match(str(name).strip()))

def _call_claude(prompt: str, retries: int = 4) -> str:
    """One `claude -p` call with retry+backoff — transient API errors (overload/rate) return
    a non-zero exit, which must NOT fail the whole post."""
    last = ""
    for attempt in range(1, retries + 1):
        r = subprocess.run(
            ["claude", "-p", "--model", LLM_MODEL, "--effort", LLM_EFFORT,
             "--no-session-persistence"],
            input=prompt, text=True, capture_output=True, cwd=_CLEAN_CWD)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
        last = (r.stderr or r.stdout or "no output")[:200]
        if attempt < retries:
            time.sleep(8 * attempt)  # 8/16/24s backoff
    raise RuntimeError(f"claude -p failed after {retries} attempts: {last}")


def attribute(text: str, participants: list[str]) -> list[dict]:
    """Run the LLM over each chunk and merge. Names identified in earlier chunks are carried
    forward as participant hints so a speaker keeps one identity across chunk boundaries."""
    results = []
    known: list[str] = list(participants)
    for ch in chunk_transcript(text):
        hint = ", ".join(dict.fromkeys(known)) or "unknown"
        prompt = PROMPT.format(participants=hint, chunk=ch["text"])
        out = _call_claude(prompt)
        segs = parse_segments(out)
        results.append(segs)
        for s in segs:  # learn real names for the next chunk's hint
            nm = str(s.get("speaker", "")).strip()
            if nm and not _is_placeholder(nm) and nm not in known:
                known.append(nm)
    return merge_chunks(results)
