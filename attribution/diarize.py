"""Local pyannote diarization + map timestamped whisper segments to voice turns."""
import subprocess

from .config import HF_TOKEN_OP, PYANNOTE_MODEL, WHISPER_MODEL

def assign_segments_to_turns(segments, turns):
    """Tag each whisper segment with the diarization turn it overlaps most."""
    out = []
    for s in segments:
        best, best_ov = None, 0.0
        for (t0, t1, label) in turns:
            ov = max(0.0, min(s["end"], t1) - max(s["start"], t0))
            if ov > best_ov:
                best, best_ov = label, ov
        out.append({**s, "voice": best})
    return out

def _hf_token() -> str:
    return subprocess.check_output(["op", "read", HF_TOKEN_OP], text=True).strip()

def diarize_audio(mp3_path: str):
    """Return [(start, end, voice_label)] from pyannote (local)."""
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=_hf_token())
    diar = pipe(mp3_path)
    return [(seg.start, seg.end, label) for seg, _, label in diar.itertracks(yield_label=True)]

def transcribe_timestamped(mp3_path: str):
    """Whisper segments WITH timestamps (we need them to align to diarization)."""
    import mlx_whisper
    res = mlx_whisper.transcribe(mp3_path, path_or_hf_repo=WHISPER_MODEL)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in res.get("segments", [])]
