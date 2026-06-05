"""Route each post to an attribution mode."""
from dataclasses import dataclass

STRUCTURED = "structured"          # caption-only talk -> text-LLM only
CONVERSATIONAL = "conversational"  # podcast (audio) -> both engines

@dataclass
class Route:
    mode: str
    has_audio: bool
    panel_candidate: bool  # multi-voice video: future audio-diarization candidate

def _fmt(row):
    try:
        seq = list(row.get("formats"))
        return str(seq[0]) if seq else None
    except TypeError:
        return None

def _n_auth(row):
    try:
        return len([x for x in list(row.get("author_slugs")) if x])
    except TypeError:
        return 0

def route_post(row) -> Route:
    fmt = _fmt(row)
    if fmt == "podcasts":
        return Route(CONVERSATIONAL, True, False)
    # videos (and anything else with a transcript) are caption-only/structured in v1.
    return Route(STRUCTURED, False, _n_auth(row) >= 2)
