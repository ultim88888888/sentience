"""Decide which segments reach the clean per-person corpus (precision-first)."""
from .config import CONF_MIN

def gate_segment(seg) -> tuple[bool, str | None]:
    """Return (kept, dropped_reason). Conversational requires method agreement."""
    if seg["method"] == "fused" and not seg.get("agreed", False):
        return False, "method_disagreement"
    if seg.get("confidence", 0.0) < CONF_MIN:
        return False, "low_confidence"
    return True, None
