"""doppelganger.soul_audit — verify a soul card's inline citations against the corpus.

Two failure modes the audit catches:
  - hallucinated: a cited quote matches NO evidence item.
  - leaked: a cited quote's real (or claimed) date is AFTER the soul's t0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# A cited quote counts as grounded if at least this fraction of it appears as one
# contiguous run inside a real evidence item. Tolerates a trailing period / 1-char
# typographic drift while still rejecting fabrications (which score near zero).
_MATCH_THRESHOLD = 0.85

_CITE = re.compile(r'\[(\d{4}-\d{2}-\d{2})\]\s+"([^"]{3,})"')


@dataclass(frozen=True)
class Citation:
    date: date
    quote: str


def parse_citations(card: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _CITE.finditer(card):
        y, mo, d = (int(x) for x in m.group(1).split("-"))
        out.append(Citation(date(y, mo, d), m.group(2).strip()))
    return out


@dataclass
class AuditReport:
    checked: int
    matched: int
    hallucinated: list[Citation]
    leaked: list[Citation]

    @property
    def ok(self) -> bool:
        return not self.hallucinated and not self.leaked


# Fold typographic quotes/apostrophes so cited quotes match raw corpus text.
# The corpus uses curly quotes (e.g. "can't", "endgame"); models tend to cite
# with straight quotes. Without folding, real quotes read as hallucinated.
_QUOTE_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # ' ' ‚ ‛
    "“": '"', "”": '"', "„": '"', "‟": '"',  # " " „ ‟
    "′": "'", "″": '"',                                  # ′ ″
    "—": "-", "–": "-", "―": "-",                          # em / en / horizontal-bar dashes
    "…": "...",                                            # ellipsis
})


def _norm(s: str) -> str:
    return " ".join(str(s).translate(_QUOTE_FOLD).lower().split())


def _coverage(quote_norm: str, text_norm: str) -> float:
    """Fraction of the (normalized) quote present as one contiguous run in text."""
    if not quote_norm:
        return 0.0
    m = SequenceMatcher(None, quote_norm, text_norm, autojunk=False)
    longest = m.find_longest_match(0, len(quote_norm), 0, len(text_norm)).size
    return longest / len(quote_norm)


def audit_soul(card_path: Path, evidence_path: Path, t0: date) -> AuditReport:
    card = Path(card_path).read_text()
    cites = parse_citations(card)

    ev = pd.read_parquet(evidence_path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    norm_items = [(_norm(t), pd.Timestamp(ts).date())
                  for t, ts in zip(ev["text"], ev["timestamp"])]

    matched, hallucinated, leaked = 0, [], []
    for c in cites:
        q = _norm(c.quote)
        hits = [d for text, d in norm_items if _coverage(q, text) >= _MATCH_THRESHOLD]
        if not hits:
            hallucinated.append(c)            # quote matches no evidence item
        elif min(hits) > t0 or c.date > t0:
            leaked.append(c)                  # only matches post-t0 items, or cites a post-t0 date
        else:
            matched += 1
    return AuditReport(len(cites), matched, hallucinated, leaked)
