"""doppelganger.soul_audit — verify a soul card's inline citations against the corpus.

Two failure modes the audit catches:
  - hallucinated: a cited quote matches NO evidence item.
  - leaked: a cited quote's real (or claimed) date is AFTER the soul's t0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

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
