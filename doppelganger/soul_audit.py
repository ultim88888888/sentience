"""doppelganger.soul_audit — verify a soul card's inline citations against the corpus.

Two failure modes the audit catches:
  - hallucinated: a cited quote matches NO evidence item.
  - leaked: a cited quote's real (or claimed) date is AFTER the soul's t0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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


@dataclass
class AuditReport:
    checked: int
    matched: int
    hallucinated: list[Citation]
    leaked: list[Citation]

    @property
    def ok(self) -> bool:
        return not self.hallucinated and not self.leaked


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


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
        hits = [d for text, d in norm_items if q in text]
        if not hits:
            hallucinated.append(c)            # quote matches no evidence item
        elif min(hits) > t0 or c.date > t0:
            leaked.append(c)                  # only matches post-t0 items, or cites a post-t0 date
        else:
            matched += 1
    return AuditReport(len(cites), matched, hallucinated, leaked)
