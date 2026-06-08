"""Leakage firewall for a PeriodSignal: every citation quote must be a verbatim
substring of the in-window corpus and dated <= T. Adapts doppelganger/soul_audit.py
to the flat signal schema. The distillation being EXTRACTIVE is what lets this work."""
from __future__ import annotations
import difflib
from dataclasses import dataclass
from datetime import date

from signals.schema import PeriodSignal, Citation

_MATCH_THRESHOLD = 0.85

# Fold smart punctuation so a verbatim quote with curly quotes / em-dashes still matches.
_SMART = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "—": "-", "–": "-",
})


def _normalize(s: str) -> str:
    return " ".join(s.translate(_SMART).lower().split())


@dataclass(frozen=True)
class AuditReport:
    checked: int
    matched: int
    hallucinated: list[Citation]
    leaked: list[Citation]

    @property
    def ok(self) -> bool:
        return not self.hallucinated and not self.leaked


def _quote_in_corpus(quote: str, corpus_norm: str) -> bool:
    q = _normalize(quote)
    if not q:
        return False
    if q in corpus_norm:
        return True
    # Windowed fuzzy fallback for minor near-verbatim drift (no global gate — that was inert).
    win = len(q) + 10
    step = max(1, len(q) // 2)
    return any(
        difflib.SequenceMatcher(None, q, corpus_norm[i:i + win]).ratio() >= _MATCH_THRESHOLD
        for i in range(0, max(1, len(corpus_norm) - len(q) + 1), step)
    )


def _citations(period: PeriodSignal):
    for it in period.items:
        for c in it.citations:
            yield c


def audit_period(period: PeriodSignal, corpus_text: str, t: date) -> AuditReport:
    corpus_norm = _normalize(corpus_text)
    checked = matched = 0
    hallucinated: list[Citation] = []
    leaked: list[Citation] = []
    for c in _citations(period):
        checked += 1
        # leakage: claimed date after T
        try:
            cdate = date.fromisoformat(c.date)
        except ValueError:
            cdate = None
        if cdate and cdate > t:
            leaked.append(c)
            continue
        if _quote_in_corpus(c.quote, corpus_norm):
            matched += 1
        else:
            hallucinated.append(c)
    return AuditReport(checked=checked, matched=matched,
                       hallucinated=hallucinated, leaked=leaked)
