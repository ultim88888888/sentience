"""doppelganger.schema — shared data types for ingestion artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    subject: str
    timestamp: datetime           # tz-aware UTC
    source_type: str
    text: str
    speaker_slug: str
    attribution_confidence: float
    thread_id: str | None = None
    context: str | None = None
    context_missing: bool = False
    engagement: int | None = None


@dataclass
class Experience:
    title: str
    company: str
    start: date | None
    end: date | None
    description: str | None


@dataclass
class Education:
    school: str
    degree: str | None
    field: str | None
    start: date | None
    end: date | None


@dataclass
class IdentityProfile:
    slug: str
    name: str
    headline: str | None
    bio: str | None
    current_role: str | None
    experience: list[Experience] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    socials: dict[str, str] = field(default_factory=dict)

    def as_of(self, t: date) -> "IdentityProfile":
        """Return a copy truncated to what was true on/before date t."""
        exp = [e for e in self.experience if e.start is None or e.start <= t]
        edu = [e for e in self.education if e.start is None or e.start <= t]
        # current role: prefer a role active at t (start<=t and (end is None or end>t));
        # else the latest-started role with start<=t.
        active = [e for e in exp if (e.end is None or e.end > t)]
        pick = active or exp
        current = max(pick, key=lambda e: e.start or date.min).title if pick else None
        return replace(self, experience=exp, education=edu, current_role=current)
