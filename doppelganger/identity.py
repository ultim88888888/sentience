"""doppelganger.identity — merge LinkedIn + a16z bio into an IdentityProfile."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.registry import resolve_subject
from doppelganger.schema import Education, Experience, IdentityProfile

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_li_date(s: str | None) -> date | None:
    """Parse LinkedIn date strings: 'May 2026', '2008', None/'' -> date|None (day=1)."""
    if not s:
        return None
    parts = s.strip().split()
    if len(parts) == 2 and parts[0][:3].lower() in _MONTHS:
        return date(int(parts[1]), _MONTHS[parts[0][:3].lower()], 1)
    if len(parts) == 1 and parts[0].isdigit():
        return date(int(parts[0]), 1, 1)
    return None


def build_identity(
    slug: str,
    *,
    linkedin_path: Path | None = None,
    team_path: Path | None = None,
    tracked_people_path: Path | None = None,
) -> IdentityProfile:
    ref = resolve_subject(slug, tracked_people_path=tracked_people_path)

    li_path = linkedin_path or (config.LINKEDIN_DIR / (ref.linkedin_file or ""))
    li = json.loads(Path(li_path).read_text()) if Path(li_path).exists() else {}

    team = pd.read_parquet(team_path or config.TEAM_PARQUET)
    row = team[team["slug"] == slug]
    bio_a16z = str(row.iloc[0]["bio"]) if len(row) else None
    socials = {}
    if len(row):
        for col in ("x_url", "linkedin_url", "farcaster_url"):
            v = row.iloc[0].get(col)
            if isinstance(v, str) and v:
                socials[col] = v

    # bio: prefer the richer a16z bio, fall back to LinkedIn bio
    bio = bio_a16z or li.get("bio")

    experience = [
        Experience(e.get("title", ""), e.get("company", ""),
                   parse_li_date(e.get("start")), parse_li_date(e.get("end")), e.get("description"))
        for e in li.get("experience", [])
    ]
    education = [
        Education(e.get("school", ""), e.get("degree"), e.get("field"),
                  parse_li_date(e.get("start")), parse_li_date(e.get("end")))
        for e in li.get("education", [])
    ]

    return IdentityProfile(
        slug=slug, name=li.get("name") or ref.name, headline=li.get("headline"),
        bio=bio, current_role=None, experience=experience, education=education, socials=socials,
    )
