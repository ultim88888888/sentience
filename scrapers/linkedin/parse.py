"""Parse a Voyager profileView payload into a structured Profile."""
from __future__ import annotations

from .models import Education, Experience, Profile


def _fmt_date(date: dict | None) -> str | None:
    """Voyager date {month, year} -> 'YYYY-MM' or 'YYYY'. None if no year."""
    if not date:
        return None
    year = date.get("year")
    if not year:
        return None
    month = date.get("month")
    return f"{year:04d}-{month:02d}" if month else f"{year:04d}"


def _period(node: dict) -> tuple[str | None, str | None]:
    tp = node.get("timePeriod") or {}
    return _fmt_date(tp.get("startDate")), _fmt_date(tp.get("endDate"))


def parse_profile(slug: str, payload: dict) -> Profile:
    profile = payload.get("profile") or {}
    name = " ".join(
        p for p in (profile.get("firstName"), profile.get("lastName")) if p
    ) or None

    experience: list[Experience] = []
    for el in (payload.get("positionView") or {}).get("elements", []):
        start, end = _period(el)
        experience.append(Experience(
            title=el.get("title"),
            company=el.get("companyName"),
            start=start,
            end=end,
            description=el.get("description"),
        ))

    education: list[Education] = []
    for el in (payload.get("educationView") or {}).get("elements", []):
        start, end = _period(el)
        education.append(Education(
            school=el.get("schoolName"),
            degree=el.get("degreeName"),
            field=el.get("fieldOfStudy"),
            start=start,
            end=end,
        ))

    return Profile(
        slug=slug,
        name=name,
        headline=profile.get("headline"),
        location=profile.get("locationName"),
        bio=profile.get("summary"),
        experience=experience,
        education=education,
    )
