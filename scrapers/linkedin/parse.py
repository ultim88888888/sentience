"""Parse a public LinkedIn profile page into a structured Profile.

Two complementary sources in the public SSR page:
  1. A schema.org `ld+json` Person node — clean name, bio (description), and a
     coarse worksFor/alumniOf list (org/school + years, no role titles).
  2. Stable, non-obfuscated HTML sections (`li.experience-item`,
     `li.education__list-item`) — per-entry detail (title, dates, degree).

We take name/bio from the ld+json and per-entry detail from the HTML sections,
falling back to the ld+json lists when the HTML sections are absent (which is
itself a signal the profile is restricted/thin for logged-out viewers).
"""
from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from .models import Education, Experience, Profile

_DASH = re.compile(r"\s*[–—-]\s*")
# LinkedIn redacts fields for logged-out viewers of less-public profiles by
# replacing the text with runs of asterisks ('********** ********').
_MASK = re.compile(r"\*{3,}")


def _unmask(t: str | None) -> str | None:
    """None for empty or asterisk-masked (redacted) text, else the text."""
    if not t:
        return None
    return None if _MASK.search(t) else t


def _text(node) -> str | None:
    if node is None:
        return None
    return _unmask(node.get_text(" ", strip=True))


def _ld_person(soup: BeautifulSoup) -> dict:
    """Return the schema.org Person node from any ld+json block, or {}."""
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        graph = data.get("@graph", [data]) if isinstance(data, dict) else data
        for node in graph if isinstance(graph, list) else []:
            if isinstance(node, dict) and node.get("@type") == "Person":
                return node
    return {}


def _has_text(s: str | None) -> bool:
    """True if s carries real content (letters/digits), not just separators."""
    return bool(s) and bool(re.search(r"[A-Za-z0-9]", s))


def _split_dates(text: str | None) -> tuple[str | None, str | None]:
    """'1973 - 1975' / '2000 - Present 26 years' -> (start, end); end None if present.

    LinkedIn appends a duration ('· 1 yr', 'Present 26 years') we strip off.
    """
    if not text:
        return None, None
    text = text.split("·")[0].strip()
    parts = _DASH.split(text, maxsplit=1)
    start = parts[0].strip() or None
    end = parts[1].strip() if len(parts) > 1 else None
    if end:
        if re.search(r"(?i)present|current", end):
            end = None
        else:
            end = re.sub(r"\s*\d+\s+(?:years?|yrs?|months?|mos?)\b.*$", "", end).strip() or None
    return start, end


def _headline_from_title(soup: BeautifulSoup) -> str | None:
    """<title> is 'Name - Headline | LinkedIn'."""
    title = _text(soup.title)
    if not title:
        return None
    title = re.sub(r"\s*\|\s*LinkedIn\s*$", "", title)
    if " - " in title:
        return title.split(" - ", 1)[1].strip() or None
    return None


def _parse_experience(soup: BeautifulSoup) -> list[Experience]:
    out: list[Experience] = []
    for li in soup.select("li.experience-item"):
        start, end = _split_dates(_text(li.select_one(".date-range")))
        out.append(Experience(
            title=_text(li.select_one(".experience-item__title")),
            company=_text(li.select_one(".experience-item__subtitle")),
            start=start,
            end=end,
            description=_text(li.select_one(".experience-item__description")),
        ))
    return out


def _parse_education(soup: BeautifulSoup) -> list[Education]:
    out: list[Education] = []
    for li in soup.select("li.education__list-item"):
        start, end = _split_dates(_text(li.select_one(".date-range")))
        # School is the heading link; degree/field lines are the following <h4>s.
        school = _text(li.find("h3"))
        subs = [_text(h) for h in li.find_all("h4")]
        subs = [s for s in subs if _has_text(s)]
        out.append(Education(
            school=school,
            degree=subs[0] if subs else None,
            field=subs[1] if len(subs) > 1 else None,
            start=start,
            end=end,
        ))
    return out


def _experience_from_ld(person: dict) -> list[Experience]:
    out: list[Experience] = []
    for org in person.get("worksFor") or []:
        if not isinstance(org, dict):
            continue
        member = org.get("member") or {}
        start = member.get("startDate")
        end = member.get("endDate")
        out.append(Experience(
            company=_unmask((org.get("name") or "").strip()),
            start=str(start) if start else None,
            end=str(end) if end else None,
        ))
    return out


def _education_from_ld(person: dict) -> list[Education]:
    out: list[Education] = []
    for org in person.get("alumniOf") or []:
        if not isinstance(org, dict):
            continue
        member = org.get("member") or {}
        start = member.get("startDate")
        end = member.get("endDate")
        out.append(Education(
            school=_unmask((org.get("name") or "").strip()),
            start=str(start) if start else None,
            end=str(end) if end else None,
        ))
    return out


def parse_profile(slug: str, html: str) -> Profile:
    soup = BeautifulSoup(html, "html.parser")
    person = _ld_person(soup)

    name = _unmask((person.get("name") or "").strip()) or _text(soup.find("h1"))
    bio = _unmask((person.get("description") or "").strip())

    location = None
    addr = person.get("address")
    if isinstance(addr, dict):
        location = _unmask(addr.get("addressLocality"))

    experience = _parse_experience(soup) or _experience_from_ld(person)
    education = _parse_education(soup) or _education_from_ld(person)
    # Drop entries left empty after unmasking (redacted-only rows carry no signal).
    experience = [e for e in experience if e.title or e.company or e.start]
    education = [e for e in education if e.school or e.start]

    return Profile(
        slug=slug,
        name=name,
        headline=_headline_from_title(soup),
        location=location,
        bio=bio,
        experience=experience,
        education=education,
    )


def has_masked_content(html: str) -> bool:
    """LinkedIn redacts fields for logged-out viewers of less-public profiles with
    runs of asterisks. Their presence means the public page is partially withheld."""
    return bool(re.search(r"\*{4,}", html))


def is_restricted(profile: Profile, html: str) -> bool:
    """Flag for the authenticated (Chrome) fallback when the public page yields no
    usable substance (no experience/education/bio) OR LinkedIn masked its content."""
    thin = not (profile.experience or profile.education or profile.bio)
    return thin or has_masked_content(html)
