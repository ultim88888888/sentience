"""doppelganger.registry — resolve a subject slug to its source locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from doppelganger import config


@dataclass
class SubjectRef:
    slug: str
    name: str
    x_handle: str | None
    linkedin_url: str | None
    linkedin_file: str | None     # "<segment>.json" or None


def _linkedin_file(url: str | None) -> str | None:
    if not url:
        return None
    seg = url.rstrip("/").split("/")[-1]
    return f"{seg}.json" if seg else None


def resolve_subject(slug: str, *, tracked_people_path: Path | None = None) -> SubjectRef:
    path = tracked_people_path or config.TRACKED_PEOPLE
    people = yaml.safe_load(Path(path).read_text())["people"]
    for p in people:
        if p["slug"] == slug:
            return SubjectRef(
                slug=slug, name=p.get("name", slug), x_handle=p.get("x_handle"),
                linkedin_url=p.get("linkedin_url"), linkedin_file=_linkedin_file(p.get("linkedin_url")),
            )
    raise KeyError(f"subject {slug!r} not found in {path}")
