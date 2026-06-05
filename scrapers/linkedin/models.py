"""Structured LinkedIn profile schema."""
from __future__ import annotations

from pydantic import BaseModel


class Experience(BaseModel):
    title: str | None = None
    company: str | None = None
    start: str | None = None        # "YYYY-MM" or "YYYY"
    end: str | None = None          # None == present
    description: str | None = None


class Education(BaseModel):
    school: str | None = None
    degree: str | None = None
    field: str | None = None
    start: str | None = None
    end: str | None = None


class Profile(BaseModel):
    slug: str
    name: str | None = None
    headline: str | None = None
    location: str | None = None
    bio: str | None = None
    experience: list[Experience] = []
    education: list[Education] = []
