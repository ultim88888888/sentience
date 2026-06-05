"""Schema defaults for the LinkedIn profile models — pure, no network."""
from scrapers.linkedin.models import Education, Experience, Profile


def test_experience_defaults_all_optional():
    exp = Experience()
    assert exp.title is None
    assert exp.company is None
    assert exp.start is None
    assert exp.end is None
    assert exp.description is None


def test_education_defaults_all_optional():
    edu = Education()
    assert edu.school is None
    assert edu.degree is None
    assert edu.field is None


def test_profile_requires_slug_and_defaults_lists():
    p = Profile(slug="ada-lovelace")
    assert p.slug == "ada-lovelace"
    assert p.name is None
    assert p.experience == []
    assert p.education == []
