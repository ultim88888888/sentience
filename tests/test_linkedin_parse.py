"""Parse Voyager profileView JSON -> Profile. Pure, no network.

The fixture mirrors the real profileView decoration: top-level `profile`,
`positionView.elements`, `educationView.elements`, with Voyager date shapes.
"""
from scrapers.linkedin.parse import parse_profile

FIXTURE = {
    "profile": {
        "firstName": "Ada",
        "lastName": "Lovelace",
        "headline": "Mathematician | First Programmer",
        "locationName": "London, England, United Kingdom",
        "summary": "Pioneer of computing.",
    },
    "positionView": {
        "elements": [
            {
                "title": "Analyst",
                "companyName": "Analytical Engine Project",
                "description": "Wrote the first algorithm.",
                "timePeriod": {
                    "startDate": {"month": 6, "year": 1842},
                    "endDate": {"month": 8, "year": 1843},
                },
            },
            {
                "title": "Correspondent",
                "companyName": "Royal Society",
                "timePeriod": {"startDate": {"year": 1840}},
            },
        ]
    },
    "educationView": {
        "elements": [
            {
                "schoolName": "Self-taught",
                "degreeName": "Mathematics",
                "fieldOfStudy": "Analysis",
                "timePeriod": {
                    "startDate": {"year": 1832},
                    "endDate": {"year": 1835},
                },
            }
        ]
    },
}


def test_parse_identity_and_bio():
    p = parse_profile("ada-lovelace", FIXTURE)
    assert p.slug == "ada-lovelace"
    assert p.name == "Ada Lovelace"
    assert p.headline == "Mathematician | First Programmer"
    assert p.location == "London, England, United Kingdom"
    assert p.bio == "Pioneer of computing."


def test_parse_experience_with_date_formats():
    p = parse_profile("ada-lovelace", FIXTURE)
    assert len(p.experience) == 2
    first = p.experience[0]
    assert first.title == "Analyst"
    assert first.company == "Analytical Engine Project"
    assert first.start == "1842-06"
    assert first.end == "1843-08"
    assert first.description == "Wrote the first algorithm."
    # year-only start, no end -> present
    second = p.experience[1]
    assert second.start == "1840"
    assert second.end is None


def test_parse_education():
    p = parse_profile("ada-lovelace", FIXTURE)
    assert len(p.education) == 1
    edu = p.education[0]
    assert edu.school == "Self-taught"
    assert edu.degree == "Mathematics"
    assert edu.field == "Analysis"
    assert edu.start == "1832"
    assert edu.end == "1835"


def test_parse_missing_summary_is_none():
    payload = {"profile": {"firstName": "Grace", "lastName": "Hopper"}}
    p = parse_profile("grace-hopper", payload)
    assert p.name == "Grace Hopper"
    assert p.bio is None
    assert p.experience == []
    assert p.education == []


def test_parse_empty_payload():
    p = parse_profile("nobody", {})
    assert p.slug == "nobody"
    assert p.name is None
    assert p.experience == []
    assert p.education == []
