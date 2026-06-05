"""Parse public LinkedIn profile HTML -> Profile. Pure, no network.

The primary fixture is real (trimmed) markup from a public profile, so the parser
is tested against LinkedIn's actual SSR structure, not a hand-drawn approximation.
"""
from pathlib import Path

from scrapers.linkedin.parse import is_thin, parse_profile

FIXTURE = (Path(__file__).parent / "fixtures" / "linkedin_public_profile.html").read_text()


def test_parse_identity_and_bio_from_ldjson():
    p = parse_profile("williamhgates", FIXTURE)
    assert p.name == "Bill Gates"
    assert p.headline == "Chair, Gates Foundation and Founder, Breakthrough Energy"
    assert p.location == "Seattle, Washington, United States"
    assert p.bio.startswith("Chair of the Gates Foundation")
    assert is_thin(p) is False


def test_parse_experience_from_html_sections():
    p = parse_profile("williamhgates", FIXTURE)
    titles = [(e.title, e.company) for e in p.experience]
    assert ("Co-chair", "Gates Foundation") in titles
    assert ("Co-founder", "Microsoft") in titles
    first = p.experience[0]
    assert first.start == "2000"
    assert first.end is None          # 'Present 26 years' -> None, duration stripped


def test_parse_education_from_html_sections():
    p = parse_profile("williamhgates", FIXTURE)
    schools = {e.school: e for e in p.education}
    assert "Harvard University" in schools
    harvard = schools["Harvard University"]
    assert harvard.start == "1973"
    assert harvard.end == "1975"
    assert harvard.degree is None     # separator artifacts filtered out


def test_falls_back_to_ldjson_lists_when_no_html_sections():
    # A profile page with only the ld+json Person and no experience/education
    # sections (e.g. a sparser public view) still yields org/school + years.
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type":"Person","name":"Grace Hopper","description":"Rear admiral.",
     "worksFor":[{"@type":"Organization","name":"US Navy",
                  "member":{"startDate":1943,"endDate":1986}}],
     "alumniOf":[{"@type":"EducationalOrganization","name":"Yale",
                  "member":{"startDate":1930,"endDate":1934}}]}
    </script></head><body></body></html>
    """
    p = parse_profile("grace-hopper", html)
    assert p.name == "Grace Hopper"
    assert p.bio == "Rear admiral."
    assert [(e.company, e.start, e.end) for e in p.experience] == [("US Navy", "1943", "1986")]
    assert [(e.school, e.start, e.end) for e in p.education] == [("Yale", "1930", "1934")]
    assert is_thin(p) is False


def test_thin_profile_detected():
    p = parse_profile("nobody", "<html><head><title>Sign Up | LinkedIn</title></head><body></body></html>")
    assert p.experience == []
    assert p.education == []
    assert p.bio is None
    assert is_thin(p) is True
