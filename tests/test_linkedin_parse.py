"""Parse public LinkedIn profile HTML -> Profile. Pure, no network.

The primary fixture is real (trimmed) markup from a public profile, so the parser
is tested against LinkedIn's actual SSR structure, not a hand-drawn approximation.
"""
from pathlib import Path

from scrapers.linkedin.parse import has_masked_content, is_restricted, parse_profile

FIXTURE = (Path(__file__).parent / "fixtures" / "linkedin_public_profile.html").read_text()


def test_parse_identity_and_bio_from_ldjson():
    p = parse_profile("williamhgates", FIXTURE)
    assert p.name == "Bill Gates"
    assert p.headline == "Chair, Gates Foundation and Founder, Breakthrough Energy"
    assert p.location == "Seattle, Washington, United States"
    assert p.bio.startswith("Chair of the Gates Foundation")
    assert is_restricted(p, FIXTURE) is False


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
    assert is_restricted(p, html) is False


def test_thin_profile_detected():
    html = "<html><head><title>Sign Up | LinkedIn</title></head><body></body></html>"
    p = parse_profile("nobody", html)
    assert p.experience == []
    assert p.education == []
    assert p.bio is None
    assert is_restricted(p, html) is True


def test_masked_fields_are_nulled_and_profile_flagged_restricted():
    # LinkedIn redacts logged-out views of less-public profiles with asterisk runs.
    # Masked orgs must NOT become junk experience entries, and the profile must be
    # flagged restricted even though name/headline/one real school came through.
    html = """
    <html><head><title>Eddy - Andreessen Horowitz | LinkedIn</title>
    <script type="application/ld+json">
    {"@type":"Person","name":"Eddy Lazzarin",
     "worksFor":[{"@type":"Organization","name":"********** ********",
                  "member":{}}],
     "alumniOf":[{"@type":"EducationalOrganization","name":"Washington University in St. Louis",
                  "member":{"startDate":2010,"endDate":2011}},
                 {"@type":"EducationalOrganization","name":"**********",
                  "member":{"startDate":2006,"endDate":2010}}]}
    </script></head><body></body></html>
    """
    p = parse_profile("eddy", html)
    assert p.name == "Eddy Lazzarin"
    assert p.experience == []                       # masked org dropped, not "****"
    schools = [(e.school, e.start, e.end) for e in p.education]
    assert ("Washington University in St. Louis", "2010", "2011") in schools
    assert (None, "2006", "2010") in schools        # masked school name nulled, dates kept
    assert has_masked_content(html) is True
    assert is_restricted(p, html) is True           # flagged for Chrome fallback
