"""Parsing logic for the team scraper — pure, no network.

Fixtures mirror the real a16zcrypto markup, covering the two cases that bite:
multi-section dedup on the roster, and the publications <ul> that must NOT be
mistaken for the socials <ul> on research-member profiles.
"""
from scrapers.a16z_team.extract import extract
from scrapers.a16z_team.listing import parse_roster


def _card(slug, name, title):
    return (f'<div><a class="block group/member" href="/team/{slug}">'
            f'<div class="text-center">'
            f'<div class="capitalize"><span>{name}</span></div>'
            f'<div class="text-body--xs"><span>{title}</span></div>'
            f'</div></a></div>')


ROSTER = (
    '<div class="grid">'
    '<h2 class="flex-1 uppercase">Investing</h2>'
    + _card("eddy-lazzarin", "Eddy Lazzarin", "General Partner")
    + _card("ali-yahya", "Ali Yahya", "General Partner") +
    '<h2 class="flex-1 uppercase">Engineering</h2>'
    + _card("eddy-lazzarin", "Eddy Lazzarin", "General Partner")  # same person, 2nd section
    + _card("noah-citron", "Noah Citron", "Engineering") +
    '</div>'
)


def test_roster_parses_each_member():
    members = {m.slug: m for m in parse_roster(ROSTER)}
    assert set(members) == {"eddy-lazzarin", "ali-yahya", "noah-citron"}
    assert members["ali-yahya"].name == "Ali Yahya"
    assert members["ali-yahya"].listing_title == "General Partner"


def test_roster_dedups_and_merges_sections():
    members = {m.slug: m for m in parse_roster(ROSTER)}
    # Listed twice -> one record, both sections, in document order.
    assert members["eddy-lazzarin"].sections == ["Investing", "Engineering"]
    assert members["noah-citron"].sections == ["Engineering"]


# --- profile extraction -------------------------------------------------------

def _profile(title, bio_html, socials_ul="", extra_ul=""):
    return (
        f'<section class="pt-l"><div><div class="mb-l-xl">'
        f'<h1><span>Jane Doe</span></h1><div>{title}</div></div>'
        f'<div class="grid md:flex">'
        f'<div class="md:max-w-[308px]"><img></div>'
        f'<div class="flex-1 font-[350]"><div class="max-w-3xl">'
        f'<div>{bio_html}</div>{extra_ul}{socials_ul}'
        f'</div></div></div></div></section>'
    )

SOCIALS_UL = (
    '<ul class="text-body--sm flex flex-wrap gap-x-6">'
    '<li><a target="_blank" href="https://twitter.com/jane">X</a></li>'
    '<li><a target="_blank" href="https://www.linkedin.com/in/jane">LinkedIn</a></li>'
    '<li><a target="_blank" href="https://jane.dev/">Web</a></li>'
    '</ul>'
)
PUBLICATIONS_UL = (
    '<p>Representative works include:</p>'
    '<ul><li><a href="https://x/paper1">Twenty Lectures on Algorithmic Game Theory</a></li>'
    '<li><a href="https://x/paper2">A Simple Theory of Vampire Attacks</a></li></ul>'
)


def test_extract_name_title_bio_socials():
    p = extract(_profile("General Partner", "<p>First para.</p><p>Second para.</p>", SOCIALS_UL))
    assert p.name == "Jane Doe"
    assert p.title == "General Partner"
    assert p.bio == "First para.\n\nSecond para."
    assert [(s["platform"], s["url"]) for s in p.socials] == [
        ("x", "https://twitter.com/jane"),
        ("linkedin", "https://www.linkedin.com/in/jane"),
        ("website", "https://jane.dev/"),
    ]


def test_publications_ul_not_mistaken_for_socials():
    # Research member: a publications <ul> precedes the real socials <ul>.
    p = extract(_profile("Research", "<p>Bio.</p>", SOCIALS_UL, extra_ul=PUBLICATIONS_UL))
    platforms = [s["platform"] for s in p.socials]
    assert platforms == ["x", "linkedin", "website"]  # papers excluded
    assert all("paper" not in s["url"] for s in p.socials)


def test_member_without_socials():
    p = extract(_profile("Operations", "<p>Bio only.</p>"))
    assert p.bio == "Bio only." and p.socials == []


def test_member_without_bio():
    # Stub profile: empty bio column, no paragraphs.
    p = extract(_profile("Executive Assistant", ""))
    assert p.title == "Executive Assistant"
    assert p.bio is None and p.socials == []


def test_empty_html_is_safe():
    p = extract("")
    assert p.name is None and p.bio is None and p.socials == []
