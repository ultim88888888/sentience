"""Parse the team roster page into one record per member.

The roster groups members under section headers (Investing, Engineering, Research,
…, Advisors). A person can appear under more than one section (e.g. Eddy Lazzarin
in both Engineering and Research), so members are deduped by slug and their sections
merged.
"""
from dataclasses import dataclass, field

import lxml.html


@dataclass
class Member:
    slug: str                       # /team/<slug> — the profile path + a stable id
    name: str
    listing_title: str              # title as shown on the roster card
    sections: list[str] = field(default_factory=list)  # all sections the person is listed under


def _text(el) -> str:
    return " ".join(t.strip() for t in el.itertext() if t.strip())


def parse_roster(html: str) -> list[Member]:
    """Return the deduped member roster from the team page HTML."""
    root = lxml.html.fromstring(html)
    # Section headers and member anchors, in document order, so each member inherits
    # the most recent header above it.
    nodes = root.xpath(
        "//h2[contains(@class,'uppercase')] | //a[contains(@class,'group/member')]")
    by_slug: dict[str, Member] = {}
    section = None
    for node in nodes:
        if node.tag == "h2":
            section = _text(node) or None
            continue
        href = node.get("href") or ""
        slug = href.rstrip("/").rsplit("/", 1)[-1]
        if not slug or "/team/" not in href:
            continue
        # The two stacked <div>s inside .text-center hold name then title.
        cells = node.xpath(".//div[contains(@class,'text-center')]/div")
        name = _text(cells[0]) if cells else _text(node)
        title = _text(cells[1]) if len(cells) > 1 else ""
        m = by_slug.get(slug)
        if m is None:
            m = by_slug[slug] = Member(slug=slug, name=name, listing_title=title)
        if section and section not in m.sections:
            m.sections.append(section)
    return list(by_slug.values())
