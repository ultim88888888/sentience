"""Extract name, title, bio, and social accounts from a member profile page.

Profile layout (stable across the roster):

    <section>
      <h1><span>Name</span></h1>
      <div>Title</div>                         <!-- sibling right after h1 -->
      …
      <div class="… flex-1 …">                  <!-- the bio column -->
        <p>…bio paragraph…</p> …
        <ul><li><a href="…">X</a></li> …</ul>  <!-- the person's own socials -->
      </div>
    </section>

Scoping the bio/socials to the profile <section> (anchored on the h1) keeps the
site-wide footer socials (a16zcrypto, a16z) out of the result.
"""
import json
import urllib.parse
from dataclasses import dataclass, field

import lxml.html

# Map a social link to a normalized platform key, by link label first then host.
_LABEL_PLATFORM = {
    "x": "x", "twitter": "x", "linkedin": "linkedin", "farcaster": "farcaster",
    "warpcast": "farcaster", "github": "github", "web": "website",
    "website": "website", "instagram": "instagram", "youtube": "youtube",
    "substack": "substack", "medium": "medium",
}
_HOST_PLATFORM = {
    "twitter.com": "x", "x.com": "x", "linkedin.com": "linkedin",
    "farcaster.xyz": "farcaster", "warpcast.com": "farcaster",
    "github.com": "github", "instagram.com": "instagram",
    "youtube.com": "youtube", "substack.com": "substack", "medium.com": "medium",
}


def _platform(label: str, url: str) -> str:
    key = (label or "").strip().lower()
    if key in _LABEL_PLATFORM:
        return _LABEL_PLATFORM[key]
    host = urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    return _HOST_PLATFORM.get(host, key or "other")


@dataclass
class Profile:
    name: str | None = None
    title: str | None = None         # authoritative title from the profile page
    bio: str | None = None
    socials: list[dict] = field(default_factory=list)  # [{platform, label, url}]


def extract(html: str) -> Profile:
    """Parse a profile page; missing fields come back as None / [] rather than raising."""
    if not html:
        return Profile()
    root = lxml.html.fromstring(html)
    h1 = root.xpath("//h1")
    if not h1:
        return Profile()
    h1 = h1[0]
    name = " ".join(t.strip() for t in h1.itertext() if t.strip()) or None

    sec = h1.xpath("./ancestor::section[1]")
    scope = sec[0] if sec else root

    title_el = h1.xpath("following-sibling::div[1]")
    title = title_el[0].text_content().strip() if title_el else None

    bio, socials = None, []
    # The bio column is the first flex-1 div in the section that carries prose <p>s.
    for blk in scope.xpath(
            ".//div[contains(concat(' ',normalize-space(@class),' '),' flex-1 ')]"):
        ps = blk.xpath(".//p")
        if not ps:
            continue
        bio = "\n\n".join(p.text_content().strip() for p in ps).strip() or None
        # The socials list is the flex-wrap <ul>; a plain <ul> in the same block is a
        # publications list (research members) and must not be mistaken for socials.
        for a in blk.xpath(".//ul[contains(@class,'flex-wrap')]/li/a[@href]"):
            url = a.get("href").strip()
            label = a.text_content().strip()
            socials.append({"platform": _platform(label, url),
                            "label": label, "url": url})
        break
    return Profile(name=name, title=title, bio=bio, socials=socials)


def socials_json(socials: list[dict]) -> str:
    return json.dumps(socials, ensure_ascii=False)
