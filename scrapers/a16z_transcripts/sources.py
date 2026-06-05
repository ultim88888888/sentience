"""Decide, per corpus post, where its transcript comes from.

Parses the stored ``raw_html`` to recover media handles (YouTube id, Simplecast uuid) and
routes each video/podcast post to a transcript source. Pure functions over a DataFrame row
— no network — so routing is unit-testable without fetching anything.
"""
import re
from dataclasses import dataclass

# YouTube id in an embed/watch/short url. Ids are exactly 11 url-safe chars.
_YT_RE = re.compile(r"(?:youtube\.com/(?:embed/|watch\?v=)|youtu\.be/)([A-Za-z0-9_-]{11})")
# Simplecast player embeds carry the episode uuid: player.simplecast.com/<uuid>
_SIMPLECAST_RE = re.compile(r"player\.simplecast\.com/([0-9a-fA-F-]{36})")

YOUTUBE = "youtube"
WHISPER = "whisper"
NONE = "none"


@dataclass
class Route:
    object_id: str
    title: str
    fmt: str            # 'videos' | 'podcasts'
    source: str         # YOUTUBE | WHISPER | NONE
    media_id: str | None  # youtube id (YOUTUBE) or simplecast uuid (WHISPER)
    note: str = ""


def _first(pattern, html: str) -> str | None:
    if not isinstance(html, str):
        return None
    m = pattern.search(html)
    return m.group(1) if m else None


def _fmt_of(formats) -> str | None:
    """Corpus stores ``formats`` as a list-like (e.g. ['videos']). Return the single tag."""
    if formats is None:
        return None
    try:
        seq = list(formats)
    except TypeError:
        return None
    return str(seq[0]) if seq else None


def route_row(row) -> Route | None:
    """Route one corpus row. Returns None for formats we don't transcribe (articles/papers)."""
    fmt = _fmt_of(row.get("formats"))
    if fmt not in ("videos", "podcasts"):
        return None

    html = row.get("raw_html") or ""
    oid = str(row.get("object_id"))
    title = str(row.get("title") or "")

    yt = _first(_YT_RE, html)
    if yt:
        return Route(oid, title, fmt, YOUTUBE, yt)

    # No YouTube embed. Videos should always have one — flag if not. Podcasts fall back to
    # Simplecast audio + local whisper.
    simc = _first(_SIMPLECAST_RE, html)
    if fmt == "podcasts" and simc:
        return Route(oid, title, fmt, WHISPER, simc)

    note = "video without youtube embed" if fmt == "videos" else "podcast without youtube or simplecast"
    return Route(oid, title, fmt, NONE, None, note=note)


def route_corpus(df) -> list[Route]:
    """Route every transcribable post in the corpus DataFrame."""
    routes = []
    for _, row in df.iterrows():
        r = route_row(row)
        if r is not None:
            routes.append(r)
    return routes
