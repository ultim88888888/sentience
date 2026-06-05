"""Paginate the Algolia index for all research posts and normalize each hit."""
import html as _html
import json
from datetime import datetime, timezone

import httpx

from .config import FACET_FILTERS, HITS_PER_PAGE
from .keys import AlgoliaKey, mint_key


def _query_url(key: AlgoliaKey) -> str:
    return f"https://{key.app_id}-dsn.algolia.net/1/indexes/{key.index}/query"


def _headers(key: AlgoliaKey) -> dict:
    return {
        "Content-Type": "application/json",
        "X-Algolia-Application-Id": key.app_id,
        "X-Algolia-API-Key": key.key,
        "Origin": "https://a16zcrypto.com",
        "Referer": "https://a16zcrypto.com/",
    }


def fetch_raw_hits(key: AlgoliaKey | None = None) -> list[dict]:
    """Return every raw Algolia hit for the research category, re-minting the key if it expires."""
    if key is None:
        key = mint_key()
    hits: list[dict] = []
    page = 0
    with httpx.Client(timeout=30) as client:
        while True:
            if key.expired:
                key = mint_key()
            body = {"query": "", "hitsPerPage": HITS_PER_PAGE, "page": page,
                    "facetFilters": FACET_FILTERS}
            r = client.post(_query_url(key), headers=_headers(key), json=body)
            if r.status_code in (401, 403):  # key died early -> re-mint and retry page
                key = mint_key()
                continue
            r.raise_for_status()
            j = r.json()
            hits.extend(j["hits"])
            if page >= j["nbPages"] - 1:
                break
            page += 1
    return hits


def _epoch_to_iso(v) -> str | None:
    try:
        return datetime.fromtimestamp(int(v), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _u(v):
    """Unescape HTML entities in a string or list of strings."""
    if isinstance(v, str):
        return _html.unescape(v)
    if isinstance(v, list):
        return [_html.unescape(x) if isinstance(x, str) else x for x in v]
    return v


def _canonical(url: str | None) -> str | None:
    """Algolia stores backend permalinks on cms.a16zcrypto.com; the public canonical
    host is a16zcrypto.com (the cms host 404s for some posts)."""
    if not url:
        return url
    return url.replace("://cms.a16zcrypto.com/", "://a16zcrypto.com/", 1)


def normalize_hit(h: dict) -> dict:
    """Flatten an Algolia hit into a tabular metadata record."""
    tax = h.get("taxonomies") or {}
    author = h.get("post_author") or {}
    acf = _u(h.get("acf_content") or "")
    return {
        "object_id": h.get("objectID"),
        "post_id": h.get("post_id"),
        "title": _u(h.get("post_title")),
        "permalink": _canonical(h.get("permalink")),
        "permalink_raw": h.get("permalink"),
        "post_date": _epoch_to_iso(h.get("post_date")),
        "post_modified": _epoch_to_iso(h.get("post_modified")),
        "post_date_formatted": h.get("post_date_formatted"),
        "categories": _u(tax.get("category") or []),
        "tags": _u(tax.get("post_tag") or []),
        "formats": _u(tax.get("format") or []),
        "author_slugs": tax.get("author") or [],
        "poster_display_name": author.get("display_name"),
        "description": _u(h.get("post_description") or ""),
        "excerpt": _u(h.get("post_excerpt") or ""),
        "meta_description": _u(h.get("post_meta_description") or ""),
        "acf_content": acf,  # Algolia's plain-text body (bonus)
        "acf_content_len": len(acf),
        # images is str on some posts, struct on others -> store as JSON text for a stable schema.
        "images": json.dumps(h.get("images")) if h.get("images") else "",
    }
