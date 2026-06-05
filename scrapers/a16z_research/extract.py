"""Extract clean main-article content from fetched HTML using trafilatura."""
import trafilatura


def extract(html: str, url: str | None = None) -> dict:
    """Return clean text, markdown, and basic doc metadata from an article page.

    Falls back gracefully: empty/failed extraction yields empty strings, never raises.
    """
    if not html:
        return _empty()
    text = trafilatura.extract(
        html, url=url, output_format="txt",
        include_links=False, include_images=False,
        include_tables=True, favor_recall=True) or ""
    markdown = trafilatura.extract(
        html, url=url, output_format="markdown",
        include_links=True, include_images=True,
        include_tables=True, favor_recall=True) or ""
    meta_title = meta_author = meta_date = None
    try:
        md = trafilatura.extract_metadata(html)
        if md:
            meta_title, meta_author, meta_date = md.title, md.author, md.date
    except Exception:
        pass
    return {
        "extracted_text": text,
        "extracted_markdown": markdown,
        "extracted_text_len": len(text),
        "meta_title": meta_title,
        "meta_author": meta_author,
        "meta_date": meta_date,
    }


def _empty() -> dict:
    return {"extracted_text": "", "extracted_markdown": "", "extracted_text_len": 0,
            "meta_title": None, "meta_author": None, "meta_date": None}
