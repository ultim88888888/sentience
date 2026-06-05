"""CLI: scrape public LinkedIn profiles to data/linkedin/{raw,parsed}/.

Public-first: full data lands automatically for public profiles. Targets that come
back thin (restricted for logged-out viewers) are flagged in `_restricted.txt` for
an authenticated (Chrome) fallback pull.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

from .config import PARSED_DIR, RAW_DIR, REQUEST_DELAY, RESTRICTED_LIST
from .fetch import fetch_profile, scrapedo_token
from .parse import is_restricted, parse_profile


def normalize_slug(value: str) -> str:
    """A vanity slug or a full /in/ URL -> bare slug."""
    value = value.strip().rstrip("/")
    if "linkedin.com/in/" in value:
        value = value.split("linkedin.com/in/", 1)[1].split("/")[0].split("?")[0]
    return value


def read_slugs(arg: str) -> list[str]:
    """arg is a file path (one slug/URL per line) or a single slug/URL."""
    p = Path(arg)
    lines = p.read_text().splitlines() if p.exists() else [arg]
    return [normalize_slug(x) for x in lines if x.strip()]


async def main(slugs: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    token = scrapedo_token()
    restricted: list[str] = []
    full = 0
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for i, slug in enumerate(slugs, 1):
            print(f"[{i}/{len(slugs)}] {slug}", file=sys.stderr)
            result = await fetch_profile(client, token, slug)
            if result.status != 200 or not result.html:
                print(f"  FAILED ({result.error}) -> flagged", file=sys.stderr)
                restricted.append(slug)
                continue
            (RAW_DIR / f"{slug}.html").write_text(result.html)
            profile = parse_profile(slug, result.html)
            (PARSED_DIR / f"{slug}.json").write_text(profile.model_dump_json(indent=2))
            if is_restricted(profile, result.html):
                print(f"  RESTRICTED/THIN (name={profile.name!r}) -> flagged for Chrome fallback",
                      file=sys.stderr)
                restricted.append(slug)
            else:
                full += 1
                print(f"  ok ({len(profile.experience)} exp, {len(profile.education)} edu)",
                      file=sys.stderr)
            if i < len(slugs):
                await asyncio.sleep(REQUEST_DELAY)

    if restricted:
        RESTRICTED_LIST.write_text("\n".join(restricted) + "\n")
    print(f"\nDone: {full} full, {len(restricted)} thin/failed.", file=sys.stderr)
    if restricted:
        print(f"Thin/restricted -> {RESTRICTED_LIST} (pull via authenticated Chrome):",
              file=sys.stderr)
        for s in restricted:
            print(f"  - {s}", file=sys.stderr)


def cli() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m scrapers.linkedin.run <slug|url|file>", file=sys.stderr)
        raise SystemExit(1)
    asyncio.run(main(read_slugs(sys.argv[1])))


if __name__ == "__main__":
    cli()
