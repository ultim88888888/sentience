"""CLI: scrape a list of LinkedIn profiles to data/linkedin/{raw,parsed}/."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

from .auth import load_auth
from .config import PARSED_DIR, RAW_DIR, REQUEST_DELAY
from .fetch import AuthExpiredError, fetch_profile
from .parse import parse_profile


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
    auth = load_auth()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for i, slug in enumerate(slugs, 1):
            print(f"[{i}/{len(slugs)}] {slug}", file=sys.stderr)
            try:
                result = await fetch_profile(client, auth, slug)
            except AuthExpiredError as e:
                print(f"FATAL: {e}", file=sys.stderr)
                raise SystemExit(2)
            if result.status != 200 or result.payload is None:
                print(f"  skip ({result.error})", file=sys.stderr)
                continue
            (RAW_DIR / f"{slug}.json").write_text(json.dumps(result.payload, indent=2))
            profile = parse_profile(slug, result.payload)
            (PARSED_DIR / f"{slug}.json").write_text(profile.model_dump_json(indent=2))
            print(f"  ok ({len(profile.experience)} exp, {len(profile.education)} edu)",
                  file=sys.stderr)
            if i < len(slugs):
                await asyncio.sleep(REQUEST_DELAY)


def cli() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m scrapers.linkedin.run <slug|url|file>", file=sys.stderr)
        raise SystemExit(1)
    asyncio.run(main(read_slugs(sys.argv[1])))


if __name__ == "__main__":
    cli()
