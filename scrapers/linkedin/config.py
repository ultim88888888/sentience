"""Configuration for the LinkedIn profile scraper.

Approach: LinkedIn's Voyager API is blocked at the edge for proxied requests
(ROTATION_FAILED), and the authenticated profile page is an obfuscated React DOM.
The *public* profile page, by contrast, is a crawler-facing SSR template with a
clean schema.org `ld+json` Person node and stable, non-obfuscated section markup.
We fetch that via scrape.do (residential, to clear Cloudflare) — no auth needed —
and parse the ld+json plus the stable HTML sections. Completeness is therefore
bounded by each target's public-profile visibility.
"""
from pathlib import Path

# scrape.do transport
SCRAPEDO_BASE = "https://api.scrape.do/"
SCRAPEDO_OP_ITEM = "scrape.do"
OP_VAULT = "local"

# Target: the public profile page. {slug} is a vanity slug.
PROFILE_URL = "https://www.linkedin.com/in/{slug}/"

# scrape.do params: super=true uses a residential proxy (clears LinkedIn's
# Cloudflare WAF); geoCode pins the proxy region. No render needed — the public
# page is server-rendered. No cookies — we want the logged-out public template.
SCRAPEDO_GEO = "us"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")

# Politeness / resilience
REQUEST_DELAY = 4.0    # seconds between profiles
FETCH_RETRIES = 3
FETCH_TIMEOUT = 90.0

# Output
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "linkedin"
RAW_DIR = DATA_DIR / "raw"          # raw .html per profile (durable, re-parseable)
PARSED_DIR = DATA_DIR / "parsed"    # structured .json per profile
RESTRICTED_LIST = DATA_DIR / "_restricted.txt"   # slugs that came back thin/authwalled
