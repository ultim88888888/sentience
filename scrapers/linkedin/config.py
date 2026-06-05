"""Configuration for the LinkedIn profile scraper."""
from pathlib import Path

# scrape.do transport
SCRAPEDO_BASE = "https://api.scrape.do/"
SCRAPEDO_OP_ITEM = "scrape.do"
OP_VAULT = "local"

# LinkedIn auth secrets — 1Password items in vault `local` (Jax creates these).
LI_AT_OP_ITEM = "linkedin_li_at"
JSESSIONID_OP_ITEM = "linkedin_jsessionid"

# Voyager API — LinkedIn's own internal JSON endpoint for full profile data.
VOYAGER_PROFILE_VIEW = (
    "https://www.linkedin.com/voyager/api/identity/profiles/{slug}/profileView"
)

# Session-security guard: a fixed sticky session pins ONE residential IP for the
# whole run, geo-matched to where the account normally logs in. This — not request
# volume — is what trips LinkedIn's "sign-in from a new location" lockout.
SCRAPEDO_SESSION_ID = "778899"   # any fixed value -> same upstream IP across the run
SCRAPEDO_GEO = "us"               # match the account's usual login region

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")

# Politeness / resilience
REQUEST_DELAY = 4.0    # seconds between profiles
FETCH_RETRIES = 3
FETCH_TIMEOUT = 60.0

# Output
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "linkedin"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
