"""Configuration for the a16z crypto research scraper."""
from pathlib import Path

# The research listing page — loading it lets AWS WAF challenge.js solve and the
# page mint a short-lived (300s) secured Algolia key via POST /api/generate-key.
RESEARCH_URL = "https://a16zcrypto.com/posts/focus-areas/research/"
KEY_ENDPOINT = "https://a16zcrypto.com/api/generate-key"

# Algolia query that defines the "research" corpus (taxonomy category == research).
FACET_FILTERS = [["taxonomies.category:research"], "post_type:post"]
HITS_PER_PAGE = 100  # Algolia max per page; 235 hits -> 3 pages

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")

# scrape.do — article pages are server-rendered, so plain GET (no JS render) suffices.
SCRAPEDO_BASE = "https://api.scrape.do/"
SCRAPEDO_OP_ITEM = "scrape.do"
SCRAPEDO_OP_VAULT = "local"
CONCURRENCY = 90
FETCH_RETRIES = 3
FETCH_TIMEOUT = 90.0

# Output
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "a16z_research"
METADATA_PARQUET = DATA_DIR / "metadata.parquet"
ARTICLES_PARQUET = DATA_DIR / "articles.parquet"
