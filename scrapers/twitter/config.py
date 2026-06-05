"""Configuration for the twitterapi.io scraper."""
from pathlib import Path

# twitterapi.io REST API
BASE = "https://api.twitterapi.io"
ADVANCED_SEARCH_PATH = "/twitter/tweet/advanced_search"

# API key in 1Password (pulled at runtime via `op`, never hardcoded).
OP_ITEM = "twitterapi-io"
OP_VAULT = "local"

HTTP_TIMEOUT = 60.0

# Self-imposed pacing. twitterapi.io tolerates high QPS, but we pace to be polite
# and keep cost predictable. Sequential cursor walk per user; concurrency is across users.
RATE_LIMIT_PER_SEC = 10.0     # max request issuances/sec, shared across all users
MAX_CONCURRENCY = 5           # max users fetched simultaneously

# Backoff for 429/5xx retries
_BACKOFF_BASE = 4.0           # seconds
_BACKOFF_FACTOR = 2.0
_BACKOFF_MAX_RETRIES = 4

# Safety guard: cap pages per user (20 tweets/page). 5000 pages = 100k tweets.
MAX_PAGES = 5000

# Output
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "twitter"

# Curated columns, in order. `raw_json` holds the full tweet object.
COLUMNS = [
    "id", "created_at", "type", "text", "lang",
    "author_id", "author_username", "author_name",
    "reply_count", "retweet_count", "like_count", "quote_count",
    "view_count", "bookmark_count",
    "conversation_id", "in_reply_to_id", "in_reply_to_user_id", "in_reply_to_username",
    "quoted_id", "retweeted_id", "url", "raw_json",
]
