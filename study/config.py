"""Configuration for the a16z research-coverage trading-signal study."""
from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PARQUET = DATA_DIR / "a16z_research" / "articles.parquet"
PRICE_CACHE_DIR = DATA_DIR / "prices"
STUDY_DIR = DATA_DIR / "study"
BASKETS_YAML = Path(__file__).resolve().parent / "baskets.yaml"
FINDINGS_MD = PROJECT_ROOT / "findings.md"
HEATMAP_PNG = STUDY_DIR / "coverage_heatmap.png"

# --- Coinglass ---
COINGLASS_BASE = "https://open-api-v4.coinglass.com"
COINGLASS_OP_ITEM = "coinglass"
COINGLASS_OP_VAULT = "local"
EXCHANGE = "Binance"
PRICE_INTERVAL = "1d"
PRICE_LIMIT = 4500          # ~12yr daily; validated max per call
HTTP_TIMEOUT = 60.0
RATE_LIMIT_RETRIES = 5       # retries on rate-limit before giving up
RATE_LIMIT_BACKOFF = 8.0     # base seconds; doubles each retry (8,16,32,...)
REQUEST_THROTTLE = 2.5       # min seconds between live API calls

# --- signal params ---
MOMENTUM_LOOKBACK = 3        # months of trailing average for coverage_momentum
FWD_WINDOWS = (1, 3)         # forward-return windows in months
ATTRIBUTION_MODES = ("fractional", "full")
CONVICTION_AGGS = ("sum", "mean")
