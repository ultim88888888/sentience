"""Configuration and seed taxonomy for the A1 signal pipeline."""
from pathlib import Path

DATA_DIR = Path("data")
TWITTER_DIR = DATA_DIR / "twitter"
RESEARCH_ARTICLES = DATA_DIR / "a16z_research" / "articles.parquet"
TRANSCRIPTS = DATA_DIR / "a16z_research" / "transcripts.parquet"
TRACKED_PEOPLE = DATA_DIR / "tracked_people.yaml"

SIGNAL_OUT_DIR = DATA_DIR / "signal"
DISTILLATE_CACHE = SIGNAL_OUT_DIR / "transcript_distillates.jsonl"
TWEET_DISTILLATE_CACHE = SIGNAL_OUT_DIR / "tweet_distillates.jsonl"
REGISTRY_PATH = SIGNAL_OUT_DIR / "registry.json"
PANEL_PATH = SIGNAL_OUT_DIR / "signal_panel.parquet"

DEFAULT_WINDOW_MONTHS = 18  # holding-period scale; test 24 (see spec stage 2)

# Seed sector taxonomy — precise, lowercase-kebab ids. The LLM fits to one of these
# or mints a new one (semantic judgment); see signals/canonicalize.py.
SEED_SECTORS = [
    "liquid-staking",
    "restaking",
    "l2-scaling",
    "zk",
    "pos-l1",
    "pow-l1",
    "modular-da",
    "defi",
    "stablecoins",
    "perp-dex",
    "gaming",
    "nft",
    "depin",
    "payments",
    "rwa",
    "ai-crypto",
    "infra-devtools",
    "privacy",
    "governance-dao",
    "consumer-social",
]

STANCE_SIGN = {"bullish": 1, "neutral": 0, "bearish": -1}
