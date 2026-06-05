"""Configuration for the a16z crypto team scraper.

Reuses the research scraper's scrape.do fetch layer (token, retry, concurrency);
this module only adds the team-roster URLs and output paths.
"""
from pathlib import Path

# The public team roster. Server-rendered — a plain scrape.do GET returns the full
# member grid (no JS render needed). Each member links to /team/<slug>.
TEAM_URL = "https://a16zcrypto.com/our-team/"
PROFILE_BASE = "https://a16zcrypto.com/team/"

# Output
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "a16z_team"
TEAM_PARQUET = DATA_DIR / "team.parquet"
