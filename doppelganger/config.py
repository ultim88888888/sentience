"""doppelganger.config — paths and tuning constants for ingestion.

All paths are relative to the sentience repo root (the process CWD when run via
`python -m doppelganger.run`). Override DATA_DIR in tests by passing explicit paths.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")

# --- Source locations ---
TWITTER_DIR = DATA_DIR / "twitter"                              # <x_handle>.parquet
RESEARCH_ARTICLES = DATA_DIR / "a16z_research" / "articles.parquet"
TEAM_PARQUET = DATA_DIR / "a16z_team" / "team.parquet"
LINKEDIN_DIR = DATA_DIR / "linkedin" / "parsed"                 # <linkedin-slug>.json
# Produced by the corpus-attribution session (branch doppelganger/corpus-attribution).
# Must be merged/copied to this path before the podcast adapter runs on real data.
ATTRIBUTED_TRANSCRIPTS = DATA_DIR / "a16z_research" / "attributed_transcripts.jsonl"
TRACKED_PEOPLE = DATA_DIR / "tracked_people.yaml"

# --- Output ---
OUT_DIR = DATA_DIR / "doppelganger"                            # <slug>/evidence.parquet, identity.json

# --- Tuning (eval-tuned later; documented defaults, not unexamined) ---
MIN_REPLY_CONTENT_CHARS = 50    # X replies with less substantive content than this are dropped
PODCAST_MIN_CONFIDENCE = 0.8    # keep podcast segments at/above this attribution confidence

SOURCE_TYPES = {
    "x_original", "x_quote", "x_reply", "research", "research_firm", "podcast",
}
