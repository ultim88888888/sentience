"""Configuration for the a16z transcript collector.

Enriches the a16z research corpus (which is body-shallow for videos/podcasts — see the
study design spec §8) with full spoken transcripts:

  * videos  -> YouTube captions (every video page embeds a YouTube id)
  * podcasts with a YouTube id -> YouTube captions
  * podcasts without one        -> Simplecast audio -> local mlx-whisper transcription

Output is a join table keyed on ``object_id`` so the read-only corpus parquet is never
mutated; a downstream step can merge transcripts into article bodies as needed.
"""
from pathlib import Path

# ── Inputs / outputs ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "a16z_research"
CORPUS_PARQUET = DATA_DIR / "articles.parquet"
TRANSCRIPTS_PARQUET = DATA_DIR / "transcripts.parquet"
AUDIO_CACHE = DATA_DIR / "audio_cache"  # downloaded mp3s (gitignored, re-fetchable)

# ── YouTube caption fetch ───────────────────────────────────────────────────────────
# YouTube hard-blocks this IP after ~40 direct caption pulls (IpBlocked), so every request
# is routed through scrape.do's *residential* rotating proxy (super=true) — each request
# gets a fresh residential IP, sidestepping the block. Datacenter proxies are also blocked
# by YouTube, hence super=true. scrape.do MITMs TLS, so the session must skip cert verify.
YT_LANGUAGES = ["en", "en-US", "en-GB"]   # preference order; falls back to any if absent
YT_PAUSE_S = 0.2                          # small politeness delay (IPs rotate, no per-IP cap)
YT_RETRIES = 3
YT_BACKOFF_S = 4.0                        # multiplied by attempt number

# scrape.do proxy (token in 1Password, same item the base scraper uses). Both legs route
# through it. YouTube needs residential IPs (super=true); audio uses cheaper datacenter IPs.
SCRAPEDO_OP_ITEM = "scrape.do"
SCRAPEDO_OP_VAULT = "local"
SCRAPEDO_PROXY_HOST = "proxy.scrape.do:8080"        # proxy mode (YouTube leg)
SCRAPEDO_PROXY_PARAMS_YT = "render=false&super=true&geoCode=us"
SCRAPEDO_API_BASE = "https://api.scrape.do/"        # API mode (audio leg — follows redirects)

# ── Podcast audio transcription (mlx-whisper, Apple-Silicon Metal) ──────────────────
# Resolve the Simplecast player uuid -> public episode JSON -> enclosure_url (no key).
SIMPLECAST_EPISODE_API = "https://api.simplecast.com/episodes/{uuid}"
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
AUDIO_TIMEOUT = 120.0
