# a16z transcript collector

Enriches the a16z research corpus with **full spoken transcripts** for the formats that are
body-shallow in the base scrape — the 139 videos (abstracts only, no transcripts) and the
podcasts (mostly short show-notes). This closes the content gap noted in the study design
spec §8, which a v2 LLM-expert encoder would otherwise be reasoning around blind.

## Routing

Each video/podcast post is routed to a transcript source by parsing its stored `raw_html`
(no re-scrape needed):

| Post kind | Handle in page | Source |
|---|---|---|
| Video | YouTube embed id (all 139 have one) | YouTube captions |
| Podcast w/ YouTube embed (3) | YouTube id | YouTube captions |
| Podcast w/o YouTube (7) | Simplecast player uuid | Simplecast audio → local whisper |

Both legs fetch through **scrape.do** (token in 1Password, same item the base scraper uses):

- **YouTube leg** (`youtube.py`) — `youtube-transcript-api` driven through scrape.do's
  **residential rotating proxy** (`super=true`). YouTube hard-blocks this machine's raw IP
  after ~40 direct caption pulls (`IpBlocked`) *and* blocks datacenter proxies — residential
  is the only path that holds. Sequential with a small pause + backoff (IPs rotate, so
  there's no per-IP cap to respect). Prefers English, falls back to any track. Missing/
  disabled captions are logged and skipped, never fatal.
- **Audio leg** (`audio.py`) — Simplecast exposes a public episode JSON carrying the mp3
  `enclosure_url`. Both the JSON lookup and the mp3 download go through scrape.do's **API
  mode** (which follows redirects server-side — proxy mode loops on Simplecast's 301). The
  download is cached, then transcribed on-device with **mlx-whisper** (`whisper-large-v3`,
  Metal-accelerated — fast on Apple Silicon, free, no API key). Requires **ffmpeg** on PATH.

Re-runs **resume**: rows already `status == ok` in the output parquet are kept as-is, so a
re-run only re-fetches what failed (no re-downloading mp3s or re-transcribing). `--no-resume`
forces a full refetch.

## Run

```bash
. .venv/bin/activate
pip install -r scrapers/a16z_transcripts/requirements.txt
brew install ffmpeg                                  # one-time, for the audio leg
python -m scrapers.a16z_transcripts.run              # full run
python -m scrapers.a16z_transcripts.run --youtube-only   # skip whisper leg
python -m scrapers.a16z_transcripts.run --limit 1    # smoke (first N per source)
```

The first run downloads the whisper-large-v3 weights (~3 GB) from Hugging Face.

## Output — `data/a16z_research/transcripts.parquet`

Join table, one row per video/podcast post, keyed on `object_id` (the corpus parquet is
**never mutated** — merge downstream as needed):

`object_id, title, format, source, media_id, transcript, transcript_len, lang, status,
error, fetched_at`

`status`: `ok` | `no_captions` | `unavailable` | `no_audio` | `error` | `unroutable`.
Every non-`ok` row is printed at the end of the run — no silent gaps. Downloaded mp3s are
cached under `data/a16z_research/audio_cache/` (gitignored, re-fetchable).
