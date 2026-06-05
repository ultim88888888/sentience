"""Fetch YouTube captions for routed posts.

YouTube IP-blocks this machine after ~40 direct caption pulls, so every request is routed
through scrape.do's residential rotating proxy (see config). Sequential with a small pause
+ bounded backoff. Missing/disabled captions are logged and skipped, never fatal.
"""
import time

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from .config import YT_BACKOFF_S, YT_LANGUAGES, YT_PAUSE_S, YT_RETRIES
from .proxy import proxied_session


def _join(snippets) -> str:
    """Concatenate caption snippets into clean prose."""
    parts = [s.text.strip() for s in snippets if getattr(s, "text", "").strip()]
    return " ".join(parts)


def fetch_one(api: YouTubeTranscriptApi, video_id: str) -> dict:
    """Return {text, lang, status, error}. status: 'ok' | 'no_captions' | 'unavailable' | 'error'."""
    last_err = None
    for attempt in range(1, YT_RETRIES + 1):
        try:
            tl = api.list(video_id)
            # Prefer human/auto English; fall back to whatever single transcript exists.
            try:
                tr = tl.find_transcript(YT_LANGUAGES)
            except NoTranscriptFound:
                tr = next(iter(tl), None)
                if tr is None:
                    return {"text": "", "lang": None, "status": "no_captions", "error": None}
            data = tr.fetch()
            text = _join(data)
            return {"text": text, "lang": tr.language_code,
                    "status": "ok" if text else "no_captions", "error": None}
        except (TranscriptsDisabled, NoTranscriptFound):
            return {"text": "", "lang": None, "status": "no_captions", "error": None}
        except VideoUnavailable as e:
            return {"text": "", "lang": None, "status": "unavailable", "error": repr(e)}
        except RequestBlocked as e:
            # A proxy IP got blocked; the next retry rotates to a fresh residential IP.
            last_err = "RequestBlocked"
            if attempt < YT_RETRIES:
                time.sleep(YT_BACKOFF_S * attempt)
                continue
        except CouldNotRetrieveTranscript as e:
            # Often transient — back off and retry.
            last_err = repr(e)
            if attempt < YT_RETRIES:
                time.sleep(YT_BACKOFF_S * attempt)
                continue
        except Exception as e:  # network blips etc.
            last_err = repr(e)
            if attempt < YT_RETRIES:
                time.sleep(YT_BACKOFF_S * attempt)
                continue
    return {"text": "", "lang": None, "status": "error", "error": last_err}


def fetch_many(video_ids, log=print) -> dict[str, dict]:
    """Fetch captions for an iterable of video ids, sequentially, via scrape.do residential
    proxy. Returns id -> result dict."""
    api = YouTubeTranscriptApi(http_client=proxied_session(residential=True))
    out: dict[str, dict] = {}
    ids = list(video_ids)
    for i, vid in enumerate(ids, 1):
        res = fetch_one(api, vid)
        out[vid] = res
        log(f"  [yt {i}/{len(ids)}] {vid} {res['status']} "
            f"({len(res['text'])} chars, lang={res['lang']})")
        time.sleep(YT_PAUSE_S)
    return out
