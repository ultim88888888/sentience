"""Transcribe podcasts that have no YouTube embed, via Simplecast audio + local whisper.

Simplecast exposes a public episode JSON (no key) carrying ``enclosure_url`` — the mp3.
Both the JSON lookup and the mp3 download go through scrape.do (consistent with the rest of
the module). We cache the download, then transcribe on-device with mlx-whisper (Metal).
ffmpeg must be on PATH (mlx-whisper shells out to it to decode the mp3).
"""
import time

import requests

from .config import AUDIO_TIMEOUT, AUDIO_CACHE, SIMPLECAST_EPISODE_API, WHISPER_MODEL
from .proxy import api_url, scrapedo_token


def resolve_audio_url(uuid: str, token: str | None = None) -> str | None:
    """Simplecast episode uuid -> direct mp3 url, via scrape.do API mode. None if unresolved."""
    token = token or scrapedo_token()
    r = requests.get(api_url(SIMPLECAST_EPISODE_API.format(uuid=uuid), token), timeout=AUDIO_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("enclosure_url") or data.get("audio_file_url")


def download(url: str, dest, token: str | None = None) -> None:
    """Stream an mp3 to dest via scrape.do API mode (skip if already cached and non-empty)."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    token = token or scrapedo_token()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(api_url(url, token), timeout=AUDIO_TIMEOUT, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    tmp.rename(dest)


def transcribe_file(path) -> dict:
    """Local mlx-whisper transcription. Returns {text, lang, status, error}."""
    import mlx_whisper  # imported lazily so the YouTube-only path needs no mlx/ffmpeg
    try:
        res = mlx_whisper.transcribe(str(path), path_or_hf_repo=WHISPER_MODEL)
        text = (res.get("text") or "").strip()
        return {"text": text, "lang": res.get("language"),
                "status": "ok" if text else "empty", "error": None}
    except Exception as e:
        return {"text": "", "lang": None, "status": "error", "error": repr(e)}


def transcribe_podcasts(routes, log=print) -> dict[str, dict]:
    """For WHISPER-routed podcasts: resolve -> download -> transcribe. object_id -> result."""
    out: dict[str, dict] = {}
    items = list(routes)
    token = scrapedo_token() if items else None
    for i, r in enumerate(items, 1):
        t0 = time.time()
        try:
            url = resolve_audio_url(r.media_id, token=token)
            if not url:
                out[r.object_id] = {"text": "", "lang": None, "status": "no_audio",
                                    "error": "no enclosure_url"}
                log(f"  [audio {i}/{len(items)}] {r.title[:40]!r} no_audio")
                continue
            dest = AUDIO_CACHE / f"{r.media_id}.mp3"
            download(url, dest, token=token)
            res = transcribe_file(dest)
            out[r.object_id] = res
            log(f"  [audio {i}/{len(items)}] {r.title[:40]!r} {res['status']} "
                f"({len(res['text'])} chars, {time.time()-t0:.0f}s)")
        except Exception as e:
            out[r.object_id] = {"text": "", "lang": None, "status": "error", "error": repr(e)}
            log(f"  [audio {i}/{len(items)}] {r.title[:40]!r} error: {e!r}")
    return out
