"""Shared scrape.do proxy session for both transcript legs.

Every external fetch in this module goes through scrape.do (consistent with the base
a16z_research scraper). scrape.do MITMs TLS, so verify is off.

- YouTube captions need *residential* IPs (`super=true`): YouTube blocks both this machine's
  raw IP and datacenter proxies.
- Podcast audio (Simplecast JSON + mp3) uses plain datacenter proxies: the CDN doesn't IP-
  block, and residential bandwidth is metered/expensive for tens-of-MB mp3s.
"""
import subprocess
import urllib.parse

import requests
import urllib3

from .config import (SCRAPEDO_API_BASE, SCRAPEDO_OP_ITEM, SCRAPEDO_OP_VAULT,
                     SCRAPEDO_PROXY_HOST, SCRAPEDO_PROXY_PARAMS_YT)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def scrapedo_token() -> str:
    return subprocess.check_output(
        ["op", "item", "get", SCRAPEDO_OP_ITEM, "--vault", SCRAPEDO_OP_VAULT,
         "--fields", "credential", "--reveal"], text=True).strip()


def proxied_session(residential: bool) -> requests.Session:
    """requests.Session through scrape.do *proxy* mode (TLS verify off — scrape.do MITMs).

    Used by the YouTube leg: youtube-transcript-api needs to drive the requests itself, so a
    proxied Session is the only injection point. residential=True (super) dodges YouTube's
    IP ban. (Proxy mode doesn't auto-follow redirects, so the audio leg uses API mode below.)
    """
    token = scrapedo_token()
    params = SCRAPEDO_PROXY_PARAMS_YT  # only the residential YouTube path uses proxy mode
    proxy = f"http://{token}:{params}@{SCRAPEDO_PROXY_HOST}"
    sess = requests.Session()
    sess.verify = False
    sess.proxies = {"http": proxy, "https": proxy}
    return sess


def api_url(target: str, token: str | None = None) -> str:
    """scrape.do *API* mode URL for ``target``. Follows redirects server-side and returns the
    final body — the right mode for the audio leg (JSON lookup + binary mp3)."""
    token = token or scrapedo_token()
    q = urllib.parse.quote(target, safe="")
    return f"{SCRAPEDO_API_BASE}?token={token}&url={q}"
