"""Mint a short-lived Algolia secured key by solving the AWS WAF challenge in a
headless browser. The key (TTL ~300s) is pinned by a16z's backend; querying Algolia
from this same machine satisfies its restrictSources check."""
import json
import time
from dataclasses import dataclass

from playwright.sync_api import sync_playwright

from .config import RESEARCH_URL, UA

# JS run in the page context. AwsWafIntegration.fetch attaches the solved WAF token;
# fall back to plain fetch (the token is also set as a cookie).
_MINT_JS = """async () => {
    const f = (window.AwsWafIntegration && typeof window.AwsWafIntegration.fetch === 'function')
        ? window.AwsWafIntegration.fetch.bind(window.AwsWafIntegration) : fetch;
    const r = await f('/api/generate-key', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: '{}'});
    return {status: r.status, body: await r.text()};
}"""


@dataclass
class AlgoliaKey:
    key: str
    app_id: str
    index: str
    expires_at: float  # monotonic-ish wall clock (time.time) when the key dies

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - 15  # 15s safety margin


def mint_key(wait_ms: int = 6000, timeout_ms: int = 60000) -> AlgoliaKey:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_context(user_agent=UA).new_page()
            page.goto(RESEARCH_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(wait_ms)  # let challenge.js solve the WAF
            res = page.evaluate(_MINT_JS)
        finally:
            browser.close()
    if res["status"] != 200:
        raise RuntimeError(f"generate-key returned HTTP {res['status']}: {res['body'][:200]}")
    d = json.loads(res["body"])
    if not d.get("success") or not d.get("securedApiKey"):
        raise RuntimeError(f"generate-key payload missing key: {res['body'][:200]}")
    return AlgoliaKey(
        key=d["securedApiKey"], app_id=d["appId"], index=d["indexName"],
        expires_at=time.time() + int(d.get("expiresInSeconds", 300)),
    )


if __name__ == "__main__":
    k = mint_key()
    print(f"appId={k.app_id} index={k.index} expires_in={int(k.expires_at - time.time())}s "
          f"key={k.key[:12]}...{k.key[-6:]}")
