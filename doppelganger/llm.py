"""doppelganger.llm — the shared `claude -p` subprocess wrapper (Max sub, no API cost).

Instructions go via --system-prompt; the large payload via stdin. Runs from an
isolated dir so the call does NOT inherit this repo's CLAUDE.md.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import hashlib
from pathlib import Path

CLAUDE_MODEL = "opus"
CLAUDE_EFFORT = "high"  # was "max"; dialed back 2026-06-07 to conserve credits
CLAUDE_TIMEOUT_S = 900   # 15 min; a single pass over a big corpus can be slow
CLAUDE_RETRIES = 4       # transient rate/usage windows: retry with backoff, never silently drop work
_BACKOFF_BASE_S = 8.0    # 8, ~20, ~50, ~125s (+jitter) — rides out a rate-limit window


def _jitter(slug: str, attempt: int) -> float:
    """Deterministic per-call jitter (no Math.random in this env) so concurrent retries desync."""
    h = int(hashlib.sha1(f"{slug}-{attempt}".encode()).hexdigest()[:4], 16) / 0xFFFF
    return _BACKOFF_BASE_S * (2.5 ** attempt) * (0.7 + 0.6 * h)


def run_claude(system: str, user: str, *, workdir: Path | None = None,
               timeout: int = CLAUDE_TIMEOUT_S, effort: str = CLAUDE_EFFORT,
               model: str = CLAUDE_MODEL, retries: int = CLAUDE_RETRIES) -> str:
    """Run `claude -p`; return stdout (stripped). Retries transient failures with exponential backoff
    (the overnight rate-limit window produced exit-1/empty-output that was silently skipped — never
    again). Raises RuntimeError only after exhausting retries.

    `effort` defaults to high; mechanical preprocessing can pass low/medium to go faster.
    `model` defaults to opus; pass 'sonnet' for aggregation/summary work."""
    wd = workdir or Path(tempfile.mkdtemp(prefix="doppelganger-"))
    Path(wd).mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                ["claude", "-p", "--model", model, "--effort", effort,
                 "--system-prompt", system, "--no-session-persistence"],
                input=user, cwd=str(wd), capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last = "timeout"
            proc = None
        if proc is not None and proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        if proc is not None:
            last = f"exit {proc.returncode}; stderr: {proc.stderr[:300]!r}; empty={not proc.stdout.strip()}"
        if attempt < retries - 1:
            time.sleep(_jitter(system[:24], attempt))   # backoff, then retry
    raise RuntimeError(f"claude -p failed after {retries} attempts. last: {last}")


def run_claude_pool(calls: list, *, max_workers: int = 2) -> list:
    """Run independent run_claude tasks with bounded concurrency (2-3 keeps us under the rate window).
    `calls`: list of zero-arg callables (each returns whatever run_claude returns, or raises).
    Returns results in order; a task that raises after retries yields the exception object (caller
    filters) so one bad call never sinks the batch."""
    from concurrent.futures import ThreadPoolExecutor
    results: list = [None] * len(calls)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(c): i for i, c in enumerate(calls)}
        for fut in futs:
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:   # noqa: BLE001 — surface, don't sink the batch
                results[i] = e
    return results
