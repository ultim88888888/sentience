"""doppelganger.llm — the shared `claude -p` subprocess wrapper (Max sub, no API cost).

Instructions go via --system-prompt; the large payload via stdin. Runs from an
isolated dir so the call does NOT inherit this repo's CLAUDE.md.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

CLAUDE_MODEL = "opus"
CLAUDE_EFFORT = "high"  # was "max"; dialed back 2026-06-07 to conserve credits
CLAUDE_TIMEOUT_S = 900   # 15 min; a single pass over a big corpus can be slow


def run_claude(system: str, user: str, *, workdir: Path | None = None,
               timeout: int = CLAUDE_TIMEOUT_S, effort: str = CLAUDE_EFFORT) -> str:
    """Run `claude -p`; return stdout (stripped). Raise RuntimeError on non-zero exit.

    `effort` defaults to high (the reasoning/extraction surface); mechanical preprocessing
    like content-filtering can pass a lower level (low/medium) to go much faster."""
    wd = workdir or Path(tempfile.mkdtemp(prefix="doppelganger-"))
    Path(wd).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["claude", "-p", "--model", CLAUDE_MODEL, "--effort", effort,
         "--system-prompt", system, "--no-session-persistence"],
        input=user, cwd=str(wd), capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exited {proc.returncode}. stderr:\n{proc.stderr}")
    return proc.stdout.strip()
