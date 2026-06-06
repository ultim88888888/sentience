"""doppelganger.soul — build the frozen-at-T0 soul card via a single claude -p pass.

The LLM call is isolated in `_run_claude` so the rest is deterministically testable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import subprocess
import tempfile

import pandas as pd

from doppelganger import config
from doppelganger.identity import build_identity
from doppelganger.ingest import build_evidence_stream
from doppelganger.schema import IdentityProfile


def load_soul_inputs(
    slug: str, t0: date, *,
    evidence_path: Path | None = None,
    identity_path: Path | None = None,
    team_path: Path | None = None,
    tracked_people_path: Path | None = None,
    twitter_path: Path | None = None,
    articles_path: Path | None = None,
    podcast_path: Path | None = None,
) -> tuple[IdentityProfile, pd.DataFrame]:
    """Return (identity truncated to <=t0, evidence DataFrame filtered to <=t0, sorted)."""
    identity = build_identity(
        slug, linkedin_path=identity_path, team_path=team_path,
        tracked_people_path=tracked_people_path,
    ).as_of(t0)

    if evidence_path is not None and Path(evidence_path).exists():
        ev = pd.read_parquet(evidence_path)
    else:
        items = build_evidence_stream(
            slug, twitter_path=twitter_path, articles_path=articles_path,
            podcast_path=podcast_path, tracked_people_path=tracked_people_path,
        )
        from dataclasses import asdict
        ev = pd.DataFrame([asdict(e) for e in items])

    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    ev = ev[ev["timestamp"].dt.date <= t0].sort_values("timestamp").reset_index(drop=True)
    return identity, ev


SECTIONS = [
    "Bio Lens", "How He Thinks", "What He Believes",
    "What He Attends To", "Open Contradictions", "How He Talks",
]

_INSTRUCTIONS = f"""You are an expert analyst building a CHARACTERIZATION of a person \
from their own words, to be used as a lens for reconstructing their market views. \
You are given the subject's bio and the complete record of what they said up to a cutoff date.

Write a Markdown "soul card" with EXACTLY these H2 sections, in this order \
(views first; voice last):

1. ## Bio Lens — how their background shapes their analytical lens. Note where the \
corpus CONFIRMS vs OVERRIDES what the bio alone would predict.
2. ## How He Thinks — their reasoning *moves*, epistemic style (how they update, hedge, \
calibrate certainty), and named frameworks / mental models. This is the priority section.
3. ## What He Believes — durable, RECURRING convictions only (stable across the whole span). \
Exclude one-off or volatile positions.
4. ## What He Attends To — what they fixate on vs. dismiss.
5. ## Open Contradictions — genuine tensions in their thinking. Preserve them; never average away.
6. ## How He Talks — brief; lowest priority. Only register/voice that reveals how they think.

RULES:
- Ground EVERY factual claim in the evidence. Immediately after each claim, cite it inline \
in EXACTLY this format: a bracketed ISO date then the verbatim quote in straight double quotes — \
[<YYYY-MM-DD>] "exact quote from the evidence". Keep quotes <= 25 words, verbatim, no ellipsis.
- Use ONLY the provided evidence. Do not use anything you know about this person from outside it. \
Do not reference events after the cutoff.
- Be specific and concrete. Generic statements that could describe any investor are failures.
- Weight solo first-person evidence over co-authored/firm material.

Output ONLY the Markdown soul card (starting with the first ## heading). No preamble."""


def build_extraction_prompt(identity: IdentityProfile, evidence: pd.DataFrame) -> tuple[str, str]:
    """Return (system_instructions, user_content). user_content is piped to claude via stdin."""
    lines = [f"# SUBJECT: {identity.name} ({identity.slug})", ""]
    lines.append("## BIO")
    lines.append(f"Headline: {identity.headline or ''}")
    lines.append(f"Current role (as of cutoff): {identity.current_role or ''}")
    lines.append(f"Bio: {identity.bio or ''}")
    if identity.experience:
        lines.append("Experience: " + "; ".join(
            f"{e.title} @ {e.company}" for e in identity.experience))
    if identity.education:
        lines.append("Education: " + "; ".join(
            f"{e.degree or ''} {e.field or ''} @ {e.school}".strip() for e in identity.education))
    lines += ["", f"## EVIDENCE ({len(evidence)} items, chronological)", ""]
    for _, r in evidence.iterrows():
        d = pd.Timestamp(r["timestamp"]).date().isoformat()
        ctx = r.get("context")
        ctx_s = f" (context: {ctx})" if isinstance(ctx, str) and ctx else ""
        lines.append(f"[{d}] ({r['source_type']}){ctx_s} {r['text']}")
    return _INSTRUCTIONS, "\n".join(lines)


CLAUDE_MODEL = "opus"
CLAUDE_EFFORT = "max"
CLAUDE_TIMEOUT_S = 900   # 15 min; soul extraction over a big corpus can be slow


def _run_claude(system: str, user: str, *, workdir: Path | None = None,
                timeout: int = CLAUDE_TIMEOUT_S) -> str:
    """Run `claude -p` (Max subscription, no API cost) from an isolated dir.

    Instructions go via --system-prompt; the large bio+evidence payload via stdin.
    Returns stdout (stripped). Raises RuntimeError on non-zero exit.
    """
    wd = workdir or Path(tempfile.mkdtemp(prefix="doppelganger-soul-"))
    Path(wd).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["claude", "-p", "--model", CLAUDE_MODEL, "--effort", CLAUDE_EFFORT,
         "--system-prompt", system, "--no-session-persistence"],
        input=user, cwd=str(wd), capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exited {proc.returncode}. stderr:\n{proc.stderr}")
    return proc.stdout.strip()
