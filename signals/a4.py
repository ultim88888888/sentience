"""A4 — TRUE doppelganger market-aware deliberation (the non-shortcut version of A3).

A3 fed each member their A2 *extracted view* (a stance list) — their conclusions, not their mind.
A4 uses the actual doppelganger: the SOUL card (how they think — reasoning moves, epistemic style, built
from evidence <= T) + the time-gated MEMORY feed (their own words <= T) + the period DIGEST → the member
reasons in character and makes calls. Soul and memory both apply a hard <= T firewall (no lookahead).

Soul cadence: annual anchors (the soul is stable "how they think"; the per-period <= T memory + digest
carry the fresh facts). For period T we use the soul built at the most recent year-end <= T. 4 souls/member
instead of 14, lookahead-safe. Outputs A2-member schema → A4a (consensus) / A4b (council) reuse the runners.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import pandas as pd

from doppelganger.llm import run_claude, run_claude_pool
from doppelganger.soul import extract_soul, extract_soul_chunked
from doppelganger.memory import load_memory
from signals.schema import PeriodSignal, RiskRegime
from signals.extract import parse_extraction
from signals.digest import build_digest, digest_text
from signals.a3 import audit_reasons, _extract_rbh

A4_SOULS_DIR = Path("data/doppelganger/a4_souls")
_ANCHORS = [date(2022, 12, 31), date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)]


def soul_anchor(t: date) -> date:
    """Most recent annual anchor <= t (so the soul never sees the future)."""
    cands = [a for a in _ANCHORS if a <= t]
    return cands[-1] if cands else _ANCHORS[0]


def ensure_soul(slug: str, anchor: date, *, evidence_path: Path | None = None) -> str:
    """Return the soul-card text for (slug, anchor), building + caching it if absent.
    Reuses a prebuilt canonical soul (data/doppelganger/<slug>/soul.md) when its t0 matches the anchor."""
    cache = A4_SOULS_DIR / anchor.isoformat() / slug / "soul.md"
    if cache.exists():
        return cache.read_text()
    canon = Path("data/doppelganger") / slug / "soul.md"
    if canon.exists() and f"t0: {anchor.isoformat()}" in canon.read_text()[:400]:
        return canon.read_text()        # reuse the existing doppelganger soul (ali/eddy at 2022-12-31)
    # Use the FULL corpus (no sampling — the soul needs all the member's words). For prolific subjects
    # whose corpus exceeds the AUP size ceiling, chunk-and-merge processes every tweet in safe pieces.
    # chunk_items=2500 (~222k tok/chunk) is AUP-safe and proven (scott's 7770-item=317k soul built;
    # full 500k corpus is AUP-rejected). Each chunk clears the ceiling; the merge synthesizes them.
    path = extract_soul_chunked(slug, anchor, out_dir=A4_SOULS_DIR / anchor.isoformat(),
                                evidence_path=evidence_path, chunk_items=2500)
    return Path(path).read_text()


_A4_INSTRUCTIONS = """

================  YOUR TASK (it is {t})  ================
The text above is WHO YOU ARE — your soul: how you think, what you value, how you update. Below is the
record of what you have said and seen, and a market digest — ALL dated on or before {t}. The future has
not happened; never use knowledge of anything after {t}.

Reason as YOU would — apply your framework to this period's market action and events — and MAKE YOUR CALLS.
Not a summary of your past views: fresh calls for the period ahead, in your voice.

Every excited/concerned entry: name (the sector or token itself, never your own name), why (ONE concise
sentence <=25 words, grounded in your framework + the evidence/digest — audited), conviction (0-100),
horizon ("tactical"|"structural"), parent_sector (for tokens).

Risk appetite BY HORIZON (your long-term VC view is likely constructive; be honest about SHORT-TERM risk),
each 'why' <=20 words: short (<=1-3mo), medium (3-12mo), long (>12mo) — {{"stance":"risk_on|risk_off|neutral","why":"..."}}.

Keep it tight. Output JSON only:
{{"sectors_excited":[...],"sectors_concerned":[...],"tokens_excited":[...],"tokens_concerned":[...],
  "risk_by_horizon":{{"short":{{"stance":"...","why":"..."}},"medium":{{...}},"long":{{...}}}}}}"""


# Soul carries the long-term framework (full evidence <= anchor), so the memory feed only needs the RECENT
# <= T window. Cap keeps soul+memory+digest under the AUP-safe ceiling (~50k tok): full memory for a
# prolific member is ~90k tok and would be rejected. Tail = most recent (evidence is sorted ascending).
_MAX_MEM_CHARS = 140_000


def member_call_soul(t: date, slug: str, name: str, digest: dict, *,
                     evidence_path: Path | None = None, model: str = "opus") -> PeriodSignal:
    """Soul (how they think, <=anchor) + recent time-gated memory (<=T window) + digest → reasoned calls."""
    soul = ensure_soul(slug, soul_anchor(t), evidence_path=evidence_path)
    mem = load_memory(slug, t, evidence_path=evidence_path).text
    if len(mem) > _MAX_MEM_CHARS:
        mem = "...[earlier history summarized in your soul above]...\n" + mem[-_MAX_MEM_CHARS:]
    system = soul + _A4_INSTRUCTIONS.format(t=t.isoformat())
    user = f"--- YOUR RECORD AND MEMORY (<= {t}) ---\n{mem}\n\n{digest_text(digest)}"
    raw = run_claude(system, user, model=model)
    p = parse_extraction(raw, t=t)
    if not p.items:
        raw = run_claude(system, user, model=model)
        p = parse_extraction(raw, t=t)
    rbh = _extract_rbh(raw)
    short = rbh.get("short", {}) if isinstance(rbh, dict) else {}
    risk = RiskRegime(stance=short.get("stance", "neutral"), conviction=50,
                      rationale=short.get("why", ""), provenance="extrapolated")
    return PeriodSignal(as_of=p.as_of, approach=f"A4a:{name}", items=p.items,
                        risk_regime=risk, notes=json.dumps({"risk_by_horizon": rbh}))


def run_a4_members(rebalance_dates, *, interval_days: int, out_root: Path, sector_map: dict,
                   prices: pd.DataFrame, oi_panel: pd.DataFrame, members: list[tuple[str, str]],
                   evidence_root: Path = Path("data/doppelganger"), audit: bool = True,
                   news: bool = True, max_workers: int = 2, model: str = "opus") -> None:
    """Per date: build digest once, then members make soul-grounded calls with bounded concurrency
    (retry+backoff in run_claude rides out rate windows). Writes A2-member schema. Resumable."""
    out_root = Path(out_root)
    for t in rebalance_dates:
        if all((out_root / slug / "periods" / f"{t.isoformat()}.json").exists() for slug, _ in members):
            continue
        digest = build_digest(t, interval_days, sector_map, prices, oi_panel, news=news)
        todo = [(slug, name) for slug, name in members
                if not (out_root / slug / "periods" / f"{t.isoformat()}.json").exists()]

        def _one(slug, name):
            ev = evidence_root / slug / "evidence.parquet"
            v = member_call_soul(t, slug, name, digest, evidence_path=ev, model=model)
            return audit_reasons(v, t) if audit else v

        results = run_claude_pool([(lambda s=s, n=n: _one(s, n)) for s, n in todo],
                                  max_workers=max_workers)
        for (slug, _name), v in zip(todo, results):
            if isinstance(v, Exception) or v is None:
                print(f"[a4 skip] {t} {slug}: {v}", flush=True); continue
            pj = out_root / slug / "periods" / f"{t.isoformat()}.json"
            pj.parent.mkdir(parents=True, exist_ok=True)
            pj.write_text(json.dumps(v.to_dict(), indent=2))
        print(f"[a4] {t.isoformat()} done ({len(todo)} attempted)", flush=True)
