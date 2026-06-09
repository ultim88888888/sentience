"""A3 — market-aware member deliberation. Each member doppelganger REASONS about the period using
(a) their established framework/known views (their A2 corpus view) and (b) a lookahead-safe period digest
(market performance + news, all <= T), producing fresh calls with mandatory REASONS and a risk appetite
BY HORIZON (short/medium/long — a VC is perma-bullish long-term, but short-term may flip; that's the
hedge signal we want).

Outputs per-member A3 views to <out_root>/<slug>/periods/<date>.json in the SAME PeriodSignal schema as
A2 members, so A3a (consensus) and A3b (council) reuse run_a2a_consensus / run_a2b unchanged.

Lookahead controls: (1) digest is constructed <= T (see signals/digest.py); (2) a Sonnet reason-audit
drops any call whose stated reason references post-T information."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import pandas as pd

from doppelganger.llm import run_claude
from signals.schema import PeriodSignal, RiskRegime
from signals.extract import parse_extraction
from signals.consensus import load_member_views
from signals.digest import build_digest, digest_text


def _render_framework(view: PeriodSignal) -> str:
    """Render a member's A2 corpus view as their 'established framework / known stances'."""
    if view is None or not view.items:
        return "(no prior corpus stance on record — reason purely from your known framework and the digest)"
    lines = []
    for it in view.items:
        lines.append(f"- [{it.item_type}] {it.item}: {it.stance} (conv {it.conviction}) — {it.rationale[:120]}")
    rr = view.risk_regime
    lines.append(f"- overall corpus risk posture: {rr.stance} (conv {rr.conviction})")
    return "\n".join(lines)


_A3_SYS = """You ARE {name}, a member of the a16z crypto team. It is {t}. The future has not happened —
every figure and event below is dated on or before {t}; never use knowledge of anything after {t}.

You are not summarizing your past statements. You are MAKING CALLS for the period ahead by applying YOUR
framework and worldview (below) to what has just happened this period (the digest below). Think as {name}
would: weigh the market action and events through your known priorities, and commit to views.

YOUR ESTABLISHED FRAMEWORK / KNOWN STANCES (from your corpus, as of {t}):
{framework}

Produce YOUR calls. Every excited/concerned entry needs: name (the sector or token itself — never your own
name), why (your reasoning, grounded in your framework + the digest — this is audited), conviction (0-100),
horizon ("tactical"|"structural"), parent_sector (for tokens).

Also give your RISK APPETITE BY HORIZON — your long-term VC view is likely constructive, but be honest
about SHORT-TERM risk; if the period's action/events warrant near-term caution, say so:
  short (<=1-3 months), medium (3-12 months), long (>12 months) — each {{"stance":"risk_on|risk_off|neutral","why":"..."}}

Output JSON only:
{{"sectors_excited":[...],"sectors_concerned":[...],"tokens_excited":[...],"tokens_concerned":[...],
  "risk_by_horizon":{{"short":{{"stance":"...","why":"..."}},"medium":{{...}},"long":{{...}}}},
  "notes":"..."}}"""


def member_call(t: date, name: str, framework_view: PeriodSignal, digest: dict,
                *, model: str = "opus") -> PeriodSignal:
    """One market-aware reasoned call for `name` at T. Opus (this IS reasoning, not summary)."""
    system = _A3_SYS.format(name=name, t=t.isoformat(), framework=_render_framework(framework_view))
    user = digest_text(digest)
    raw = run_claude(system, user, model=model)
    p = parse_extraction(raw, t=t)              # items (with reasons in .rationale) parsed here
    # capture risk-by-horizon; short-term stance drives the actionable risk_regime
    rbh = _extract_rbh(raw)
    short = rbh.get("short", {}) if isinstance(rbh, dict) else {}
    risk = RiskRegime(stance=short.get("stance", "neutral"), conviction=50,
                      rationale=short.get("why", ""), provenance="extrapolated")
    notes = json.dumps({"risk_by_horizon": rbh, "notes": p.notes})
    return PeriodSignal(as_of=p.as_of, approach=f"A3a:{name}", items=p.items,
                        risk_regime=risk, notes=notes)


def _extract_rbh(raw: str) -> dict:
    from signals.extract import _extract_json
    return _extract_json(raw).get("risk_by_horizon", {}) or {}


_AUDIT_SYS = """You are a strict lookahead auditor for a point-in-time trading study. It is {t}.
Below is a JSON list of calls, each with an index and a 'reason'. Flag the index of ANY reason that relies
on information that could only be known AFTER {t} — a future price move, a later outcome, an event's
consequence, a result not yet known at {t}. Hindsight phrasing ("this later", "would go on to", "turned out")
is a flag. Reasoning from the period's own data/events is FINE. Be precise; do not over-flag legitimate
as-of-{t} reasoning. Output JSON only: {{"lookahead_indices":[...]}}."""


def audit_reasons(view: PeriodSignal, t: date, *, model: str = "sonnet") -> PeriodSignal:
    """Sonnet reason-audit: drop items whose reason references post-T info. Returns a filtered view."""
    if not view.items:
        return view
    payload = json.dumps([{"i": i, "item": it.item, "reason": it.rationale}
                          for i, it in enumerate(view.items)])
    try:
        raw = run_claude(_AUDIT_SYS.format(t=t.isoformat()), payload, model=model, effort="low", timeout=240)
        from signals.extract import _extract_json
        flagged = set(_extract_json(raw).get("lookahead_indices", []) or [])
    except Exception:
        flagged = set()
    kept = tuple(it for i, it in enumerate(view.items) if i not in flagged)
    return PeriodSignal(as_of=view.as_of, approach=view.approach, items=kept,
                        risk_regime=view.risk_regime, notes=view.notes)


def run_a3_members(rebalance_dates, *, interval_days: int, a2_members_root: Path, out_root: Path,
                   sector_map: dict, prices: pd.DataFrame, oi_panel: pd.DataFrame,
                   members: list[tuple[str, str]] | None = None, audit: bool = True,
                   news: bool = True, model: str = "opus") -> None:
    """For each date: build the digest once, then each member makes a reasoned (audited) call.
    Writes <out_root>/<slug>/periods/<date>.json (A2-member schema). Resumable (skips existing).
    `members`: list of (slug, display_name); if None, inferred from a2_members_root dirs."""
    out_root = Path(out_root)
    if members is None:
        import glob
        members = [(Path(d).name, Path(d).name.replace("-", " ").title())
                   for d in sorted(glob.glob(str(Path(a2_members_root) / "*")))]
    for t in rebalance_dates:
        # skip if all members already done for this date
        if all((out_root / slug / "periods" / f"{t.isoformat()}.json").exists() for slug, _ in members):
            continue
        digest = build_digest(t, interval_days, sector_map, prices, oi_panel, news=news)
        a2_views = load_member_views(t, a2_members_root)   # {slug: PeriodSignal}
        for slug, disp in members:
            pj = out_root / slug / "periods" / f"{t.isoformat()}.json"
            if pj.exists():
                continue
            try:
                v = member_call(t, disp, a2_views.get(slug), digest, model=model)
                if audit:
                    v = audit_reasons(v, t)
                pj.parent.mkdir(parents=True, exist_ok=True)
                pj.write_text(json.dumps(v.to_dict(), indent=2))
            except Exception as e:
                print(f"[a3 skip] {t} {slug}: {e}", flush=True)
        print(f"[a3] {t.isoformat()} done ({len(members)} members)", flush=True)
