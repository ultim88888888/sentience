"""A2b council deliberation: the member doppelgangers DEBATE (not just aggregate) to a consensus +
a reasoned hedge decision. A compelling MINORITY view can persuade the council (propagation), unlike
A2a where conviction×coverage buries it. One deliberation call/period (single-pass debate simulation;
real multi-round multi-agent is a future upgrade). Output: consensus PeriodSignal + per-period hedge."""
from __future__ import annotations
import json, glob, os
from datetime import date
from pathlib import Path
import pandas as pd
from doppelganger.llm import run_claude
from signals import config
from signals.schema import PeriodSignal
from signals.extract import parse_extraction
from signals.canonicalize import canonicalize_items
from signals.registry import load_registry, save_registry
from signals.panel import derive_panel
from signals.consensus import load_member_views   # reuse


_COUNCIL_SYS = """You are facilitating a COUNCIL of {n} a16z crypto team members deliberating as of {t}.
Below (as input) are each member's individual market views for this period. Simulate their DISCUSSION,
do not merely average:
- Members present their views and CHALLENGE each other.
- A COMPELLING MINORITY view — held by even one member but well-argued or differentiated — CAN persuade
  the council and make it into the consensus. Do not bury a strong differentiated call just because few
  hold it; that differentiated insight is often the most valuable.
- Reach a consensus on what the council is most BULLISH and CONCERNED about (sectors and tokens).
- Separately, the council makes a HEDGE DECISION for the period: should they de-risk / hedge market
  exposure this period? Decide "hedge" or "no_hedge" with a reason, based on the members' risk posture
  and concerns — this is the council's deliberated risk call, not a mechanical rule.

Each entry in the four item arrays is the COUNCIL'S consensus call on ONE sector or token. Use exactly
these fields — do NOT add a speaker/author/"name":"council" field:
  "name": "<the sector or token name itself, e.g. \"zk\" or \"ETH\">", "why": "<one line>",
  "conviction": 0-100, "horizon": "tactical|structural", "provenance": "grounded|persisted|extrapolated",
  "parent_sector": "<sector id, for tokens only>", "citations": [].
The `name` field is the sector/token, never a person or "council".

Output JSON only:
{{"sectors_excited":[...], "sectors_concerned":[...], "tokens_excited":[...], "tokens_concerned":[...],
  "risk_regime":{{"stance":"risk_on|risk_off|neutral|no_view","conviction":0-100,"why":"...","provenance":"..."}},
  "hedge_decision":{{"stance":"hedge|no_hedge","reason":"..."}},
  "notes":"..."}}"""


def deliberate(t: date, member_views: dict) -> tuple[PeriodSignal, str]:
    """One council-deliberation call. Returns (consensus PeriodSignal, hedge_stance 'hedge'|'no_hedge')."""
    payload = json.dumps({slug: pv.to_dict() for slug, pv in member_views.items()}, indent=2)
    system = _COUNCIL_SYS.format(t=t.isoformat(), n=len(member_views))
    raw = run_claude(system, payload)
    period = parse_extraction(raw, t=t)
    obj = _hedge_from(raw)
    return period, obj


def _hedge_from(raw: str) -> str:
    from signals.extract import _extract_json
    hd = (_extract_json(raw).get("hedge_decision") or {})
    return "hedge" if isinstance(hd, dict) and hd.get("stance") == "hedge" else "no_hedge"


def run_a2b(rebalance_dates: list, *, members_root: Path | None = None, out_dir: Path | None = None,
            registry_path: Path | None = None) -> pd.DataFrame:
    """Per date: load member views, run council deliberation, canonicalize (shared registry), write
    consensus period + collect hedge decision. Resumable; skip-continue on per-date failure.
    Writes signal_panel.parquet + hedge_decisions.json ({as_of: 'hedge'|'no_hedge'})."""
    members_root = Path(members_root or (config.SIGNAL_OUT_DIR / "members"))
    out_dir = Path(out_dir or (config.SIGNAL_OUT_DIR / "a2b_council"))
    (out_dir / "periods").mkdir(parents=True, exist_ok=True)
    registry_path = registry_path or (config.SIGNAL_OUT_DIR / "registry.json")
    registry = load_registry(registry_path)
    periods, hedges = [], {}
    for t in rebalance_dates:
        pj = out_dir / "periods" / f"{t.isoformat()}.json"
        if pj.exists():
            periods.append(PeriodSignal.from_dict(json.load(open(pj)))); continue
        try:
            mv = load_member_views(t, members_root)
            if not mv:
                continue
            raw_p, hedge = deliberate(t, mv)
            canon, registry = canonicalize_items(list(raw_p.items), registry)
            period = PeriodSignal(as_of=raw_p.as_of, approach="A2b", items=tuple(canon),
                                  risk_regime=raw_p.risk_regime, notes=raw_p.notes)
            pj.write_text(json.dumps(period.to_dict(), indent=2))
            hedges[t.isoformat()] = hedge
            periods.append(period)
        except Exception as e:
            print(f"[council skip] {t}: {e}")
            continue
    save_registry(registry, registry_path)
    df = derive_panel(periods)
    df.to_parquet(out_dir / "signal_panel.parquet")
    (out_dir / "hedge_decisions.json").write_text(json.dumps(hedges, indent=2))
    return df
