"""A2a consensus: aggregate per-member views into a house consensus per period.
Deterministic dispersion (cross-member disagreement) + LLM consensus (dispersion in hand).
Keep all items — consensus weighting drowns out singular coverage; expose coverage/dispersion."""
from __future__ import annotations
import json, glob, os, statistics
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger.llm import run_claude
from signals import config
from signals.config import STANCE_SIGN
from signals.schema import PeriodSignal
from signals.extract import parse_extraction
from signals.canonicalize import canonicalize_items
from signals.registry import load_registry, save_registry
from signals.panel import derive_panel


def load_member_views(t: date, members_root: Path) -> dict:
    """slug -> PeriodSignal for members that have a view at date t."""
    out = {}
    for d in sorted(glob.glob(str(Path(members_root) / "*"))):
        pj = Path(d) / "periods" / f"{t.isoformat()}.json"
        if pj.exists():
            out[os.path.basename(d)] = PeriodSignal.from_dict(json.load(open(pj)))
    return out


def compute_dispersion(member_views: dict, n_members: int) -> dict:
    """Per canonical item across members: coverage + stance/conviction spread.
    Deterministic — this is a measurement, never an LLM. Returns {item: {...}}."""
    table = {}  # item -> list of (stance, conviction, item_type, parent_sector)
    for pv in member_views.values():
        for it in pv.items:
            table.setdefault(it.item, []).append((it.stance, it.conviction, it.item_type, it.parent_sector))
    out = {}
    for item, entries in table.items():
        signs = [STANCE_SIGN[s] for s, _, _, _ in entries]
        convs = [c for _, c, _, _ in entries]
        out[item] = {
            "coverage": len(entries),
            "coverage_frac": len(entries) / n_members if n_members else 0.0,
            "stance_dispersion": statistics.pstdev(signs) if len(signs) > 1 else 0.0,
            "conviction_spread": statistics.pstdev(convs) if len(convs) > 1 else 0.0,
            "mean_stance_sign": statistics.mean(signs),
            "mean_conviction": statistics.mean(convs),
            "item_type": entries[0][2],
            "parent_sector": entries[0][3],
        }
    return out


_CONSENSUS_SYS = """You are the a16z crypto research desk forming a single HOUSE CONSENSUS market view as of {t}.
You are given {n} team members' individual market views (JSON) plus a DISPERSION table showing, per item,
how many members hold a view (coverage) and how much they disagree (stance_dispersion, conviction_spread).

Form the consensus. Weight by BOTH conviction AND coverage — a lone low-conviction view should fade out; a
view many members hold strongly should dominate. Where members are SPLIT (high stance_dispersion), reflect
that as lower consensus conviction (or neutral) and say so in the rationale. Do NOT invent items no member
mentioned. Reuse the members' exact item names where possible.

Output the SAME JSON schema as the member views:
{{"sectors_excited":[...], "sectors_concerned":[...], "tokens_excited":[...], "tokens_concerned":[...],
  "risk_regime":{{"stance":"risk_on|risk_off|neutral|no_view","conviction":0-100,"why":"...","provenance":"..."}},
  "notes":"..."}}"""


def build_consensus(t: date, member_views: dict, dispersion: dict) -> PeriodSignal:
    """LLM consensus over the structured member views (dispersion provided). Small input → reliable."""
    payload = json.dumps({
        "members": {slug: pv.to_dict() for slug, pv in member_views.items()},
        "dispersion": dispersion,
    }, indent=2)
    system = _CONSENSUS_SYS.format(t=t.isoformat(), n=len(member_views))
    return parse_extraction(run_claude(system, payload), t=t)  # payload via stdin (non-empty)


def run_a2a_consensus(rebalance_dates: list, *, members_root: Path | None = None,
                      out_dir: Path | None = None, registry_path: Path | None = None) -> pd.DataFrame:
    """For each date: load member views, dispersion, LLM consensus, canonicalize (shared registry),
    write consensus period + dispersion rows. Then derive the A2a consensus panel. Resumable
    (skip a date whose consensus period json exists). Skip-continue on per-date failure."""
    members_root = Path(members_root or (config.SIGNAL_OUT_DIR / "members"))
    out_dir = Path(out_dir or (config.SIGNAL_OUT_DIR / "a2a_consensus"))
    (out_dir / "periods").mkdir(parents=True, exist_ok=True)
    registry_path = registry_path or (config.SIGNAL_OUT_DIR / "registry.json")
    registry = load_registry(registry_path)
    n_members = len(glob.glob(str(members_root / "*")))
    periods, disp_rows = [], []
    for t in rebalance_dates:
        pj = out_dir / "periods" / f"{t.isoformat()}.json"
        if pj.exists():
            periods.append(PeriodSignal.from_dict(json.load(open(pj)))); continue
        try:
            mv = load_member_views(t, members_root)
            if not mv:
                continue
            disp = compute_dispersion(mv, n_members)
            raw = build_consensus(t, mv, disp)
            canon_items, registry = canonicalize_items(list(raw.items), registry)
            period = PeriodSignal(as_of=raw.as_of, approach="A2a", items=tuple(canon_items),
                                  risk_regime=raw.risk_regime, notes=raw.notes)
            pj.write_text(json.dumps(period.to_dict(), indent=2))
            for item, d in disp.items():
                disp_rows.append({"as_of": t.isoformat(), "item": item, **d})
            periods.append(period)
        except Exception as e:
            print(f"[consensus skip] {t}: {e}")
            continue
    save_registry(registry, registry_path)
    df = derive_panel(periods)
    df.to_parquet(out_dir / "signal_panel.parquet")
    pd.DataFrame(disp_rows).to_parquet(out_dir / "dispersion.parquet")
    return df
