"""doppelganger.walkforward — drive respond() across a quarterly schedule.

Produces matched full + ablation views per step, audits each, and records a
resumable trajectory + coverage map. The raw material Unit 5b scores.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.respond import respond
from doppelganger.soul_audit import audit_answer

_QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def quarter_ends(start: date, end: date) -> list[date]:
    """Quarter-end dates (Mar 31 / Jun 30 / Sep 30 / Dec 31) within [start, end], inclusive."""
    out: list[date] = []
    for yr in range(start.year, end.year + 1):
        for m, d in _QUARTER_ENDS:
            qd = date(yr, m, d)
            if start <= qd <= end:
                out.append(qd)
    return out


_ARRAYS = ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned"]


def _row(d: date, variant: str, view: dict, rep) -> dict:
    prov: Counter = Counter()
    for k in _ARRAYS:
        for it in view.get(k, []) or []:
            prov[it.get("provenance")] += 1
    return {
        "date": d.isoformat(), "variant": variant,
        "abstained": bool(view.get("abstained", False)),
        "risk": (view.get("risk_regime") or {}).get("stance"),
        "n_sectors_excited": len(view.get("sectors_excited", []) or []),
        "n_sectors_concerned": len(view.get("sectors_concerned", []) or []),
        "n_tokens_excited": len(view.get("tokens_excited", []) or []),
        "n_tokens_concerned": len(view.get("tokens_concerned", []) or []),
        "grounded": prov.get("grounded", 0),
        "persisted": prov.get("persisted", 0),
        "extrapolated": prov.get("extrapolated", 0),
        "leaked": len(rep.leaked), "hallucinated": len(rep.hallucinated),
        "matched": rep.matched, "checked": rep.checked,
    }


def run_walkforward(slug: str, dates: list[date], *, ablate: bool = True,
                    out_dir: Path | None = None, evidence_path: Path | None = None,
                    soul_path: Path | None = None) -> list[dict]:
    base_dir = Path(out_dir or config.OUT_DIR)
    ev_path = evidence_path or (base_dir / slug / "evidence.parquet")
    variants = [("full", False, "views")] + ([("ablation", True, "views_ablation")] if ablate else [])

    rows: list[dict] = []
    for d in dates:
        for variant, ablate_mem, subdir in variants:
            vpath = base_dir / slug / subdir / f"{d.isoformat()}.json"
            try:
                if vpath.exists():
                    view = json.loads(vpath.read_text())          # cached — no claude -p
                else:
                    view = respond(slug, d, ablate_memory=ablate_mem, out_dir=out_dir,
                                   evidence_path=evidence_path, soul_path=soul_path)
            except Exception as e:                                # transient claude -p / parse failure
                print(f"  [skip] {slug} {d.isoformat()} {variant}: {e}")
                continue                                          # leave ungenerated; re-run fills the gap
            rep = audit_answer(view, ev_path, d)
            rows.append(_row(d, variant, view, rep))

    (base_dir / slug).mkdir(parents=True, exist_ok=True)
    (base_dir / slug / "walkforward.json").write_text(
        json.dumps({"subject": slug, "dates": [d.isoformat() for d in dates], "rows": rows}, indent=2))
    return rows
