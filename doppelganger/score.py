"""doppelganger.score — deterministic scorers + the held-out scoring orchestrator + memo."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from doppelganger import config
from doppelganger.judge import judge_step, post_t_evidence

_NAMED = ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned"]


def _names(view: dict, keys: list[str]) -> set[str]:
    out: set[str] = set()
    for k in keys:
        for it in view.get(k, []) or []:
            n = (it.get("name") or "").strip().lower()
            if n:
                out.add(n)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def discrimination(view_a: dict, view_b: dict) -> dict:
    sa, sb = _names(view_a, ["sectors_excited", "sectors_concerned"]), _names(view_b, ["sectors_excited", "sectors_concerned"])
    ta, tb = _names(view_a, ["tokens_excited", "tokens_concerned"]), _names(view_b, ["tokens_excited", "tokens_concerned"])
    return {
        "sector_overlap": _jaccard(sa, sb), "token_overlap": _jaccard(ta, tb),
        "shared_sectors": sorted(sa & sb), "shared_tokens": sorted(ta & tb),
    }


def coverage_trajectory(rows: list[dict]) -> list[dict]:
    return [{"date": r["date"], "grounded": r["grounded"], "persisted": r["persisted"],
             "extrapolated": r["extrapolated"]}
            for r in rows if r.get("variant") == "full"]


def score_subject(slug: str, *, horizon_months: int = 6, out_dir: Path | None = None,
                  evidence_path: Path | None = None) -> dict:
    base = Path(out_dir or config.OUT_DIR) / slug
    wf = json.loads((base / "walkforward.json").read_text())
    ev_path = evidence_path or (base / "evidence.parquet")

    steps = []
    for ds in wf["dates"]:
        t0 = date.fromisoformat(ds)
        post = post_t_evidence(slug, t0, horizon_months, evidence_path=ev_path)
        if not post:
            continue                                   # no held-out future -> unscorable
        row = {"date": ds, "full_confirm_rate": None, "ablation_confirm_rate": None,
               "lift": None, "missed_changes": [], "n_missed_changes": 0}
        for variant, sub in [("full", "views"), ("ablation", "views_ablation")]:
            vpath = base / sub / f"{ds}.json"
            if not vpath.exists():
                continue
            view = json.loads(vpath.read_text())
            jp = base / "judge" / sub / f"{ds}.json"
            v = judge_step(view, post, wf["subject"], t0, judge_path=jp)
            row[f"{variant}_confirm_rate"] = v.get("confirm_rate")
            if variant == "full":
                row["missed_changes"] = v.get("missed_changes", [])
                row["n_missed_changes"] = len(v.get("missed_changes", []))
        if row["full_confirm_rate"] is not None and row["ablation_confirm_rate"] is not None:
            row["lift"] = row["full_confirm_rate"] - row["ablation_confirm_rate"]
        steps.append(row)

    lifts = [s["lift"] for s in steps if s["lift"] is not None]
    metrics = {
        "subject": slug, "horizon_months": horizon_months,
        "mean_lift": (sum(lifts) / len(lifts)) if lifts else None,
        "steps": steps, "coverage": coverage_trajectory(wf["rows"]),
    }
    (base / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
