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


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return str(x)


def write_memo(metrics: dict, discrimination: dict | None = None, *, out_dir: Path | None = None) -> Path:
    slug = metrics["subject"]
    lift = metrics.get("mean_lift")
    lines = [
        f"# Doppelganger findings — {slug}",
        "",
        f"**Headline — mean corpus-lift: {_fmt(lift)}** "
        f"(full confirm-rate minus the soul-less ablation, horizon {metrics['horizon_months']} mo).",
        "",
        "_Caveats (read these with the number):_ the lift nets out **persistence** "
        "(both arms get easy credit for stable views, which partly cancels) and bounds "
        "**model-hindsight** (the **soul-less** arm is the parametric floor — what Opus produces "
        "with no corpus). The judge scored only from the subject's provided later statements.",
        "",
        "## Per-step held-out prediction",
        "",
        "| date | full | ablation (soul-less) | lift |",
        "|---|---|---|---|",
    ]
    for s in metrics["steps"]:
        lines.append(f"| {s['date']} | {_fmt(s['full_confirm_rate'])} | "
                     f"{_fmt(s['ablation_confirm_rate'])} | {_fmt(s['lift'])} |")
    lines += ["", "## Missed changes (foresight gaps)", ""]
    any_missed = False
    for s in metrics["steps"]:
        for mc in s.get("missed_changes", []):
            lines.append(f"- ({s['date']}) {mc}")
            any_missed = True
    if not any_missed:
        lines.append("- none flagged")
    lines += ["", "## Coverage trajectory (full arm)", "",
              "| date | grounded | persisted | extrapolated |", "|---|---|---|---|"]
    for c in metrics.get("coverage", []):
        lines.append(f"| {c['date']} | {c['grounded']} | {c['persisted']} | {c['extrapolated']} |")
    if discrimination is not None:
        lines += ["", "## Discrimination", "",
                  f"- sector overlap vs comparator: {_fmt(discrimination.get('sector_overlap'))}",
                  f"- token overlap vs comparator: {_fmt(discrimination.get('token_overlap'))}"]

    out = Path(out_dir or config.OUT_DIR) / slug / "findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def discrimination_trajectory(slug_a: str, slug_b: str, *, out_dir: Path | None = None) -> dict:
    """Discrimination between two subjects at every date BOTH have a view — the
    'distinct minds across time' check. Returns per-date overlaps + the means."""
    base = Path(out_dir or config.OUT_DIR)
    da, db = base / slug_a / "views", base / slug_b / "views"
    dates_a = {p.stem for p in da.glob("*.json")} if da.exists() else set()
    dates_b = {p.stem for p in db.glob("*.json")} if db.exists() else set()

    pairs = []
    for ds in sorted(dates_a & dates_b):
        va = json.loads((da / f"{ds}.json").read_text())
        vb = json.loads((db / f"{ds}.json").read_text())
        pairs.append({"date": ds, **discrimination(va, vb)})

    so = [p["sector_overlap"] for p in pairs]
    to = [p["token_overlap"] for p in pairs]
    return {
        "subject_a": slug_a, "subject_b": slug_b, "pairs": pairs,
        "mean_sector_overlap": (sum(so) / len(so)) if so else None,
        "mean_token_overlap": (sum(to) / len(to)) if to else None,
    }
