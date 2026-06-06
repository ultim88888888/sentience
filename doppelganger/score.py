"""doppelganger.score — deterministic scorers + the held-out scoring orchestrator + memo."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from doppelganger import config

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
