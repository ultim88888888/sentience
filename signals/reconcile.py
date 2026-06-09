"""Non-destructive LLM nomenclature reconciliation across signal panels.

A1 and A2a canonicalized against separate registries, so the same concept can appear under
different ids (casing: btc/BTC; synonyms). This pass has an LLM group same-concept ids, picks one
canonical id per group, and writes RECONCILED COPIES (raw panels untouched) + the mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from doppelganger.llm import run_claude
from signals import config
from signals.extract import _extract_json


def collect_ids(*panel_paths) -> list[dict]:
    """Union of distinct (item, item_type) across the given panel parquets, plus any
    parent_sector ids (as sectors). Returns [{"id":..., "item_type":...}] sorted."""
    seen: dict[str, str] = {}
    for p in panel_paths:
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            seen[str(r["item"])] = r["item_type"]
            ps = r.get("parent_sector")
            if pd.notna(ps) and str(ps):
                seen.setdefault(str(ps), "sector")
    return [{"id": k, "item_type": v} for k, v in sorted(seen.items())]


_RECONCILE_SYS = """You unify crypto sector/token ids that came from two pipelines which canonicalized
independently. The SAME concept may appear under different ids — casing variants (btc/BTC, eth/ETH) or
true synonyms (e.g. 'l2-scaling'/'layer-2-scaling'). Group ONLY ids that mean the SAME underlying thing.

Be CONSERVATIVE — do NOT merge related-but-distinct things:
- a token (e.g. ETH) is NOT the same as a sector (e.g. 'ethereum-ecosystem' / 'l2-scaling');
- 'l2-scaling' (a sector) is NOT 'l2-tokens' (a token group);
- different actual sectors stay separate.

For each group pick ONE canonical id: tokens = UPPERCASE ticker; sectors = lowercase-kebab. Every input
id must appear in exactly one group (a singleton group if it has no synonym).

Output JSON only:
{"groups":[{"canonical":"BTC","item_type":"token","ids":["btc","BTC"]},
           {"canonical":"l2-scaling","item_type":"sector","ids":["l2-scaling"]}, ...]}"""


def reconcile_vocab(ids: list[dict]) -> dict:
    """LLM → {original_id: canonical_id}. Any id the LLM omits maps to itself (never dropped)."""
    raw = run_claude(_RECONCILE_SYS, json.dumps({"ids": ids}, indent=2))
    groups = _extract_json(raw).get("groups", [])
    m: dict[str, str] = {}
    for g in groups:
        canon = g.get("canonical")
        for old in g.get("ids", []):
            if canon:
                m[old] = canon
    for d in ids:           # never drop an id
        m.setdefault(d["id"], d["id"])
    return m


def apply_map(df: pd.DataFrame, m: dict) -> pd.DataFrame:
    """Return a copy of df with item and parent_sector remapped through m."""
    df = df.copy()
    df["item"] = df["item"].map(lambda x: m.get(str(x), x))
    df["parent_sector"] = df["parent_sector"].map(
        lambda x: m.get(str(x), x) if pd.notna(x) else x)
    return df


def run_reconciliation(panels: dict, *, out_dir: Path | None = None) -> dict:
    """panels: {name: panel_parquet_path}.

    Writes <out_dir>/<name>_reconciled.parquet for each panel plus
    reconciliation_map.json. Returns the mapping dict.

    NON-DESTRUCTIVE: never writes to source paths.
    """
    out_dir = Path(out_dir or (config.SIGNAL_OUT_DIR / "reconciled"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = collect_ids(*panels.values())
    m = reconcile_vocab(ids)

    (out_dir / "reconciliation_map.json").write_text(json.dumps(m, indent=2))

    for name, path in panels.items():
        dst = out_dir / f"{name}_reconciled.parquet"
        assert Path(dst) != Path(path), "refuse to overwrite source"
        apply_map(pd.read_parquet(path), m).to_parquet(dst)

    return m
