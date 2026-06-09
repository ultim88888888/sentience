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

AUDIT item_type: an id is a `token` ONLY if it is a single tradeable ticker (BTC, ARB, UNI, SOL).
If an id is a SECTOR, group, basket, thesis, or category (e.g. 'l2-tokens', 'memecoins',
'mkr-aave-comp', 'dao-gov-tokens', 'yield-bearing-stablecoins', 'revenue-generating-tokens'), its
item_type is `sector`, NOT token — correct it. The `item_type` you output is the CORRECT type, which
may differ from the input.

Output JSON only:
{"groups":[{"canonical":"BTC","item_type":"token","ids":["btc","BTC"]},
           {"canonical":"l2-scaling","item_type":"sector","ids":["l2-scaling"]}, ...]}"""


def reconcile_vocab(ids: list[dict]) -> tuple[dict, dict]:
    """LLM → (id_map, type_map).

    id_map: {original_id: canonical_id}. Any id the LLM omits maps to itself (never dropped).
    type_map: {canonical_id: corrected_item_type}. Unmapped ids keep their original type.
    """
    # Build a lookup of original types keyed by id for fallback
    orig_types: dict[str, str] = {d["id"]: d["item_type"] for d in ids}

    raw = run_claude(_RECONCILE_SYS, json.dumps({"ids": ids}, indent=2))
    groups = _extract_json(raw).get("groups", [])

    id_map: dict[str, str] = {}
    type_map: dict[str, str] = {}

    for g in groups:
        canon = g.get("canonical")
        corrected_type = g.get("item_type")
        if not canon:
            continue
        for old in g.get("ids", []):
            id_map[old] = canon
        if corrected_type:
            type_map[canon] = corrected_type

    # Never drop an id — self-map and keep original type for anything the LLM omitted
    for d in ids:
        oid = d["id"]
        id_map.setdefault(oid, oid)
        canonical = id_map[oid]
        type_map.setdefault(canonical, orig_types.get(oid, d["item_type"]))

    return id_map, type_map


def apply_map(df: pd.DataFrame, id_map: dict, type_map: dict) -> pd.DataFrame:
    """Return a copy of df with item and parent_sector remapped through id_map,
    and item_type corrected via type_map."""
    df = df.copy()
    df["item"] = df["item"].map(lambda x: id_map.get(str(x), x))
    df["parent_sector"] = df["parent_sector"].map(
        lambda x: id_map.get(str(x), x) if pd.notna(x) else x)
    df["item_type"] = df.apply(
        lambda r: type_map.get(str(r["item"]), r["item_type"]), axis=1)
    return df


def run_reconciliation(panels: dict, *, out_dir: Path | None = None) -> dict:
    """panels: {name: panel_parquet_path}.

    Writes <out_dir>/<name>_reconciled.parquet for each panel plus
    reconciliation_map.json and type_corrections.json. Returns the id_map dict.

    NON-DESTRUCTIVE: never writes to source paths.
    """
    out_dir = Path(out_dir or (config.SIGNAL_OUT_DIR / "reconciled"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = collect_ids(*panels.values())
    id_map, type_map = reconcile_vocab(ids)

    (out_dir / "reconciliation_map.json").write_text(json.dumps(id_map, indent=2))

    # Build type_corrections audit: ids whose type changed
    orig_types: dict[str, str] = {d["id"]: d["item_type"] for d in ids}
    corrections = []
    for d in ids:
        oid = d["id"]
        canon = id_map[oid]
        old_t = orig_types[oid]
        new_t = type_map.get(canon, old_t)
        if old_t != new_t:
            corrections.append({"id": canon, "old_type": old_t, "new_type": new_t})
    (out_dir / "type_corrections.json").write_text(json.dumps(corrections, indent=2))

    for name, path in panels.items():
        dst = out_dir / f"{name}_reconciled.parquet"
        assert Path(dst) != Path(path), "refuse to overwrite source"
        apply_map(pd.read_parquet(path), id_map, type_map).to_parquet(dst)

    return id_map
