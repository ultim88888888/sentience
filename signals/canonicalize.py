"""Agentic fit-or-mint: map free-form item names to the registry by SEMANTIC
judgment (not string match). The LLM decides existing-vs-new; bookkeeping
(append mints, rewrite items to canonical ids) is deterministic. Items are NEVER
silently dropped — an unmapped item falls back to a deterministic slug."""
from __future__ import annotations
import json
import re
from dataclasses import replace

from doppelganger.llm import run_claude
from signals.schema import SignalItem
from signals.registry import Registry

_SYSTEM = """You normalize crypto sector/token names to a controlled registry by MEANING,
not string match. For each raw item, decide whether it is the SAME concept as an existing
registry entry (return that canonical id, is_new=false) or genuinely new (propose a new
lowercase-kebab id, is_new=true). "zero-knowledge"/"zk proofs"/"validity proofs" are all
the existing `zk`. Tokens keep their ticker as canonical; give parent_sector as a registry
sector id (existing or newly proposed).

Output JSON only:
{"mapping": [{"raw": "<verbatim raw name>", "canonical": "<id>", "item_type": "sector|token",
  "parent_sector": "<sector-id-or-null>", "is_new": true|false}]}"""


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def canonicalize_items(raw_items: list[SignalItem], registry: Registry
                       ) -> tuple[list[SignalItem], Registry]:
    if not raw_items:
        return [], registry
    payload = {
        "registry": {"sectors": registry.sectors, "tokens": registry.tokens},
        "items": [{"raw": i.item, "item_type": i.item_type, "parent_sector": i.parent_sector}
                  for i in raw_items],
    }
    raw = run_claude(_SYSTEM, json.dumps(payload, indent=2))
    mapping = {m["raw"]: m for m in _extract_json(raw).get("mapping", [])}

    out: list[SignalItem] = []
    for it in raw_items:
        m = mapping.get(it.item)
        if m:
            canonical = m["canonical"]
            parent = m.get("parent_sector")
            if it.item_type == "token":
                registry.mint_token(canonical, parent_sector=parent)
                if parent:
                    registry.mint_sector(parent)
            else:
                registry.mint_sector(canonical)
        else:
            # never drop: deterministic slug fallback
            canonical = _slug(it.item)
            parent = _slug(it.parent_sector) if it.parent_sector else None
            (registry.mint_token(canonical, parent_sector=parent)
             if it.item_type == "token" else registry.mint_sector(canonical))
        out.append(replace(it, item=canonical, parent_sector=parent))
    return out, registry


def _extract_json(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(s[start:end + 1])
