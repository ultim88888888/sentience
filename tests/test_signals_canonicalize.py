import json
from unittest.mock import patch
from signals.schema import Citation, SignalItem
from signals.registry import Registry
from signals.canonicalize import canonicalize_items

def _raw(name, typ="sector", parent=None):
    return SignalItem(item=name, item_type=typ, parent_sector=parent, stance="bullish",
                      conviction=70, horizon="structural", rationale="r",
                      provenance="grounded", age_note=None,
                      citations=(Citation("2024-01-01", "q"),))

def test_fits_existing_and_mints_new():
    raw_items = [_raw("zero-knowledge proofs"), _raw("intent-based solvers"),
                 _raw("HYPE", typ="token", parent="perp dex")]
    mapping = json.dumps({"mapping": [
        {"raw": "zero-knowledge proofs", "canonical": "zk", "item_type": "sector",
         "parent_sector": None, "is_new": False},
        {"raw": "intent-based solvers", "canonical": "intent-solvers", "item_type": "sector",
         "parent_sector": None, "is_new": True},
        {"raw": "HYPE", "canonical": "HYPE", "item_type": "token",
         "parent_sector": "perp-dex", "is_new": True},
    ]})
    reg = Registry.seed()
    with patch("signals.canonicalize.run_claude", return_value=mapping):
        items, reg2 = canonicalize_items(raw_items, reg)
    canon = {i.item for i in items}
    assert canon == {"zk", "intent-solvers", "HYPE"}
    assert "intent-solvers" in reg2.sectors          # minted
    assert "HYPE" in reg2.tokens
    assert reg2.token_parent["HYPE"] == "perp-dex"
    # token HYPE's parent_sector rewritten to canonical
    assert next(i for i in items if i.item == "HYPE").parent_sector == "perp-dex"

def test_unmapped_item_falls_back_to_slug_not_dropped():
    raw_items = [_raw("Some Weird Sector")]
    with patch("signals.canonicalize.run_claude", return_value=json.dumps({"mapping": []})):
        items, reg = canonicalize_items(raw_items, Registry.seed())
    assert len(items) == 1                            # never silently dropped
    assert items[0].item == "some-weird-sector"       # deterministic slug fallback
