"""1B v1 collection: universe (coins-markets liquid ∪ signal-named tokens ∪ majors) -> LLM sector map
-> pull daily OHLCV/funding/OI history. Point-in-time OI floor applied downstream. Survivorship caveat:
universe is current-liquid + signal-named (broad historical set deferred)."""
import json
import pandas as pd
from pathlib import Path
from market_data.coinglass import fetch_coins_markets, pull
from signals.market import build_universe, classify_sectors
from signals.registry import load_registry

OUT = Path("data/market_data"); OUT.mkdir(parents=True, exist_ok=True)

# 1. universe
cm = fetch_coins_markets()
liquid = build_universe(cm)
print(f"[1b] coins-markets: {len(cm)} coins, {len(liquid)} clear OI floor", flush=True)
a1 = pd.read_parquet("data/signal/reconciled/a1_reconciled.parquet")
a2 = pd.read_parquet("data/signal/reconciled/a2a_reconciled.parquet")
sig_tokens = set(a1[a1.item_type=="token"].item) | set(a2[a2.item_type=="token"].item)
universe = sorted({*liquid, *sig_tokens, "BTC", "ETH"})
print(f"[1b] signal-named tokens: {len(sig_tokens)} | final universe: {len(universe)}", flush=True)
(OUT/"universe.json").write_text(json.dumps(universe, indent=2))

# 2. LLM sector classification
reg = load_registry("data/signal/registry.json")
sector_map = classify_sectors(universe, reg.sectors)
(OUT/"sector_map.json").write_text(json.dumps(sector_map, indent=2))
from collections import Counter
print(f"[1b] sector map: {len(sector_map)} tokens; top sectors {Counter(sector_map.values()).most_common(6)}", flush=True)

# 3. pull daily history (OHLCV/funding/OI) for the universe
res = pull(universe, persist=True, out_dir=OUT)
ok = sum(1 for v in res.values() if v.get("ohlcv") is not None and len(v["ohlcv"]))
print(f"[1b] pulled history for {ok}/{len(universe)} symbols -> {OUT}", flush=True)
print("[1b] DONE", flush=True)
