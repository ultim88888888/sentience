"""Background universe cleanup: pull 1000PEPE/1000SHIB/etc (Binance perp names) and save under the
base ticker (returns identical), and gate stablecoins/macro out of universe.json."""
import json
from pathlib import Path
from market_data.coinglass import pull, save_symbol
from signals.market import EXCLUDE
OUT = Path("data/market_data")
ALIASES = {"PEPE":"1000PEPE","SHIB":"1000SHIB","BONK":"1000BONK","FLOKI":"1000FLOKI"}
res = pull(list(ALIASES.values()), persist=False, out_dir=OUT)
for base, src in ALIASES.items():
    fr = res.get(src)
    if fr and fr.get("ohlcv") is not None and len(fr["ohlcv"]):
        save_symbol(base, fr, out_dir=OUT)
        print(f"[fix] {src} -> saved as {base}: {len(fr['ohlcv'])} rows", flush=True)
u = json.loads((OUT/"universe.json").read_text())
u = [s for s in u if s not in EXCLUDE]
for t in ALIASES:
    if t not in u: u.append(t)
(OUT/"universe.json").write_text(json.dumps(sorted(u), indent=2))
print(f"[fix] universe.json cleaned -> {len(u)} symbols (DAI/USDT gated, memecoins added)", flush=True)
print("[fix] DONE", flush=True)
