"""A1 blended consensus — LEAN: tweets + transcript distillates only (article_distillates={}
suppresses full bodies AND adds no research passages; research-essay distillation deferred).
Waits only for transcript distillation to finish, then runs in parallel with the member run on
its OWN registry (registry_a1.json) to avoid a write race. Resumable."""
import time, subprocess, glob
from datetime import date
from pathlib import Path
import pandas as pd
from signals.run import build_panel
from signals.distill import load_distillates

def transcript_distill_running():
    r = subprocess.run(["pgrep","-f","signals.run distill"], capture_output=True, text=True)
    return bool(r.stdout.strip())

print("[a1] waiting for transcript distillation to finish...", flush=True)
while transcript_distill_running():
    time.sleep(30)
print("[a1] transcript distill done — running A1 panel (lean: tweets + transcript distillates)", flush=True)

tw = sorted(glob.glob("data/twitter/*.parquet"))
arts = pd.read_parquet("data/a16z_research/articles.parquet")
tdist = load_distillates("data/signal/transcript_distillates.jsonl")
print(f"[a1] transcript distillates: {len(tdist)}", flush=True)

df = build_panel(date(2022,12,31), date(2026,3,31), "quarterly", window_months=18,
                 twitter_paths=tw, articles=arts, distillates=tdist, article_distillates={},
                 out_dir=Path("data/signal"), registry_path=Path("data/signal/registry_a1.json"))
print(f"[a1] DONE — {len(df)} panel rows -> data/signal/signal_panel.parquet", flush=True)
