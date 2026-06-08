"""Chained A1 blended-consensus panel. Waits for article distillation + the per-member run to
finish (so A1 includes research passages and extends the member-populated SHARED registry with
no write race), then runs the full A1 panel. Resumable. Output: data/signal/{signal_panel.parquet,
periods/,audit.json,registry.json}."""
import time, subprocess, glob
from datetime import date
from pathlib import Path
import pandas as pd
from signals.run import build_panel
from signals.distill import load_distillates

def still_running():
    r = subprocess.run(["pgrep","-f","overnight_articles|overnight_members"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())

print("[a1] waiting for article-distill + member run to finish...", flush=True)
while still_running():
    time.sleep(120)
print("[a1] prerequisites done — running A1 panel", flush=True)

tw = sorted(glob.glob("data/twitter/*.parquet"))
arts = pd.read_parquet("data/a16z_research/articles.parquet")
tdist = load_distillates("data/signal/transcript_distillates.jsonl")
adist = load_distillates("data/signal/article_distillates.jsonl")
print(f"[a1] distillates: transcripts={len(tdist)} articles={len(adist)}", flush=True)

df = build_panel(date(2022,12,31), date(2026,3,31), "quarterly", window_months=18,
                 twitter_paths=tw, articles=arts, distillates=tdist, article_distillates=adist,
                 out_dir=Path("data/signal"))   # shares data/signal/registry.json (members ran first)
print(f"[a1] DONE — {len(df)} panel rows -> data/signal/signal_panel.parquet", flush=True)
