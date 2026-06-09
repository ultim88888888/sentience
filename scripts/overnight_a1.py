"""A1 member-agnostic consensus: distilled tweets + distilled transcripts (no raw tweets, no articles).
Waits for tweet distillation to finish, then runs the 14-quarter A1 panel via stdin (AUP-clean ~86k tok).
Own registry (registry_a1.json) to avoid races with the member run. Resumable."""
import time, subprocess, glob
from datetime import date
from pathlib import Path
import pandas as pd
from signals.run import build_panel
from signals.distill import load_distillates, load_tweet_distillates

print("[a1] waiting for tweet distillation DONE marker...", flush=True)
while not Path("data/signal/tweet_distillates.DONE").exists():
    time.sleep(30)
tw_dist = load_tweet_distillates("data/signal/tweet_distillates.jsonl")
tdist = load_distillates("data/signal/transcript_distillates.jsonl")
print(f"[a1] distilled tweets={len(tw_dist)} transcript-docs={len(tdist)} — running A1 panel", flush=True)

arts = pd.read_parquet("data/a16z_research/articles.parquet")
df = build_panel(date(2022,12,31), date(2026,3,31), "quarterly", window_months=18,
                 twitter_paths=[], articles=arts, distillates=tdist,
                 tweet_distillates=tw_dist, article_distillates={},
                 out_dir=Path("data/signal"), registry_path=Path("data/signal/registry_a1.json"))
print(f"[a1] DONE — {len(df)} rows -> data/signal/signal_panel.parquet", flush=True)
