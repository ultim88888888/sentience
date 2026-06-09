"""Monthly A1 consensus (~40 points vs quarterly's 14). Reuses the quarterly period extractions
(quarter-ends are valid month-ends) + the already-built distilled-tweet/transcript caches, so only
the ~26 non-quarter-end months are newly extracted. Separate output dir (preserves quarterly panel),
shared A1 registry for vocab consistency. Resumable."""
import shutil, glob
from datetime import date
from pathlib import Path
import pandas as pd
from signals.run import build_panel
from signals.distill import load_distillates, load_tweet_distillates

out = Path("data/signal/a1_monthly"); (out/"periods").mkdir(parents=True, exist_ok=True)
for f in glob.glob("data/signal/periods/*.json"):          # reuse quarterly extractions
    dst = out/"periods"/Path(f).name
    if not dst.exists(): shutil.copy(f, dst)
tw = load_tweet_distillates("data/signal/tweet_distillates.jsonl")
td = load_distillates("data/signal/transcript_distillates.jsonl")
arts = pd.read_parquet("data/a16z_research/articles.parquet")
print(f"[a1-monthly] reused {len(glob.glob(str(out/'periods'/'*.json')))} quarterly periods; distilled tweets={len(tw)}", flush=True)
df = build_panel(date(2022,12,31), date(2026,3,31), "monthly", window_months=18,
                 twitter_paths=[], articles=arts, distillates=td,
                 tweet_distillates=tw, article_distillates={},
                 out_dir=out, registry_path=Path("data/signal/registry_a1.json"))
print(f"[a1-monthly] DONE — {len(df)} rows, {df['as_of'].nunique()} months", flush=True)
