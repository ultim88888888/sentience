"""One-time extractive tweet distillation (A1-only): keep trade-relevant stance-bearing tweets,
drop chatter. All 10 handles, batched (AUP-safe), cached, resumable. -> data/signal/tweet_distillates.jsonl"""
import glob
from signals.distill import build_tweet_distillate_cache
if __name__ == "__main__":
    paths = sorted(glob.glob("data/twitter/*.parquet"))
    print(f"distilling tweets from {len(paths)} handles...", flush=True)
    p = build_tweet_distillate_cache(paths, since="2021-01-01")
    print("DONE:", p, flush=True)
