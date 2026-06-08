"""Overnight extractive distillation of in-range a16z articles (post-2021, >=500 chars).
Resumable. Output: data/signal/article_distillates.jsonl. Prereq for the blended A1 corpus
(full article bodies are ~800k tokens; distilled passages fit)."""
from signals.distill import build_article_distillate_cache
from signals import config

if __name__ == "__main__":
    p = build_article_distillate_cache(config.RESEARCH_ARTICLES, since="2021-01-01", min_chars=500)
    print("DONE:", p)
