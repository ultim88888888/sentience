"""Orchestrate the viability study: both attribution modes x Study A + Study B.

Run:  python -m study.run            (uses cached prices if present)
      python -m study.run --refresh  (re-fetch all prices from Coinglass)

Writes findings.md (committed) + data/study/coverage_heatmap.png + intermediate panels.
"""
import argparse
import sys

import yaml

from . import coverage as coverage_mod
from . import returns as returns_mod
from . import signal as signal_mod
from . import study_basket, study_token, findings
from .config import (ATTRIBUTION_MODES, BASKETS_YAML, CONVICTION_AGGS, FINDINGS_MD,
                     STUDY_DIR)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_baskets() -> dict:
    with open(BASKETS_YAML) as f:
        return yaml.safe_load(f)


def all_tokens(baskets_cfg: dict) -> list[str]:
    toks = set()
    for tokens in baskets_cfg["baskets"].values():
        toks.update(tokens)
    return sorted(toks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch prices (ignore cache)")
    args = ap.parse_args()
    use_cache = not args.refresh

    cfg = load_baskets()
    benchmark = cfg["benchmark"]
    corpus = coverage_mod.load_corpus()

    _log("Fetching token returns...")
    token_rets = returns_mod.monthly_token_returns(all_tokens(cfg), use_cache=use_cache)
    basket_rets = returns_mod.basket_returns(token_rets, cfg)

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    # Heatmap shows the fractional-mode coverage share — selected by name, not loop order.
    heatmap_coverage = coverage_mod.monthly_coverage(corpus, cfg, mode="fractional")
    results = {}
    for mode in ATTRIBUTION_MODES:
        _log(f"Running attribution mode: {mode}")
        cov = coverage_mod.monthly_coverage(corpus, cfg, mode=mode)
        panel_a = signal_mod.basket_signal_panel(cov, basket_rets, benchmark)
        res_a = study_basket.run_study_a(panel_a)
        res_b = {agg: study_token.run_study_b(cov, token_rets, cfg, agg)
                 for agg in CONVICTION_AGGS}
        results[mode] = {"study_a": res_a, "study_b": res_b}
        panel_a.to_parquet(STUDY_DIR / f"panel_a_{mode}.parquet")

    findings.render_heatmap(heatmap_coverage)
    md = findings.render_markdown(results)
    FINDINGS_MD.write_text(md)
    _log(f"Wrote {FINDINGS_MD}")
    print(md)


if __name__ == "__main__":
    main()
