"""Orchestrate the A1 signal pipeline and expose a CLI.

  python -m signals.run distill                      # one-time transcript cleaning
  python -m signals.run panel --start 2022-12-31 --end 2026-03-31 --interval quarterly
  python -m signals.run validate --t 2023-03-31      # full-vs-distilled extraction gate
"""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from signals import config
from signals.registry import load_registry, save_registry
from signals.extract import extract_a1
from signals.canonicalize import canonicalize_items
from signals.audit import audit_period
from signals.corpus import assemble_corpus
from signals.panel import derive_panel
from signals.distill import build_distillate_cache, load_distillates


def _month_end(d: date) -> date:
    return (d.replace(day=1) + relativedelta(months=1)) - relativedelta(days=1)


def rebalance_dates(start: date, end: date, interval: str) -> list[date]:
    step = 3 if interval == "quarterly" else 1
    if interval == "quarterly":
        # Snap to the first quarter-end month (3/6/9/12) >= start month
        qe_months = [3, 6, 9, 12]
        first_qe = next(m for m in qe_months if m >= start.month)
        cur = _month_end(start.replace(month=first_qe, day=1))
    else:
        cur = _month_end(start)
    out = []
    while cur <= end:
        out.append(cur)
        cur = _month_end(cur + relativedelta(months=step))
    return out


def build_panel(start: date, end: date, interval: str, *, window_months: int,
                twitter_paths, articles, distillates, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir or config.SIGNAL_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(out_dir / "registry.json")
    periods = []
    audits = []
    for t in rebalance_dates(start, end, interval):
        raw_period = extract_a1(t, window_months=window_months, twitter_paths=twitter_paths,
                                articles=articles, distillates=distillates)
        canon_items, registry = canonicalize_items(list(raw_period.items), registry)
        period = raw_period.__class__(as_of=raw_period.as_of, approach="A1",
                                      items=tuple(canon_items),
                                      risk_regime=raw_period.risk_regime, notes=raw_period.notes)
        corpus = assemble_corpus(t=t, window_months=window_months, twitter_paths=twitter_paths,
                                 articles=articles, distillates=distillates)
        rep = audit_period(period, corpus, t)
        audits.append({"as_of": t.isoformat(), "checked": rep.checked, "matched": rep.matched,
                       "hallucinated": len(rep.hallucinated), "leaked": len(rep.leaked)})
        (out_dir / "periods").mkdir(exist_ok=True)
        (out_dir / "periods" / f"{t.isoformat()}.json").write_text(json.dumps(period.to_dict(), indent=2))
        periods.append(period)

    save_registry(registry, out_dir / "registry.json")
    df = derive_panel(periods)
    df.to_parquet(out_dir / "signal_panel.parquet")
    (out_dir / "audit.json").write_text(json.dumps(audits, indent=2))
    return df


def validate_distillation(t: date, *, window_months: int, twitter_paths, articles,
                          distillates) -> dict:
    """Compare full-text vs distilled extraction on one tractable early date. Reports
    item-set overlap (Jaccard) so divergence is visible before scaling distillation."""
    full = extract_a1(t, window_months=window_months, twitter_paths=twitter_paths,
                      articles=articles, distillates={})            # transcripts excluded == full text path TBD (Task 10b)
    dist = extract_a1(t, window_months=window_months, twitter_paths=twitter_paths,
                      articles=articles, distillates=distillates)
    a = {i.item for i in full.items}
    b = {i.item for i in dist.items}
    jac = len(a & b) / len(a | b) if (a | b) else 1.0
    return {"as_of": t.isoformat(), "full_items": sorted(a), "distilled_items": sorted(b),
            "jaccard": jac}


def _tracked_twitter_paths() -> list[Path]:
    import yaml
    people = yaml.safe_load(Path(config.TRACKED_PEOPLE).read_text())["people"]
    paths = []
    for p in people:
        h = p.get("x_handle")
        if h and (config.TWITTER_DIR / f"{h}.parquet").exists():
            paths.append(config.TWITTER_DIR / f"{h}.parquet")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("distill")
    pp = sub.add_parser("panel")
    pp.add_argument("--start", required=True); pp.add_argument("--end", required=True)
    pp.add_argument("--interval", default="quarterly", choices=["quarterly", "monthly"])
    pp.add_argument("--window-months", type=int, default=config.DEFAULT_WINDOW_MONTHS)
    vp = sub.add_parser("validate"); vp.add_argument("--t", required=True)
    vp.add_argument("--window-months", type=int, default=config.DEFAULT_WINDOW_MONTHS)
    args = ap.parse_args()

    if args.cmd == "distill":
        build_distillate_cache(config.TRANSCRIPTS, config.RESEARCH_ARTICLES)
        return
    arts = pd.read_parquet(config.RESEARCH_ARTICLES)
    dist = load_distillates()
    tw = _tracked_twitter_paths()
    if args.cmd == "panel":
        build_panel(date.fromisoformat(args.start), date.fromisoformat(args.end),
                    args.interval, window_months=args.window_months,
                    twitter_paths=tw, articles=arts, distillates=dist)
    elif args.cmd == "validate":
        out = validate_distillation(date.fromisoformat(args.t), window_months=args.window_months,
                                    twitter_paths=tw, articles=arts, distillates=dist)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
