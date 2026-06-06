"""doppelganger.run — CLI entrypoint.

Usage:
    python -m doppelganger.run ingest --subject eddy-lazzarin
    python -m doppelganger.run soul   --subject eddy-lazzarin --t0 2022-12-31
    python -m doppelganger.run memory --subject eddy-lazzarin --t0 2022-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from doppelganger import config
from doppelganger.ingest import ingest
from doppelganger.memory import load_memory
from doppelganger.respond import respond
from doppelganger.soul import extract_soul
from doppelganger.walkforward import quarter_ends, run_walkforward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doppelganger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="build identity + evidence stream for a subject")
    ing.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")

    soul = sub.add_parser("soul", help="build the frozen-at-T0 soul card for a subject")
    soul.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    soul.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")

    mem = sub.add_parser("memory", help="inspect the time-gated <=T memory view for a subject")
    mem.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    mem.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")

    resp = sub.add_parser("respond", help="answer a market-view query as the subject at date T")
    resp.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    resp.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")
    resp.add_argument("--query", default=None, help="optional custom query (default: market-view survey)")

    wf = sub.add_parser("walkforward", help="run respond() across a quarterly schedule (full + ablation)")
    wf.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    wf.add_argument("--start", default="2022-12-31", help="schedule start YYYY-MM-DD")
    wf.add_argument("--end", default=None, help="schedule end YYYY-MM-DD (default: latest evidence date)")
    wf.add_argument("--no-ablate", action="store_true", help="skip the soul-only ablation arm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "ingest":
        out = ingest(args.subject)
        print(f"wrote {out['evidence']} and {out['identity']}")
    elif args.cmd == "soul":
        path = extract_soul(args.subject, date.fromisoformat(args.t0))
        print(f"wrote {path}")
    elif args.cmd == "memory":
        mv = load_memory(args.subject, date.fromisoformat(args.t0))
        print(f"n_items={mv.n_items} max_date={mv.max_date}")
        print(mv.text[:1000])
    elif args.cmd == "respond":
        view = respond(args.subject, date.fromisoformat(args.t0), query=args.query)
        import json
        print(json.dumps(view, indent=2)[:2000])
    elif args.cmd == "walkforward":
        start = date.fromisoformat(args.start)
        if args.end:
            end = date.fromisoformat(args.end)
        else:
            ev = pd.read_parquet(config.OUT_DIR / args.subject / "evidence.parquet")
            end = pd.to_datetime(ev["timestamp"], utc=True).max().date()
        dates = quarter_ends(start, end)
        rows = run_walkforward(args.subject, dates, ablate=not args.no_ablate)
        print(f"{len(rows)} rows over {len(dates)} dates -> {config.OUT_DIR / args.subject / 'walkforward.json'}")


if __name__ == "__main__":
    main()
