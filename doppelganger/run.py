"""doppelganger.run — CLI entrypoint.

Usage:
    python -m doppelganger.run ingest --subject eddy-lazzarin
    python -m doppelganger.run soul   --subject eddy-lazzarin --t0 2022-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

from doppelganger.ingest import ingest
from doppelganger.soul import extract_soul


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doppelganger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="build identity + evidence stream for a subject")
    ing.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")

    soul = sub.add_parser("soul", help="build the frozen-at-T0 soul card for a subject")
    soul.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    soul.add_argument("--t0", required=True, help="cutoff date YYYY-MM-DD, e.g. 2022-12-31")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "ingest":
        out = ingest(args.subject)
        print(f"wrote {out['evidence']} and {out['identity']}")
    elif args.cmd == "soul":
        path = extract_soul(args.subject, date.fromisoformat(args.t0))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
