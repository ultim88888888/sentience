"""doppelganger.run — CLI entrypoint.

Usage:
    python -m doppelganger.run ingest --subject eddy-lazzarin
"""

from __future__ import annotations

import argparse

from doppelganger.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(prog="doppelganger")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="build identity + evidence stream for a subject")
    ing.add_argument("--subject", required=True, help="subject slug, e.g. eddy-lazzarin")
    args = parser.parse_args()

    if args.cmd == "ingest":
        out = ingest(args.subject)
        print(f"wrote {out['evidence']} and {out['identity']}")


if __name__ == "__main__":
    main()
