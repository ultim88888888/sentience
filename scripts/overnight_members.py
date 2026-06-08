"""Overnight A2a per-member view generation for the 10 tracked team members.
Tweets-only corpus, 18mo trailing window, quarterly 2022-12-31..2026-03-31.
Resumable: re-running skips periods already written. Shared registry across members."""
from datetime import date
from pathlib import Path
from signals.run import build_member_panels

MEMBERS = [
    ("daren-matsuoka",  "Daren Matsuoka",  "data/twitter/DarenMatsuoka.parquet"),
    ("justin-thaler",   "Justin Thaler",   "data/twitter/SuccinctJT.parquet"),
    ("tim-roughgarden", "Tim Roughgarden", "data/twitter/Tim_Roughgarden.parquet"),
    ("ali-yahya",       "Ali Yahya",       "data/twitter/alive_.parquet"),
    ("chris-dixon",     "Chris Dixon",     "data/twitter/cdixon.parquet"),
    ("eddy-lazzarin",   "Eddy Lazzarin",   "data/twitter/eddylazzarin.parquet"),
    ("guy-wuollet",     "Guy Wuollet",     "data/twitter/guywuolletjr.parquet"),
    ("jason-rosenthal", "Jason Rosenthal", "data/twitter/jasonrosenthal.parquet"),
    ("miles-jennings",  "Miles Jennings",  "data/twitter/milesjennings.parquet"),
    ("scott-kominers",  "Scott Kominers",  "data/twitter/skominers.parquet"),
]

if __name__ == "__main__":
    res = build_member_panels(MEMBERS, start=date(2022, 12, 31), end=date(2026, 3, 31),
                              interval="quarterly", window_months=18, distillates={},
                              out_root=Path("data/signal"))
    print("RESULT:", res)
