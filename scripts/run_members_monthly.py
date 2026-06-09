"""Monthly A2 per-member views — reuses the quarterly month-end JSONs already in members/<slug>/
(quarter-ends are valid month-ends), extracts only the new in-between months. ~40 months x 10 members.
Shared registry; chunk-merge for prolific members. Resumable."""
from datetime import date
from pathlib import Path
from signals.run import build_member_panels
MEMBERS=[("daren-matsuoka","Daren Matsuoka","data/twitter/DarenMatsuoka.parquet"),
 ("justin-thaler","Justin Thaler","data/twitter/SuccinctJT.parquet"),
 ("tim-roughgarden","Tim Roughgarden","data/twitter/Tim_Roughgarden.parquet"),
 ("ali-yahya","Ali Yahya","data/twitter/alive_.parquet"),("chris-dixon","Chris Dixon","data/twitter/cdixon.parquet"),
 ("eddy-lazzarin","Eddy Lazzarin","data/twitter/eddylazzarin.parquet"),("guy-wuollet","Guy Wuollet","data/twitter/guywuolletjr.parquet"),
 ("jason-rosenthal","Jason Rosenthal","data/twitter/jasonrosenthal.parquet"),("miles-jennings","Miles Jennings","data/twitter/milesjennings.parquet"),
 ("scott-kominers","Scott Kominers","data/twitter/skominers.parquet")]
if __name__=="__main__":
    res=build_member_panels(MEMBERS, start=date(2022,12,31), end=date(2026,3,31), interval="monthly",
                            window_months=18, distillates={}, out_root=Path("data/signal/members_monthly"))
    print("MONTHLY A2 DONE:", res, flush=True)
