"""Pre-2022 OUT-OF-SAMPLE extension — the decisive test of whether the corpus signal generalizes to an
INDEPENDENT regime (2021 top + 2022 bear) it was never tuned on. A2 per-member views → A2b council, monthly
2021-09..2022-11 (twitter depth permits ~2021-09+). Reuses the validated extraction pipeline (chunk-merge
for prolific members, retry+backoff). Resumable. A2b only (no souls needed — A2b uses extraction)."""
import json
from datetime import date
from pathlib import Path
from signals.run import build_member_panels, rebalance_dates
from signals.council import run_a2b

MEMBERS = [("daren-matsuoka","Daren Matsuoka","data/twitter/DarenMatsuoka.parquet"),
           ("justin-thaler","Justin Thaler","data/twitter/SuccinctJT.parquet"),
           ("tim-roughgarden","Tim Roughgarden","data/twitter/Tim_Roughgarden.parquet"),
           ("ali-yahya","Ali Yahya","data/twitter/alive_.parquet"),
           ("chris-dixon","Chris Dixon","data/twitter/cdixon.parquet"),
           ("eddy-lazzarin","Eddy Lazzarin","data/twitter/eddylazzarin.parquet"),
           ("guy-wuollet","Guy Wuollet","data/twitter/guywuolletjr.parquet"),
           ("jason-rosenthal","Jason Rosenthal","data/twitter/jasonrosenthal.parquet"),
           ("miles-jennings","Miles Jennings","data/twitter/milesjennings.parquet"),
           ("scott-kominers","Scott Kominers","data/twitter/skominers.parquet")]

if __name__ == "__main__":
    dates = rebalance_dates(date(2021,9,30), date(2022,11,30), "monthly")
    base = Path("data/signal/pre2022")
    print(f"[pre2022] {len(dates)} OOS months 2021-09..2022-11 x {len(MEMBERS)} members", flush=True)
    build_member_panels(MEMBERS, start=date(2021,9,30), end=date(2022,11,30), interval="monthly",
                        window_months=18, distillates={}, out_root=base)
    print("[pre2022] member views done; A2b council...", flush=True)
    run_a2b(dates, members_root=base/"members", out_dir=base/"a2b_council",
            registry_path=base/"registry.json")
    print("PRE2022 A2B DONE", flush=True)
