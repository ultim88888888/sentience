"""Fill Kominers' remaining big-window A2 periods via the new chunk-merge path. Resumable."""
from datetime import date
from pathlib import Path
from signals.run import build_member_panels
if __name__ == "__main__":
    res = build_member_panels([("scott-kominers","Scott Kominers","data/twitter/skominers.parquet")],
                              start=date(2022,12,31), end=date(2026,3,31), interval="quarterly",
                              window_months=18, distillates={}, out_root=Path("data/signal"))
    print("RESULT:", res)
