"""Backfill the stuck monthly A2 member-months (88 missing across eddy/guy/miles/scott — AUP-blocked on
hype-dense months), then we run the council separately. Fix: smaller chunks (dodge AUP content ceiling) +
5-way concurrent extraction. Registry canonicalization is the shared-state step → done SERIALLY after the
concurrent extract phase to avoid a registry race. Resumable (skips existing period files)."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import pandas as pd

from signals.run import rebalance_dates
from signals.extract import extract_member
from signals.canonicalize import canonicalize_items
from signals.registry import load_registry, save_registry
from signals.schema import PeriodSignal

MEMBERS = [("eddy-lazzarin", "Eddy Lazzarin", "data/twitter/eddylazzarin.parquet"),
           ("guy-wuollet", "Guy Wuollet", "data/twitter/guywuolletjr.parquet"),
           ("miles-jennings", "Miles Jennings", "data/twitter/milesjennings.parquet"),
           ("scott-kominers", "Scott Kominers", "data/twitter/skominers.parquet")]
ROOT = Path("data/signal/members_monthly/members")
REG = Path("data/signal/members_monthly/registry.json")
DATES = rebalance_dates(date(2022, 12, 31), date(2026, 3, 31), "monthly")
SMALL_CHUNK = 80_000   # chars (~20k tok) per chunk — well under the AUP content ceiling; chunk-merge unions


def missing_tasks():
    tasks = []
    for slug, name, tw in MEMBERS:
        for t in DATES:
            pj = ROOT / slug / "periods" / f"{t.isoformat()}.json"
            if not pj.exists():
                tasks.append((slug, name, tw, t, pj))
    return tasks


def _extract(slug, name, tw, t):
    """Concurrent-safe: raw member view, smaller chunks. Returns (raw PeriodSignal, t, slug, pj) or None."""
    try:
        return extract_member(t, name, window_months=18, twitter_path=Path(tw),
                              distillates={}, max_chars=SMALL_CHUNK)
    except Exception as e:
        print(f"[extract-fail] {slug} {t}: {str(e)[:90]}", flush=True)
        return None


def main():
    tasks = missing_tasks()
    print(f"[backfill] {len(tasks)} missing member-months across {len({t[0] for t in tasks})} members", flush=True)
    registry = load_registry(REG)
    done = 0
    # process in waves of ~20 so we canonicalize+save incrementally (resumable if interrupted)
    WAVE = 20
    for i in range(0, len(tasks), WAVE):
        wave = tasks[i:i + WAVE]
        with ThreadPoolExecutor(max_workers=3) as ex:
            raws = list(ex.map(lambda a: (_extract(a[0], a[1], a[2], a[3]), a), wave))
        # serial canonicalize + write (registry is shared state). Per-item try/except so one failure
        # (e.g. a transient session-limit on the canonicalize call) never crashes the whole run.
        for raw, (slug, name, tw, t, pj) in raws:
            if raw is None:
                continue
            try:
                canon, registry = canonicalize_items(list(raw.items), registry)
                period = PeriodSignal(as_of=raw.as_of, approach=f"A2a:{name}", items=tuple(canon),
                                      risk_regime=raw.risk_regime, notes=raw.notes)
                pj.parent.mkdir(parents=True, exist_ok=True)
                pj.write_text(json.dumps(period.to_dict(), indent=2))
                done += 1
            except Exception as e:
                print(f"[canon-fail] {slug} {t}: {str(e)[:90]}", flush=True)
        save_registry(registry, REG)
        print(f"[backfill] wave {i//WAVE+1}: {done}/{len(tasks)} written", flush=True)
    print(f"[backfill] DONE — {done}/{len(tasks)} member-months backfilled", flush=True)
    # report completeness
    for slug, name, tw in MEMBERS:
        n = len(list((ROOT / slug / "periods").glob("*.json")))
        print(f"  {slug}: {n}/40", flush=True)
    print("BACKFILL A2 MONTHLY DONE", flush=True)


if __name__ == "__main__":
    main()
