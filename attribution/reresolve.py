"""Re-resolve speaker->roster mapping IN PLACE on the existing attributed parquet.

`Roster.resolve` is pure (no LLM, no re-transcription), so we can recompute the
`slug`/`is_a16z` columns from the already-stored raw `speaker` labels under the
corroboration-gated resolver — fixing the first-name-collision false positives
without re-running attribution. Reports every is_a16z flip, then re-exports the
canonical jsonl.

  python -m attribution.reresolve            # report only (dry run)
  python -m attribution.reresolve --write     # apply + re-export records
"""
import argparse

import pandas as pd

from .config import CORPUS, SEGMENTS_OUT, HOST_PRIOR
from .roster import Roster
from .route import route_post, CONVERSATIONAL
from .overnight_author import _present_slugs, _credited


def _corpus_rows():
    c = pd.read_parquet(CORPUS)[["object_id", "formats", "author_slugs"]]
    return {r["object_id"]: r for _, r in c.iterrows()}


def reresolve(write: bool) -> None:
    seg = pd.read_parquet(SEGMENTS_OUT)
    roster = Roster.load()
    crows = _corpus_rows()
    new_slug = seg["slug"].copy()
    new_a16z = seg["is_a16z"].copy()

    for pid, grp in seg.groupby("post_id"):
        crow = crows.get(pid)
        if crow is None:
            continue
        mode = route_post(crow).mode
        prior = HOST_PRIOR["podcast"] if mode == CONVERSATIONAL else HOST_PRIOR["research-seminar"]
        fake_segs = [{"speaker": s} for s in grp["speaker"]]
        present = _present_slugs(fake_segs, roster, prior, _credited(crow))
        for idx, spk in zip(grp.index, grp["speaker"]):
            m = roster.resolve(spk, prefer=prior, allow=present)
            new_slug.at[idx] = m.slug
            new_a16z.at[idx] = m.is_a16z

    flipped = seg[(seg["is_a16z"] != new_a16z)].copy()
    flipped["new_slug"] = new_slug[flipped.index]
    flipped["new_a16z"] = new_a16z[flipped.index]
    demoted = flipped[(seg["is_a16z"] == True) & (~new_a16z[flipped.index])]   # noqa: E712
    promoted = flipped[(seg["is_a16z"] == False) & (new_a16z[flipped.index])]  # noqa: E712

    print(f"segments: {len(seg)} | is_a16z flips: {len(flipped)} "
          f"(demoted {len(demoted)}, promoted {len(promoted)})")
    kept_dem = demoted[demoted["kept"] == True]  # noqa: E712
    print(f"\n=== KEPT a16z turns being DEMOTED ({len(kept_dem)}) — by old slug ===")
    for slug, c in kept_dem.groupby("slug").size().sort_values(ascending=False).items():
        # how many KEPT turns this person keeps after the fix
        remaining = int(((new_slug == slug) & (new_a16z) & (seg["kept"] == True)).sum())
        print(f"  {slug}: -{c} kept  (remaining kept after fix: {remaining})")

    if not write:
        print("\n(dry run — pass --write to apply and re-export records)")
        return

    seg["slug"] = new_slug
    seg["is_a16z"] = new_a16z
    tmp = SEGMENTS_OUT.with_suffix(".parquet.tmp")
    seg.to_parquet(tmp, index=False)
    tmp.rename(SEGMENTS_OUT)
    from .records import export as export_records
    n = export_records(seg)
    print(f"\napplied. re-exported attributed_transcripts.jsonl ({n} records).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    reresolve(ap.parse_args().write)


if __name__ == "__main__":
    main()
