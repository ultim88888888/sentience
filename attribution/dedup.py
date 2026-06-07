"""Detect republished/duplicate posts and pick a canonical per cluster.

a16z reposts the same recording under different titles/object_ids (a clip then the
full episode, a retitle, a re-release months later). For a training corpus that's
silent duplication — the same speech counted twice. We detect it by exact-segment-text
overlap: each post -> set of md5(normalized segment text) for substantive turns
(>=40 chars). Two posts duplicate when their containment (shared / min size) clears
a threshold. Containment, not Jaccard, because separate transcription runs of the
SAME audio diverge token-by-token, so symmetric similarity understates duplication;
a high fraction of byte-identical segments is a strong signal regardless.

Non-destructive: writes a manifest mapping each dropped duplicate -> canonical.
`records.export` stamps every record with `duplicate_of` (None for canonical/unique)
so the consumer filters `duplicate_of is None`. Nothing is deleted.

  python -m attribution.dedup            # report
  python -m attribution.dedup --write     # write manifest + re-export records
"""
import argparse
import hashlib
import json
import re

import pandas as pd

from .config import CORPUS, DATA, SEGMENTS_OUT

MANIFEST = DATA / "duplicates.json"
MIN_CONTAINMENT = 0.5
MIN_SHARED = 5
MIN_SEG_CHARS = 40


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(t).lower())).strip()


def _seghash(t: str):
    n = _norm(t)
    return hashlib.md5(n.encode()).hexdigest() if len(n) >= MIN_SEG_CHARS else None


def _post_signatures(seg: pd.DataFrame) -> dict[str, set]:
    sigs = {}
    for pid, grp in seg.groupby("post_id"):
        hs = {_seghash(t) for t in grp["text"]}
        hs.discard(None)
        if hs:
            sigs[pid] = hs
    return sigs


class _UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def find_clusters(seg: pd.DataFrame) -> list[list[str]]:
    sigs = _post_signatures(seg)
    pids = sorted(sigs)
    uf = _UF()
    for i, a in enumerate(pids):
        A = sigs[a]
        for b in pids[i + 1:]:
            B = sigs[b]
            inter = len(A & B)
            if inter >= MIN_SHARED and inter / min(len(A), len(B)) >= MIN_CONTAINMENT:
                uf.union(a, b)
    groups: dict[str, list[str]] = {}
    for p in pids:
        groups.setdefault(uf.find(p), []).append(p)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def _stats(seg: pd.DataFrame, corpus: pd.DataFrame):
    meta = corpus.set_index("object_id")
    out = {}
    for pid, grp in seg.groupby("post_id"):
        kept_a16z = int(((grp["is_a16z"] == True) & (grp["kept"] == True)).sum())  # noqa: E712
        chars = int(grp["text"].astype(str).str.len().sum())
        date = str(meta.loc[pid]["post_date"]) if pid in meta.index else ""
        title = str(meta.loc[pid]["title"]) if pid in meta.index else ""
        out[pid] = {"kept_a16z": kept_a16z, "chars": chars, "date": date, "title": title}
    return out


def _canonical(cluster: list[str], stats: dict) -> str:
    # richest training signal wins: most kept-a16z turns, then most chars,
    # then earliest date, then stable object_id.
    return sorted(cluster, key=lambda p: (-stats[p]["kept_a16z"], -stats[p]["chars"],
                                          stats[p]["date"], p))[0]


def build_manifest(seg: pd.DataFrame, corpus: pd.DataFrame) -> dict:
    clusters = find_clusters(seg)
    stats = _stats(seg, corpus)
    mapping, detail = {}, []
    for cl in clusters:
        canon = _canonical(cl, stats)
        for p in cl:
            if p != canon:
                mapping[p] = canon
        detail.append({"canonical": canon, "members": cl,
                       "posts": {p: stats[p] for p in cl}})
    return {"min_containment": MIN_CONTAINMENT, "duplicate_of": mapping,
            "clusters": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    seg = pd.read_parquet(SEGMENTS_OUT)
    corpus = pd.read_parquet(CORPUS)
    man = build_manifest(seg, corpus)
    print(f"{len(man['clusters'])} duplicate clusters | "
          f"{len(man['duplicate_of'])} posts to drop "
          f"(canonical kept per cluster).\n")
    for c in man["clusters"]:
        print(f"  canonical {c['canonical']}  '{c['posts'][c['canonical']]['title'][:48]}'")
        for p in c["members"]:
            s = c["posts"][p]
            tag = "KEEP " if p == c["canonical"] else "drop "
            print(f"    {tag}{p}: kept_a16z={s['kept_a16z']:3d} chars={s['chars']//1000:3d}k "
                  f"'{s['title'][:42]}'")
    if not args.write:
        print("\n(report only — pass --write to save manifest + re-export records)")
        return
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    from .records import export
    n = export()
    dropped = sum(1 for r in open(DATA / "attributed_transcripts.jsonl")
                  if json.loads(r).get("duplicate_of"))
    print(f"\nwrote {MANIFEST}. re-exported {n} records ({dropped} flagged duplicate_of).")


if __name__ == "__main__":
    main()
