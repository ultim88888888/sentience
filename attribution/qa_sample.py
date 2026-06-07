"""Quality-pass sampler for attributed_transcripts.jsonl.

No LLM. Pulls a stratified sample of kept a16z segments — weighted toward the
long-tail speakers (few turns = highest mis-attribution risk) — and prints each
target turn WITH its surrounding dialogue (prior/next labelled turns) so the
speaker label can be judged against conversational flow by eye.

  python3 -m attribution.qa_sample            # default: tail-weighted sample
  python3 -m attribution.qa_sample --slug X   # all kept turns for one person
"""
import argparse
import collections
import json

from .config import DATA

JSONL = DATA / "attributed_transcripts.jsonl"
CTX = 2  # neighbour turns to show on each side


def _load():
    return [json.loads(l) for l in open(JSONL)]


def _counts(records):
    c = collections.Counter()
    for r in records:
        for s in r["segments"]:
            if s["is_a16z"] and s["kept"]:
                c[s["slug"]] += 1
    return c


def _show_turn(s, mark=False):
    flag = " <<<< TARGET" if mark else ""
    keep = "" if s["kept"] else " [DROPPED]"
    conf = f"{s['confidence']:.2f}" if s.get("confidence") is not None else "  - "
    spk = (s["slug"] or s.get("speaker") or "?")[:22]
    txt = " ".join(s["text"].split())
    if not mark and len(txt) > 180:
        txt = txt[:180] + "…"
    print(f"    [{conf}] {spk:22s}{keep}: {txt}{flag}")


def _print_context(rec, idx):
    segs = rec["segments"]
    lo, hi = max(0, idx - CTX), min(len(segs), idx + CTX + 1)
    print(f"\n  ── {rec['object_id']} · {rec['title'][:60]}")
    print(f"     participants: {', '.join(rec['a16z_participants'])}")
    for j in range(lo, hi):
        _show_turn(segs[j], mark=(j == idx))


def sample(records, per_tail=2, per_head=1, tail_max=12):
    counts = _counts(records)
    # index every kept a16z turn by slug
    by_slug = collections.defaultdict(list)
    for ri, r in enumerate(records):
        for si, s in enumerate(r["segments"]):
            if s["is_a16z"] and s["kept"]:
                by_slug[s["slug"]].append((ri, si))
    tail = [sl for sl, n in counts.items() if n <= tail_max]
    head = [sl for sl, n in counts.items() if n > tail_max]
    print(f"# QA sample — {len(counts)} a16z speakers "
          f"({len(tail)} tail ≤{tail_max} turns, {len(head)} head)\n")
    # deterministic spread: stride through each slug's occurrences
    def pick(slug, k):
        occ = by_slug[slug]
        if not occ or k <= 0:
            return []
        if len(occ) <= k:
            return occ
        step = len(occ) // k
        return [occ[i * step] for i in range(k)]

    print("=" * 70)
    print(f"TAIL SPEAKERS (≤{tail_max} kept turns — highest risk)")
    print("=" * 70)
    for slug in sorted(tail, key=lambda s: counts[s]):
        print(f"\n### {slug}  ({counts[slug]} kept turns)")
        for ri, si in pick(slug, per_tail):
            _print_context(records[ri], si)

    print("\n" + "=" * 70)
    print("HEAD SPEAKERS (spot-check)")
    print("=" * 70)
    for slug in sorted(head, key=lambda s: -counts[s]):
        print(f"\n### {slug}  ({counts[slug]} kept turns)")
        for ri, si in pick(slug, per_head):
            _print_context(records[ri], si)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--per-tail", type=int, default=2)
    ap.add_argument("--per-head", type=int, default=1)
    args = ap.parse_args()
    recs = _load()
    if args.slug:
        for r in recs:
            for si, s in enumerate(r["segments"]):
                if s["slug"] == args.slug and s["is_a16z"] and s["kept"]:
                    _print_context(r, si)
        return
    sample(recs, args.per_tail, args.per_head)


if __name__ == "__main__":
    main()
