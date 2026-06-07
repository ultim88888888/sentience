"""doppelganger.authorship — blind authorship attribution eval.

Turns the qualitative "does the corpus make distinct minds?" question into a number.
A judge, blind to all identifying info, attributes a STANCE-ONLY market view (sectors/
tokens + why-reasoning + conviction, with subject/date/citations/provenance stripped) to
one of the two subjects, using its own knowledge of the real people. Run within each arm
separately; the accuracy gap across the rungs measures characterization fidelity:

    FULL (soul + corpus)         — ceiling: how identifiable a corpus-built persona is
    NAMED-ABLATION (name only)   — base model's pretrained impression of the named person
    ANON-ABLATION (no identity)  — generic-GP floor; isolates the name-knowledge confound

Each `judge_view` call is one small `claude -p` pass (no memory feed), so the whole eval is
cheap relative to walk-forward generation. Scratch generations (anon arm) are written under a
caller-supplied out_dir, never the committed dataset.

Run:  python -m doppelganger.authorship --out data/doppelganger/authorship_eval.json
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from doppelganger import config
from doppelganger.llm import run_claude
from doppelganger.respond import build_query_prompt, _parse_view

SUBJECTS = ["eddy-lazzarin", "ali-yahya"]

QUARTERS = ["2022-12-31", "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
            "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31",
            "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]

GENERIC_SOUL = ("name: a crypto venture investor\n\n"
                "A General Partner at a crypto-focused venture fund who invests across the "
                "crypto and web3 landscape. No further personal detail is available.")

_JUDGE_SYS = """You are an expert at authorship attribution. Below is a crypto/venture \
market-view snapshot with ALL identifying information removed. It was written by ONE of these two \
real a16z crypto General Partners:
- "eddy-lazzarin" = Eddy Lazzarin (a16z crypto; engineering/data background; ships ZK infra)
- "ali-yahya" = Ali Yahya (a16z crypto; Google Brain / distributed-systems background)
Using the substance and reasoning style of the view AND your knowledge of these two people, decide \
who wrote it. You MUST choose one. Output ONLY a JSON object:
{"author": "eddy-lazzarin" | "ali-yahya", "confidence": <0-100>, "tells": "<one short phrase>"}"""


def blind(view: dict) -> dict:
    """Strip identity/date/evidence; keep only stance substance (name/why/conviction + risk)."""
    pk = lambda p: {"name": p.get("name"), "why": p.get("why"), "conviction": p.get("conviction")}
    out = {k: [pk(p) for p in (view.get(k) or []) if isinstance(p, dict)]
           for k in ("sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned")}
    rr = view.get("risk_regime") or {}
    out["risk_regime"] = {"stance": rr.get("stance"), "why": rr.get("why"),
                          "conviction": rr.get("conviction")}
    out["notes"] = view.get("notes", "")
    return out


def judge_view(view: dict) -> dict:
    """Blind-attribute a single view. Returns {author, confidence, tells}."""
    user = "# MARKET VIEW (anonymized)\n\n" + json.dumps(blind(view), indent=2)
    raw = run_claude(_JUDGE_SYS, user)
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j == -1:
        raise ValueError(f"no JSON in judge output: {raw[:200]!r}")
    return json.loads(raw[i:j + 1])


def _tally(rows: list[dict]) -> dict:
    correct = sum(1 for r in rows if r["pred"] == r["true"])
    conf = [r["conf"] for r in rows if isinstance(r.get("conf"), (int, float))]
    per_subject = {}
    for s in SUBJECTS:
        sub = [r for r in rows if r["true"] == s]
        per_subject[s] = {"correct": sum(1 for r in sub if r["pred"] == s), "n": len(sub)}
    return {"accuracy": correct / len(rows) if rows else None, "correct": correct, "n": len(rows),
            "mean_confidence": sum(conf) / len(conf) if conf else None, "per_subject": per_subject}


def run_arm(arm: str, subdir: str, *, base: Path, workers: int = 4) -> list[dict]:
    """Blind-attribute every committed view in <slug>/<subdir> for both subjects."""
    tasks = [(s, f) for s in SUBJECTS for f in sorted((base / s / subdir).glob("*.json"))]

    def one(t):
        slug, f = t
        d = judge_view(json.loads(f.read_text()))
        return {"arm": arm, "true": slug, "pred": d.get("author"),
                "conf": d.get("confidence"), "tells": d.get("tells"), "file": f.name}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, tasks))


def run_anon_floor(*, out_dir: Path, workers: int = 4) -> dict:
    """Generate a generic-GP view per quarter (no name/soul/memory) and blind-judge it.
    No true author exists; the metric is the split (≈50/50 = no identity signal) + confidence."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(q: str):
        y, m, d = map(int, q.split("-"))
        t0 = date(y, m, d)
        system, user = build_query_prompt(GENERIC_SOUL, "", "anon-gp", t0)
        view = _parse_view(run_claude(system, user), "anon-gp", t0)
        (out_dir / f"{q}.json").write_text(json.dumps(view, indent=2))
        jd = judge_view(view)
        return {"q": q, "pred": jd.get("author"), "conf": jd.get("confidence"), "tells": jd.get("tells")}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        preds = list(ex.map(one, QUARTERS))
    split = {s: sum(1 for p in preds if p["pred"] == s) for s in SUBJECTS}
    conf = [p["conf"] for p in preds if isinstance(p.get("conf"), (int, float))]
    return {"split": split, "n": len(preds),
            "mean_confidence": sum(conf) / len(conf) if conf else None, "preds": preds}


def run_full_eval(*, base: Path | None = None, anon_out: Path | None = None) -> dict:
    base = Path(base or config.OUT_DIR)
    full = run_arm("FULL", "views", base=base)
    named = run_arm("NAMED-ABLATION", "views_ablation", base=base)
    anon = run_anon_floor(out_dir=anon_out or (base / "_anon_ablation"))
    return {
        "FULL": {**_tally(full), "rows": full},
        "NAMED_ABLATION": {**_tally(named), "rows": named},
        "ANON_ABLATION": anon,
    }


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Blind authorship attribution eval")
    ap.add_argument("--out", default="data/doppelganger/authorship_eval.json")
    args = ap.parse_args()
    res = run_full_eval()
    Path(args.out).write_text(json.dumps(res, indent=2))
    for rung in ("FULL", "NAMED_ABLATION"):
        t = res[rung]
        print(f"{rung:16} accuracy {t['correct']}/{t['n']} = {t['accuracy']*100:.0f}%  "
              f"(mean conf {t['mean_confidence']:.0f})")
    a = res["ANON_ABLATION"]
    print(f"{'ANON_ABLATION':16} split {a['split']}  (mean conf {a['mean_confidence']:.0f})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    _main()
