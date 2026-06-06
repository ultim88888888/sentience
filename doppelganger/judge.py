"""doppelganger.judge — held-out prediction judge.

Compares a view@T against what the subject ACTUALLY said in (T, T+horizon],
scoring each claim confirmed/contradicted/absent. One claude -p call, cached.
"""

from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.llm import run_claude


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def post_t_evidence(slug: str, t0: date, horizon_months: int = 6, *,
                    evidence_path: Path | None = None) -> str:
    path = evidence_path or (config.OUT_DIR / slug / "evidence.parquet")
    ev = pd.read_parquet(path)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    end = _add_months(t0, horizon_months)
    d = ev["timestamp"].dt.date
    win = ev[(d > t0) & (d <= end)].sort_values("timestamp")
    return "\n".join(
        f"[{pd.Timestamp(r['timestamp']).date().isoformat()}] ({r['source_type']}) {r['text']}"
        for _, r in win.iterrows()
    )


_JUDGE_INSTRUCTIONS = """You are evaluating a market-view PREDICTION against what a person \
ACTUALLY said afterward. You are given (1) the person's predicted market view as of a date, and \
(2) their REAL statements in the months that followed.

For each distinct claim in the prediction (each sector/token they were excited or concerned about, \
and their risk stance), label it against ONLY the provided later statements:
- "confirmed": they expressed or acted on this view in the window.
- "contradicted": they expressed the opposite.
- "absent": they did not address it in the window.

Judge at the level of STANCE, not wording. Use ONLY the provided later statements — do NOT use \
anything you know about what happened after the window. Also list "missed_changes": stances the \
person NEWLY took or REVERSED in the window that the prediction did not anticipate.

Output ONLY a JSON object:
{"claims":[{"claim":"...","axis":"sectors_excited|sectors_concerned|tokens_excited|tokens_concerned|risk_regime","label":"confirmed|contradicted|absent"}],
 "n_confirmed":<int>,"n_contradicted":<int>,"n_absent":<int>,"missed_changes":["..."],"notes":"..."}"""


def _parse_json(raw: str) -> dict:
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"no JSON object in judge output: {raw[:200]!r}")
    return json.loads(raw[i:j + 1])


def judge_step(view: dict, post_t_text: str, subject_name: str, t0: date, *,
               judge_path: Path | None = None) -> dict:
    if judge_path is not None and Path(judge_path).exists():
        return json.loads(Path(judge_path).read_text())          # cached

    user = (f"# {subject_name}'s PREDICTED market view as of {t0.isoformat()}\n\n"
            f"{json.dumps(view, indent=2)}\n\n"
            f"# What {subject_name} ACTUALLY said afterward\n\n"
            f"{post_t_text or '(no statements in this window)'}")
    raw = run_claude(_JUDGE_INSTRUCTIONS, user)
    v = _parse_json(raw)

    c, k = int(v.get("n_confirmed", 0)), int(v.get("n_contradicted", 0))
    v["confirm_rate"] = (c / (c + k)) if (c + k) > 0 else None
    v.setdefault("missed_changes", [])
    v.setdefault("claims", [])

    if judge_path is not None:
        Path(judge_path).parent.mkdir(parents=True, exist_ok=True)
        Path(judge_path).write_text(json.dumps(v, indent=2))
    return v
