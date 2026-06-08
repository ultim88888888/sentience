"""Deterministically derive lifecycle/delta features across periods. The signal
lives in the CHANGES, not the levels (spec stage 4). No LLM here."""
from __future__ import annotations
import pandas as pd

from signals.schema import PeriodSignal
from signals.config import STANCE_SIGN

PANEL_COLUMNS = [
    "as_of", "item", "item_type", "parent_sector", "stance", "conviction",
    "horizon", "lifecycle_state", "delta_stance", "delta_conviction", "age",
]


def derive_panel(periods: list[PeriodSignal]) -> pd.DataFrame:
    """Walk periods in chronological order, tracking per-item prior state to
    classify lifecycle and compute deltas. Emits one row per (period, item)
    that is present, plus a synthetic EXITED row the period an item drops out."""
    periods = sorted(periods, key=lambda p: p.as_of)
    rows: list[dict] = []
    # prior[item] = (stance, conviction, age) for items present in the previous period
    prior: dict[str, tuple[str, int, int]] = {}

    for p in periods:
        present = {it.item: it for it in p.items}

        # Present items: NEW / SUSTAINED / FLIPPED
        for item, it in present.items():
            if item in prior:
                prev_stance, prev_conv, prev_age = prior[item]
                if STANCE_SIGN[it.stance] != STANCE_SIGN[prev_stance] and \
                   STANCE_SIGN[it.stance] * STANCE_SIGN[prev_stance] < 0:
                    state = "FLIPPED"          # sign reversal (bullish<->bearish)
                else:
                    state = "SUSTAINED"
                age = prev_age + 1
                d_stance = STANCE_SIGN[it.stance] - STANCE_SIGN[prev_stance]
                d_conv = it.conviction - prev_conv
            else:
                state, age, d_stance, d_conv = "NEW", 1, 0, 0
            rows.append({
                "as_of": p.as_of, "item": item, "item_type": it.item_type,
                "parent_sector": it.parent_sector, "stance": it.stance,
                "conviction": it.conviction, "horizon": it.horizon,
                "lifecycle_state": state, "delta_stance": d_stance,
                "delta_conviction": d_conv, "age": age,
            })

        # Items present last period but gone now -> synthetic EXITED row
        for item, (prev_stance, prev_conv, _age) in prior.items():
            if item not in present:
                rows.append({
                    "as_of": p.as_of, "item": item, "item_type": "sector",
                    "parent_sector": None, "stance": "neutral", "conviction": 0,
                    "horizon": "structural", "lifecycle_state": "EXITED",
                    "delta_stance": 0 - STANCE_SIGN[prev_stance],
                    "delta_conviction": 0 - prev_conv, "age": 0,
                })

        # Advance prior to only currently-present items (so re-entry is NEW again)
        prior = {it.item: (it.stance, it.conviction,
                           next(r["age"] for r in rows
                                if r["as_of"] == p.as_of and r["item"] == it.item))
                 for it in p.items}

    return pd.DataFrame(rows, columns=PANEL_COLUMNS)
