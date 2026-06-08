"""Immutable signal schema for one extraction period. Extracted fields only;
derived (lifecycle/delta) fields live in signals/panel.py."""
from __future__ import annotations
from dataclasses import dataclass, field

VALID_STANCE = {"bullish", "neutral", "bearish"}
VALID_ITEM_TYPE = {"sector", "token"}
VALID_HORIZON = {"tactical", "structural"}
VALID_PROVENANCE = {"grounded", "persisted", "extrapolated"}
VALID_RISK = {"risk_on", "risk_off", "neutral", "no_view"}


def _clamp(n: int) -> int:
    return max(0, min(100, int(n)))


@dataclass(frozen=True)
class Citation:
    date: str   # ISO YYYY-MM-DD
    quote: str

    def to_dict(self) -> dict:
        return {"date": self.date, "quote": self.quote}

    @classmethod
    def from_dict(cls, d: dict) -> "Citation":
        return cls(date=d["date"], quote=d["quote"])


@dataclass(frozen=True)
class SignalItem:
    item: str                    # canonical id post-canonicalization; raw name before
    item_type: str               # "sector" | "token"
    parent_sector: str | None    # for tokens
    stance: str                  # "bullish" | "neutral" | "bearish"
    conviction: int              # 0-100 (intensity, not probability)
    horizon: str                 # "tactical" | "structural"
    rationale: str
    provenance: str              # "grounded" | "persisted" | "extrapolated"
    age_note: str | None
    citations: tuple[Citation, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.stance not in VALID_STANCE:
            raise ValueError(f"bad stance: {self.stance}")
        if self.item_type not in VALID_ITEM_TYPE:
            raise ValueError(f"bad item_type: {self.item_type}")
        if self.horizon not in VALID_HORIZON:
            raise ValueError(f"bad horizon: {self.horizon}")
        if self.provenance not in VALID_PROVENANCE:
            raise ValueError(f"bad provenance: {self.provenance}")
        object.__setattr__(self, "conviction", _clamp(self.conviction))

    def to_dict(self) -> dict:
        return {
            "item": self.item, "item_type": self.item_type,
            "parent_sector": self.parent_sector, "stance": self.stance,
            "conviction": self.conviction, "horizon": self.horizon,
            "rationale": self.rationale, "provenance": self.provenance,
            "age_note": self.age_note,
            "citations": [c.to_dict() for c in self.citations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SignalItem":
        return cls(
            item=d["item"], item_type=d["item_type"],
            parent_sector=d.get("parent_sector"), stance=d["stance"],
            conviction=d["conviction"], horizon=d["horizon"],
            rationale=d.get("rationale", ""), provenance=d["provenance"],
            age_note=d.get("age_note"),
            citations=tuple(Citation.from_dict(c) for c in d.get("citations", [])),
        )


@dataclass(frozen=True)
class RiskRegime:
    stance: str          # risk_on | risk_off | neutral | no_view
    conviction: int
    rationale: str
    provenance: str

    def __post_init__(self):
        if self.stance not in VALID_RISK:
            raise ValueError(f"bad risk stance: {self.stance}")
        object.__setattr__(self, "conviction", _clamp(self.conviction))

    def to_dict(self) -> dict:
        return {"stance": self.stance, "conviction": self.conviction,
                "rationale": self.rationale, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, d: dict) -> "RiskRegime":
        return cls(stance=d["stance"], conviction=d["conviction"],
                   rationale=d.get("rationale", ""), provenance=d["provenance"])


@dataclass(frozen=True)
class PeriodSignal:
    as_of: str           # ISO date T (rebalance date)
    approach: str        # "A1"
    items: tuple[SignalItem, ...]
    risk_regime: RiskRegime
    notes: str = ""

    def to_dict(self) -> dict:
        return {"as_of": self.as_of, "approach": self.approach,
                "items": [i.to_dict() for i in self.items],
                "risk_regime": self.risk_regime.to_dict(), "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "PeriodSignal":
        return cls(as_of=d["as_of"], approach=d["approach"],
                   items=tuple(SignalItem.from_dict(i) for i in d["items"]),
                   risk_regime=RiskRegime.from_dict(d["risk_regime"]),
                   notes=d.get("notes", ""))
