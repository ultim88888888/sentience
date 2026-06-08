# A1 Signal Panel Implementation Plan (Sprint 1a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the A1 signal pipeline — blended a16z corpus → trailing-window extraction → agentic canonicalization → a signal timeseries panel with lifecycle/delta features — as the research-artifact baseline.

**Architecture:** A new `signals/` module reuses the existing `doppelganger/` engine (LLM wrapper, ingestion adapters, citation audit) but reframes its output as a *consensus* signal over the whole blended corpus (no per-member soul — that is A2, a later sprint). At each rebalance date T, assemble a blended corpus from a trailing holding-period window (tweets/articles verbatim, transcripts from a one-time extractive distillate cache), extract a consensus market-view JSON (LLM, recency-privileging, free-form sector/token names), canonicalize each item to a seed-or-minted sector/token registry (LLM, semantic fit), audit citations for leakage, then deterministically derive lifecycle/delta features across periods into a `signal_panel.parquet`.

**Tech Stack:** Python 3.12, pandas, pyarrow, pytest (`asyncio_mode = auto`). LLM calls via `doppelganger/llm.py::run_claude` (`claude -p --model opus --effort high`), mocked in all tests. Spec: `docs/superpowers/specs/2026-06-07-corpus-signal-strategy-design.md`.

**Scope boundary:** This sprint builds A1 signal generation only (spec stages 1–4 for the A1 approach). Market data (1b) and strategy/backtest (1c) are separate later plans. No per-member extraction, no consensus collapse, no dispersion — those are A2.

---

## Reuse reference (read before starting)

These existing files are the patterns to mirror. Read them once up front:

- `doppelganger/llm.py` — `run_claude(system: str, user: str, *, workdir=None, timeout=900) -> str`. Subprocess wrapper, returns raw stdout. Raises `RuntimeError` on non-zero exit. **Mock target in every LLM test:** `patch("signal.<module>.run_claude", return_value=...)`.
- `doppelganger/respond.py` — `_parse_view(raw, subject, t0)`: extracts a JSON object from raw LLM output (handles ```json fences), coerces `conviction` to int[0,100], normalizes missing keys. The market-view JSON schema (`sectors_excited/sectors_concerned/tokens_excited/tokens_concerned` each with `name/why/conviction/provenance/age_note/citations[{date,quote}]`, plus `risk_regime{stance,conviction,why,provenance}`). Mirror this parser.
- `doppelganger/soul_audit.py` — `audit_answer(view, evidence_path, t0) -> AuditReport`; `AuditReport(checked, matched, hallucinated, leaked, ok)`. Substring-matches each citation quote against evidence (>85% threshold) and flags quotes dated after t0. Adapt to the flat `PeriodSignal` schema.
- `doppelganger/adapters/twitter.py::load_twitter(parquet_path, subject_slug) -> list[EvidenceItem]`, `adapters/research.py::load_research(articles_path, subject_slug)`. `EvidenceItem(id, subject, timestamp, source_type, text, speaker_slug, attribution_confidence, thread_id, context, context_missing, engagement)` from `doppelganger/schema.py`.
- `doppelganger/config.py` — `DATA_DIR`, `TWITTER_DIR`, `RESEARCH_ARTICLES`, `TRACKED_PEOPLE`. Data: `data/twitter/<handle>.parquet` (col `created_at` tz-aware UTC, `type`, `text`, `url`), `data/a16z_research/articles.parquet` (`permalink`, `post_date`, `extracted_text`, `author_slugs`), `data/a16z_research/transcripts.parquet` (`object_id`, `title`, `transcript`, `status`; join to articles via `object_id` for `post_date`), `data/tracked_people.yaml` (`people[].slug/.x_handle`).

**Worktree:** This plan executes in `.claude/worktrees/signal-strategy` (branch `signals/corpus-strategy`). All paths below are repo-relative.

---

## Task 1: Module scaffold, config, seed taxonomy

**Files:**
- Create: `signals/__init__.py`
- Create: `signals/config.py`
- Test: `tests/test_signals_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_config.py
from signals import config

def test_seed_sectors_are_lowercase_kebab_and_nonempty():
    assert config.SEED_SECTORS
    for s in config.SEED_SECTORS:
        assert s == s.lower()
        assert " " not in s  # kebab-case ids

def test_paths_point_under_data():
    assert config.SIGNAL_OUT_DIR.parts[-2:] == ("data", "signal")
    assert config.DISTILLATE_CACHE.name == "transcript_distillates.jsonl"

def test_default_window_is_holding_period_scale():
    assert config.DEFAULT_WINDOW_MONTHS == 18
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/__init__.py
```
```python
# signals/config.py
"""Configuration and seed taxonomy for the A1 signal pipeline."""
from pathlib import Path

DATA_DIR = Path("data")
TWITTER_DIR = DATA_DIR / "twitter"
RESEARCH_ARTICLES = DATA_DIR / "a16z_research" / "articles.parquet"
TRANSCRIPTS = DATA_DIR / "a16z_research" / "transcripts.parquet"
TRACKED_PEOPLE = DATA_DIR / "tracked_people.yaml"

SIGNAL_OUT_DIR = DATA_DIR / "signal"
DISTILLATE_CACHE = SIGNAL_OUT_DIR / "transcript_distillates.jsonl"
REGISTRY_PATH = SIGNAL_OUT_DIR / "registry.json"
PANEL_PATH = SIGNAL_OUT_DIR / "signal_panel.parquet"

DEFAULT_WINDOW_MONTHS = 18  # holding-period scale; test 24 (see spec stage 2)

# Seed sector taxonomy — precise, lowercase-kebab ids. The LLM fits to one of these
# or mints a new one (semantic judgment); see signals/canonicalize.py.
SEED_SECTORS = [
    "liquid-staking",
    "restaking",
    "l2-scaling",
    "zk",
    "pos-l1",
    "pow-l1",
    "modular-da",
    "defi",
    "stablecoins",
    "perp-dex",
    "gaming",
    "nft",
    "depin",
    "payments",
    "rwa",
    "ai-crypto",
    "infra-devtools",
    "privacy",
    "governance-dao",
    "consumer-social",
]

STANCE_SIGN = {"bullish": 1, "neutral": 0, "bearish": -1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/__init__.py signals/config.py tests/test_signals_config.py
git commit -m "feat(signal): module scaffold + seed sector taxonomy + config"
```

---

## Task 2: Signal schema dataclasses

**Files:**
- Create: `signals/schema.py`
- Test: `tests/test_signals_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_schema.py
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal

def _item(**kw):
    base = dict(item="zk", item_type="sector", parent_sector=None, stance="bullish",
               conviction=80, horizon="structural", rationale="r",
               provenance="grounded", age_note=None,
               citations=(Citation("2023-01-01", "validity proofs are the endgame"),))
    base.update(kw)
    return SignalItem(**base)

def test_period_roundtrips_through_dict():
    p = PeriodSignal(as_of="2023-03-31", approach="A1", items=(_item(),),
                     risk_regime=RiskRegime("risk_on", 70, "why", "grounded"),
                     notes="n")
    d = p.to_dict()
    p2 = PeriodSignal.from_dict(d)
    assert p2 == p
    assert d["items"][0]["citations"][0]["quote"] == "validity proofs are the endgame"

def test_conviction_is_clamped_on_construction():
    assert _item(conviction=150).conviction == 100
    assert _item(conviction=-5).conviction == 0

def test_invalid_stance_rejected():
    import pytest
    with pytest.raises(ValueError):
        _item(stance="mega-bull")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/schema.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/schema.py tests/test_signals_schema.py
git commit -m "feat(signal): immutable signal schema with validation + roundtrip"
```

---

## Task 3: Sector/token registry (deterministic bookkeeping)

**Files:**
- Create: `signals/registry.py`
- Test: `tests/test_signals_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_registry.py
from signals.registry import Registry, load_registry, save_registry
from signals import config

def test_seed_registry_contains_seed_sectors():
    reg = Registry.seed()
    assert "zk" in reg.sectors
    assert reg.tokens == []

def test_mint_appends_and_is_idempotent():
    reg = Registry.seed()
    n = len(reg.sectors)
    reg.mint_sector("intent-solvers")
    reg.mint_sector("intent-solvers")  # idempotent
    assert reg.sectors.count("intent-solvers") == 1
    assert len(reg.sectors) == n + 1

def test_mint_token_records_parent():
    reg = Registry.seed()
    reg.mint_token("HYPE", parent_sector="perp-dex")
    assert "HYPE" in reg.tokens
    assert reg.token_parent["HYPE"] == "perp-dex"

def test_save_load_roundtrip(tmp_path):
    reg = Registry.seed()
    reg.mint_sector("intent-solvers")
    reg.mint_token("HYPE", parent_sector="perp-dex")
    p = tmp_path / "registry.json"
    save_registry(reg, p)
    reg2 = load_registry(p)
    assert reg2.sectors == reg.sectors
    assert reg2.token_parent == reg.token_parent

def test_load_missing_returns_seed(tmp_path):
    reg = load_registry(tmp_path / "nope.json")
    assert "zk" in reg.sectors
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/registry.py
"""The canonical vocabulary: seed sectors + everything minted so far. Pure
bookkeeping — the semantic fit-or-mint judgment is in signals/canonicalize.py."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

from signals import config


@dataclass
class Registry:
    sectors: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    token_parent: dict[str, str] = field(default_factory=dict)

    @classmethod
    def seed(cls) -> "Registry":
        return cls(sectors=list(config.SEED_SECTORS), tokens=[], token_parent={})

    def mint_sector(self, name: str) -> None:
        if name not in self.sectors:
            self.sectors.append(name)

    def mint_token(self, ticker: str, *, parent_sector: str | None = None) -> None:
        if ticker not in self.tokens:
            self.tokens.append(ticker)
        if parent_sector:
            self.token_parent[ticker] = parent_sector

    def to_dict(self) -> dict:
        return {"sectors": self.sectors, "tokens": self.tokens,
                "token_parent": self.token_parent}


def save_registry(reg: Registry, path: Path | None = None) -> None:
    path = path or config.REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg.to_dict(), indent=2))


def load_registry(path: Path | None = None) -> Registry:
    path = path or config.REGISTRY_PATH
    if not Path(path).exists():
        return Registry.seed()
    d = json.loads(Path(path).read_text())
    return Registry(sectors=d.get("sectors", []), tokens=d.get("tokens", []),
                    token_parent=d.get("token_parent", {}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/registry.py tests/test_signals_registry.py
git commit -m "feat(signal): sector/token registry with seed + mint + persistence"
```

---

## Task 4: Panel lifecycle/delta derivation (deterministic core)

This is the deterministic spine — the tradeable signal lives in the *changes*. Pure functions over a chronological list of `PeriodSignal`, no LLM.

**Files:**
- Create: `signals/panel.py`
- Test: `tests/test_signals_panel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_panel.py
import pandas as pd
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal
from signals.panel import derive_panel

def _it(item, stance, conv=70, typ="sector", parent=None, horizon="structural"):
    return SignalItem(item=item, item_type=typ, parent_sector=parent, stance=stance,
                      conviction=conv, horizon=horizon, rationale="r",
                      provenance="grounded", age_note=None,
                      citations=(Citation("2023-01-01", "q"),))

def _period(date, items):
    return PeriodSignal(as_of=date, approach="A1", items=tuple(items),
                        risk_regime=RiskRegime("risk_on", 60, "w", "grounded"))

def test_new_then_sustained_then_flip_then_exit():
    periods = [
        _period("2023-03-31", [_it("zk", "bullish", 70)]),
        _period("2023-06-30", [_it("zk", "bullish", 85)]),     # sustained, conv up
        _period("2023-09-30", [_it("zk", "bearish", 60)]),     # flip
        _period("2023-12-31", []),                              # zk exits
    ]
    df = derive_panel(periods)
    zk = df[df["item"] == "zk"].sort_values("as_of").reset_index(drop=True)
    assert list(zk["lifecycle_state"]) == ["NEW", "SUSTAINED", "FLIPPED", "EXITED"]
    assert list(zk["age"]) == [1, 2, 3, 0]                     # age resets at exit
    assert zk.loc[1, "delta_conviction"] == 15                 # 85 - 70
    assert zk.loc[2, "delta_stance"] == -2                     # bullish(+1) -> bearish(-1)

def test_exit_row_is_synthetic_with_zero_conviction():
    periods = [
        _period("2023-03-31", [_it("defi", "bullish", 50)]),
        _period("2023-06-30", []),
    ]
    df = derive_panel(periods)
    exit_row = df[(df["item"] == "defi") & (df["as_of"] == "2023-06-30")].iloc[0]
    assert exit_row["lifecycle_state"] == "EXITED"
    assert exit_row["conviction"] == 0
    assert exit_row["stance"] == "neutral"

def test_token_parent_sector_carried_into_panel():
    periods = [_period("2023-03-31", [_it("HYPE", "bullish", typ="token", parent="perp-dex")])]
    df = derive_panel(periods)
    assert df.iloc[0]["parent_sector"] == "perp-dex"

def test_reentry_after_exit_is_new_again_with_age_reset():
    periods = [
        _period("2023-03-31", [_it("gaming", "bullish")]),
        _period("2023-06-30", []),                              # exit
        _period("2023-09-30", [_it("gaming", "bullish")]),      # re-enter
    ]
    df = derive_panel(periods)
    g = df[df["item"] == "gaming"].sort_values("as_of").reset_index(drop=True)
    assert list(g["lifecycle_state"]) == ["NEW", "EXITED", "NEW"]
    assert list(g["age"]) == [1, 0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.panel'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/panel.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_panel.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/panel.py tests/test_signals_panel.py
git commit -m "feat(signal): deterministic lifecycle/delta panel derivation"
```

---

## Task 5: Transcript distillation + cache (one-time cleaning stage)

Extractive, firewall-preserving distillation of bloated transcripts into verbatim dated passages. Run once, cached, resumable. LLM mocked in tests.

**Files:**
- Create: `signals/distill.py`
- Test: `tests/test_signals_distill.py`
- Fixtures: `tests/fixtures/signals/transcripts_small.parquet`, `tests/fixtures/signals/articles_small.parquet`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_distill.py
import json
import pandas as pd
from unittest.mock import patch
from signals.distill import distill_one, build_distillate_cache

FAKE_LLM = json.dumps({"passages": [
    {"date": "2023-02-10", "passage": "I think zk rollups are the endgame for scaling."},
    {"date": "2023-02-10", "passage": "Restaking introduces real systemic risk."},
]})

def test_distill_one_returns_verbatim_passages():
    with patch("signals.distill.run_claude", return_value=FAKE_LLM):
        out = distill_one(object_id="x1", title="T", transcript="...long...",
                          post_date="2023-02-10")
    assert len(out) == 2
    assert out[0]["passage"].startswith("I think zk rollups")
    assert out[0]["date"] == "2023-02-10"

def test_build_cache_is_resumable(tmp_path):
    tx = pd.DataFrame({"object_id": ["x1", "x2"], "title": ["A", "B"],
                       "transcript": ["t1", "t2"], "status": ["ok", "ok"]})
    arts = pd.DataFrame({"object_id_join": ["x1", "x2"],
                         "post_date": ["2023-01-01", "2023-02-01"]})
    tx_path = tmp_path / "tx.parquet"; tx.to_parquet(tx_path)
    cache = tmp_path / "distillates.jsonl"
    # pre-seed cache with x1 already done
    cache.write_text(json.dumps({"object_id": "x1", "passages": []}) + "\n")
    calls = []
    def fake(system, user, **kw):
        calls.append(user)
        return FAKE_LLM
    with patch("signals.distill.run_claude", side_effect=fake):
        build_distillate_cache(tx_path, arts, cache_path=cache,
                               post_dates={"x1": "2023-01-01", "x2": "2023-02-01"})
    assert len(calls) == 1  # only x2 distilled; x1 skipped (resume)
    lines = [json.loads(l) for l in cache.read_text().splitlines()]
    assert {l["object_id"] for l in lines} == {"x1", "x2"}

def test_skips_empty_transcripts(tmp_path):
    tx = pd.DataFrame({"object_id": ["x3"], "title": ["C"], "transcript": [""],
                       "status": ["no_captions"]})
    tx_path = tmp_path / "tx.parquet"; tx.to_parquet(tx_path)
    cache = tmp_path / "d.jsonl"
    with patch("signals.distill.run_claude", return_value=FAKE_LLM) as m:
        build_distillate_cache(tx_path, None, cache_path=cache, post_dates={"x3": "2023-01-01"})
    m.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_distill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.distill'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/distill.py
"""One-time extractive distillation of transcripts → verbatim, dated, stance-bearing
passages. EXTRACTIVE not abstractive: every passage must be a verbatim substring of
the source so the leakage audit (signals/audit.py) still works. Cached + resumable."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from doppelganger.llm import run_claude
from signals import config

_SYSTEM = """You extract VERBATIM stance-bearing passages from a transcript.

Rules:
- Output ONLY passages that express a view/stance on a crypto sector, token, or market
  regime (bullish/bearish/concerned/excited, a thesis, a prediction, a risk).
- Each passage MUST be copied VERBATIM from the transcript — do NOT paraphrase,
  summarize, or rewrite. Preserve enough surrounding context that the stance is
  interpretable on its own (what "it"/"that" refers to).
- Be CONSERVATIVE on dropping: when unsure whether something is a view, keep it.
  Be strict on verbatim: never alter wording.
- Skip pure filler, logistics, pleasantries, and off-topic chatter.

Output JSON only: {"passages": [{"date": "<YYYY-MM-DD>", "passage": "<verbatim text>"}]}
Use the provided publish date for every passage's date."""


def distill_one(*, object_id: str, title: str, transcript: str, post_date: str) -> list[dict]:
    user = (f"PUBLISH DATE: {post_date}\nTITLE: {title}\n\nTRANSCRIPT:\n{transcript}")
    raw = run_claude(_SYSTEM, user)
    obj = _extract_json(raw)
    passages = obj.get("passages", [])
    # Force the publish date (single-date doc) and keep only non-empty passages.
    return [{"date": post_date, "passage": p["passage"]}
            for p in passages if p.get("passage", "").strip()]


def build_distillate_cache(transcripts_path, articles, *, cache_path: Path | None = None,
                           post_dates: dict[str, str] | None = None) -> Path:
    """Distill each ok transcript once; append to a JSONL cache. Resumable: rows whose
    object_id already appears in the cache are skipped."""
    cache_path = Path(cache_path or config.DISTILLATE_CACHE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache_path.exists():
        done = {json.loads(l)["object_id"] for l in cache_path.read_text().splitlines() if l.strip()}

    tx = pd.read_parquet(transcripts_path)
    post_dates = post_dates or _join_post_dates(tx, articles)

    with cache_path.open("a") as fh:
        for _, row in tx.iterrows():
            oid = row["object_id"]
            if oid in done:
                continue
            if row.get("status") != "ok" or not str(row.get("transcript", "")).strip():
                continue
            passages = distill_one(object_id=oid, title=row.get("title", ""),
                                   transcript=row["transcript"],
                                   post_date=post_dates.get(oid, ""))
            fh.write(json.dumps({"object_id": oid, "passages": passages}) + "\n")
            fh.flush()
    return cache_path


def load_distillates(cache_path: Path | None = None) -> dict[str, list[dict]]:
    """object_id -> list of {date, passage}."""
    cache_path = Path(cache_path or config.DISTILLATE_CACHE)
    if not cache_path.exists():
        return {}
    out = {}
    for line in cache_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["object_id"]] = d["passages"]
    return out


def _join_post_dates(tx: pd.DataFrame, articles) -> dict[str, str]:
    if articles is None:
        return {}
    arts = articles if isinstance(articles, pd.DataFrame) else pd.read_parquet(articles)
    # articles join key may be object_id or object_id_join depending on source
    key = "object_id" if "object_id" in arts.columns else "object_id_join"
    return dict(zip(arts[key].astype(str), arts["post_date"].astype(str)))


def _extract_json(raw: str) -> dict:
    """Mirror doppelganger/respond.py: pull the first JSON object, tolerate ``` fences."""
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return {"passages": []}
    return json.loads(s[start:end + 1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_distill.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/distill.py tests/test_signals_distill.py
git commit -m "feat(signal): extractive transcript distillation + resumable cache"
```

---

## Task 6: Windowed blended-corpus assembly

Assemble the as-of-T blended corpus: all tracked people's tweets + all firm articles (verbatim) + distilled transcript passages, each filtered to the trailing window, formatted chronologically.

**Files:**
- Create: `signals/corpus.py`
- Test: `tests/test_signals_corpus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_corpus.py
from datetime import date
import pandas as pd
from signals.corpus import in_window, assemble_corpus

def test_in_window_respects_trailing_bound_and_t():
    t = date(2024, 6, 30)
    assert in_window("2024-01-01", t, 18)       # inside
    assert not in_window("2024-07-01", t, 18)   # after T (leakage)
    assert not in_window("2022-01-01", t, 18)   # before window start

def test_assemble_is_chronological_and_tagged(tmp_path, monkeypatch):
    tw = pd.DataFrame({
        "created_at": pd.to_datetime(["2024-05-01", "2024-03-01"], utc=True),
        "type": ["original", "original"], "text": ["late tweet", "early tweet"],
        "url": ["u1", "u2"],
    })
    tw_path = tmp_path / "eddy.parquet"; tw.to_parquet(tw_path)
    arts = pd.DataFrame({"post_date": ["2024-04-01"], "extracted_text": ["an article body"],
                         "permalink": ["p1"], "object_id": ["o1"]})
    text = assemble_corpus(
        t=date(2024, 6, 30), window_months=18,
        twitter_paths=[tw_path], articles=arts, distillates={},
    )
    # chronological: early tweet (Mar) before article (Apr) before late tweet (May)
    assert text.index("early tweet") < text.index("an article body") < text.index("late tweet")
    assert "(x)" in text and "(research)" in text

def test_distillate_passages_included_in_window(tmp_path):
    arts = pd.DataFrame({"post_date": ["2024-04-01"], "extracted_text": ["body"],
                         "permalink": ["p1"], "object_id": ["o1"]})
    distillates = {"o1": [{"date": "2024-04-01", "passage": "zk is the endgame"}]}
    text = assemble_corpus(t=date(2024, 6, 30), window_months=18,
                           twitter_paths=[], articles=arts, distillates=distillates)
    assert "zk is the endgame" in text
    assert "(transcript)" in text

def test_post_t_evidence_excluded(tmp_path):
    arts = pd.DataFrame({"post_date": ["2025-01-01"], "extracted_text": ["future body"],
                         "permalink": ["p1"], "object_id": ["o1"]})
    text = assemble_corpus(t=date(2024, 6, 30), window_months=18,
                           twitter_paths=[], articles=arts, distillates={})
    assert "future body" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.corpus'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/corpus.py
"""Assemble the as-of-T blended corpus over a trailing holding-period window.
Uniform window across sources (spec stage 2). Tweets + articles verbatim;
transcripts come from the distillate cache (signals/distill.py)."""
from __future__ import annotations
from datetime import date
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta


def in_window(d: str | date, t: date, window_months: int) -> bool:
    dd = pd.to_datetime(d).date() if not isinstance(d, date) else d
    start = t - relativedelta(months=window_months)
    return start < dd <= t


def assemble_corpus(*, t: date, window_months: int, twitter_paths: list[Path],
                    articles, distillates: dict[str, list[dict]]) -> str:
    """Return one chronological, source-tagged text block of all in-window evidence."""
    rows: list[tuple[str, str, str]] = []  # (iso_date, source_tag, text)

    # Tweets (all tracked people), verbatim, drop retweets
    for p in twitter_paths:
        tw = pd.read_parquet(p)
        tw = tw[tw["type"] != "retweet"]
        for _, r in tw.iterrows():
            d = r["created_at"].date()
            if in_window(d, t, window_months):
                rows.append((d.isoformat(), "x", str(r["text"]).strip()))

    # Firm research articles, verbatim extracted_text
    arts = articles if isinstance(articles, pd.DataFrame) else (
        pd.read_parquet(articles) if articles is not None else pd.DataFrame())
    for _, r in arts.iterrows():
        if not in_window(r["post_date"], t, window_months):
            continue
        body = str(r.get("extracted_text") or "").strip()
        if body:
            rows.append((pd.to_datetime(r["post_date"]).date().isoformat(), "research", body))

    # Distilled transcript passages (keyed by article object_id, dated by passage)
    for _, r in arts.iterrows():
        oid = str(r.get("object_id") or "")
        for passage in distillates.get(oid, []):
            if in_window(passage["date"], t, window_months):
                rows.append((passage["date"], "transcript", passage["passage"].strip()))

    rows.sort(key=lambda x: x[0])
    return "\n".join(f"[{d}] ({tag}) {txt}" for d, tag, txt in rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_corpus.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/corpus.py tests/test_signals_corpus.py
git commit -m "feat(signal): windowed blended-corpus assembly (uniform trailing window)"
```

---

## Task 7: A1 extraction (LLM prompt + parse)

Build the recency-privileging consensus prompt, call the LLM, parse into a raw (pre-canonicalization) `PeriodSignal`. LLM mocked in tests.

**Files:**
- Create: `signals/extract.py`
- Test: `tests/test_signals_extract.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_extract.py
import json
from datetime import date
from unittest.mock import patch
from signals.extract import parse_extraction, build_a1_prompt
from signals.schema import PeriodSignal

RAW = json.dumps({
    "sectors_excited": [{"name": "zk rollups", "why": "scaling endgame",
                         "conviction": 88, "horizon": "structural", "provenance": "grounded",
                         "age_note": None, "citations": [{"date": "2024-02-01", "quote": "zk is the endgame"}]}],
    "sectors_concerned": [{"name": "memecoins", "why": "froth", "conviction": 60,
                           "horizon": "tactical", "provenance": "grounded", "age_note": None,
                           "citations": [{"date": "2024-03-01", "quote": "memecoins are pure froth"}]}],
    "tokens_excited": [{"name": "HYPE", "why": "flow", "conviction": 70, "horizon": "tactical",
                        "parent_sector": "perp dex", "provenance": "grounded", "age_note": None,
                        "citations": []}],
    "tokens_concerned": [],
    "risk_regime": {"stance": "risk_on", "conviction": 65, "why": "liquidity", "provenance": "grounded"},
    "notes": "n",
})

def test_parse_maps_arrays_to_stance_and_clamps():
    p = parse_extraction(RAW, t=date(2024, 6, 30))
    assert isinstance(p, PeriodSignal)
    by = {i.item: i for i in p.items}
    assert by["zk rollups"].stance == "bullish"        # from sectors_excited
    assert by["memecoins"].stance == "bearish"         # from sectors_concerned
    assert by["zk rollups"].item_type == "sector"
    assert by["HYPE"].item_type == "token"
    assert by["HYPE"].parent_sector == "perp dex"      # raw, pre-canonicalization
    assert p.risk_regime.stance == "risk_on"
    assert p.as_of == "2024-06-30"

def test_parse_tolerates_fenced_json_and_missing_arrays():
    raw = "```json\n" + json.dumps({"sectors_excited": [], "risk_regime":
          {"stance": "neutral", "conviction": 50, "why": "", "provenance": "extrapolated"}}) + "\n```"
    p = parse_extraction(raw, t=date(2024, 6, 30))
    assert p.items == ()
    assert p.risk_regime.stance == "neutral"

def test_prompt_demands_recency_weighting_and_no_taxonomy():
    system, user = build_a1_prompt("[2024-01-01] (x) zk good", t=date(2024, 6, 30))
    assert "recent" in system.lower() or "recency" in system.lower()
    assert "2024-06-30" in system
    # must NOT leak the seed taxonomy into extraction (free-form naming)
    assert "perp-dex" not in system and "liquid-staking" not in system
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.extract'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/extract.py
"""A1 consensus extraction: read the blended as-of-T corpus, emit the a16z house
market view. Free-form sector/token names (no taxonomy — canonicalize.py fits them
later). Recency-privileging (long window contains stale + fresh statements)."""
from __future__ import annotations
import json
from datetime import date

from doppelganger.llm import run_claude
from signals.corpus import assemble_corpus
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal

_SYSTEM_TMPL = """You are analyzing the public corpus of the a16z crypto team as of {t}.
It is {t}. The future has not happened yet. Use ONLY the statements provided.

Produce the TEAM'S CONSENSUS market view — what the house is collectively excited about,
concerned about, and its overall risk posture — as of {t}.

CRITICAL — recency: the corpus spans up to {window} months. When statements conflict,
weight the MOST RECENT view. If an old stance was later reversed, report the current
stance and record the reversal in age_note (e.g. "bullish 2022, cooled 2024"). A view
stated once long ago and never restated is `persisted` (cite it, note its age).

Name sectors and tokens in your OWN words — do not force them into any fixed taxonomy.
For tokens, give the sector they belong to in parent_sector.

Every excited/concerned entry needs: name, why, conviction (0-100 intensity, NOT a
probability), horizon ("tactical" | "structural"), provenance ("grounded"=stated with a
dated verbatim quote | "persisted"=held but not recently restated | "extrapolated"=inferred),
age_note (or null), and citations (verbatim quotes <=25 words with ISO dates; required for
grounded/persisted).

Output JSON only:
{{"sectors_excited": [...], "sectors_concerned": [...], "tokens_excited": [...],
  "tokens_concerned": [...],
  "risk_regime": {{"stance": "risk_on|risk_off|neutral|no_view", "conviction": 0-100,
                   "why": "...", "provenance": "..."}},
  "notes": "..."}}"""


def build_a1_prompt(corpus_text: str, t: date, *, window_months: int = 18) -> tuple[str, str]:
    system = _SYSTEM_TMPL.format(t=t.isoformat(), window=window_months)
    return system, corpus_text


def extract_a1(t: date, *, window_months: int, twitter_paths, articles, distillates) -> PeriodSignal:
    corpus = assemble_corpus(t=t, window_months=window_months, twitter_paths=twitter_paths,
                             articles=articles, distillates=distillates)
    system, user = build_a1_prompt(corpus, t, window_months=window_months)
    raw = run_claude(system, user)
    return parse_extraction(raw, t=t)


_ARRAY_STANCE = {"sectors_excited": ("sector", "bullish"),
                 "sectors_concerned": ("sector", "bearish"),
                 "tokens_excited": ("token", "bullish"),
                 "tokens_concerned": ("token", "bearish")}


def parse_extraction(raw: str, t: date) -> PeriodSignal:
    obj = _extract_json(raw)
    items: list[SignalItem] = []
    for key, (item_type, stance) in _ARRAY_STANCE.items():
        for e in obj.get(key, []) or []:
            items.append(SignalItem(
                item=e["name"], item_type=item_type,
                parent_sector=e.get("parent_sector"), stance=stance,
                conviction=e.get("conviction", 50), horizon=e.get("horizon", "tactical"),
                rationale=e.get("why", ""), provenance=e.get("provenance", "extrapolated"),
                age_note=e.get("age_note"),
                citations=tuple(Citation(c["date"], c["quote"]) for c in e.get("citations", []) or []),
            ))
    rr = obj.get("risk_regime") or {"stance": "no_view", "conviction": 0, "why": "", "provenance": "extrapolated"}
    risk = RiskRegime(stance=rr.get("stance", "no_view"), conviction=rr.get("conviction", 0),
                      rationale=rr.get("why", ""), provenance=rr.get("provenance", "extrapolated"))
    return PeriodSignal(as_of=t.isoformat(), approach="A1", items=tuple(items),
                        risk_regime=risk, notes=obj.get("notes", ""))


def _extract_json(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(s[start:end + 1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_extract.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/extract.py tests/test_signals_extract.py
git commit -m "feat(signal): A1 recency-privileging consensus extraction + parser"
```

---

## Task 8: Agentic canonicalization (fit-or-mint)

Map each raw item's free-form name to the seed-or-minted registry by LLM semantic judgment; update the registry. LLM mocked in tests.

**Files:**
- Create: `signals/canonicalize.py`
- Test: `tests/test_signals_canonicalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_canonicalize.py
import json
from unittest.mock import patch
from signals.schema import Citation, SignalItem
from signals.registry import Registry
from signals.canonicalize import canonicalize_items

def _raw(name, typ="sector", parent=None):
    return SignalItem(item=name, item_type=typ, parent_sector=parent, stance="bullish",
                      conviction=70, horizon="structural", rationale="r",
                      provenance="grounded", age_note=None,
                      citations=(Citation("2024-01-01", "q"),))

def test_fits_existing_and_mints_new():
    raw_items = [_raw("zero-knowledge proofs"), _raw("intent-based solvers"),
                 _raw("HYPE", typ="token", parent="perp dex")]
    mapping = json.dumps({"mapping": [
        {"raw": "zero-knowledge proofs", "canonical": "zk", "item_type": "sector",
         "parent_sector": None, "is_new": False},
        {"raw": "intent-based solvers", "canonical": "intent-solvers", "item_type": "sector",
         "parent_sector": None, "is_new": True},
        {"raw": "HYPE", "canonical": "HYPE", "item_type": "token",
         "parent_sector": "perp-dex", "is_new": True},
    ]})
    reg = Registry.seed()
    with patch("signals.canonicalize.run_claude", return_value=mapping):
        items, reg2 = canonicalize_items(raw_items, reg)
    canon = {i.item for i in items}
    assert canon == {"zk", "intent-solvers", "HYPE"}
    assert "intent-solvers" in reg2.sectors          # minted
    assert "HYPE" in reg2.tokens
    assert reg2.token_parent["HYPE"] == "perp-dex"
    # token HYPE's parent_sector rewritten to canonical
    assert next(i for i in items if i.item == "HYPE").parent_sector == "perp-dex"

def test_unmapped_item_falls_back_to_slug_not_dropped():
    raw_items = [_raw("Some Weird Sector")]
    with patch("signals.canonicalize.run_claude", return_value=json.dumps({"mapping": []})):
        items, reg = canonicalize_items(raw_items, Registry.seed())
    assert len(items) == 1                            # never silently dropped
    assert items[0].item == "some-weird-sector"       # deterministic slug fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_canonicalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.canonicalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/canonicalize.py
"""Agentic fit-or-mint: map free-form item names to the registry by SEMANTIC
judgment (not string match). The LLM decides existing-vs-new; bookkeeping
(append mints, rewrite items to canonical ids) is deterministic. Items are NEVER
silently dropped — an unmapped item falls back to a deterministic slug."""
from __future__ import annotations
import json
import re
from dataclasses import replace

from doppelganger.llm import run_claude
from signals.schema import SignalItem
from signals.registry import Registry

_SYSTEM = """You normalize crypto sector/token names to a controlled registry by MEANING,
not string match. For each raw item, decide whether it is the SAME concept as an existing
registry entry (return that canonical id, is_new=false) or genuinely new (propose a new
lowercase-kebab id, is_new=true). "zero-knowledge"/"zk proofs"/"validity proofs" are all
the existing `zk`. Tokens keep their ticker as canonical; give parent_sector as a registry
sector id (existing or newly proposed).

Output JSON only:
{"mapping": [{"raw": "<verbatim raw name>", "canonical": "<id>", "item_type": "sector|token",
  "parent_sector": "<sector-id-or-null>", "is_new": true|false}]}"""


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def canonicalize_items(raw_items: list[SignalItem], registry: Registry
                       ) -> tuple[list[SignalItem], Registry]:
    if not raw_items:
        return [], registry
    payload = {
        "registry": {"sectors": registry.sectors, "tokens": registry.tokens},
        "items": [{"raw": i.item, "item_type": i.item_type, "parent_sector": i.parent_sector}
                  for i in raw_items],
    }
    raw = run_claude(_SYSTEM, json.dumps(payload, indent=2))
    mapping = {m["raw"]: m for m in _extract_json(raw).get("mapping", [])}

    out: list[SignalItem] = []
    for it in raw_items:
        m = mapping.get(it.item)
        if m:
            canonical = m["canonical"]
            parent = m.get("parent_sector")
            if it.item_type == "token":
                registry.mint_token(canonical, parent_sector=parent)
                if parent:
                    registry.mint_sector(parent)
            else:
                registry.mint_sector(canonical)
        else:
            # never drop: deterministic slug fallback
            canonical = _slug(it.item)
            parent = _slug(it.parent_sector) if it.parent_sector else None
            (registry.mint_token(canonical, parent_sector=parent)
             if it.item_type == "token" else registry.mint_sector(canonical))
        out.append(replace(it, item=canonical, parent_sector=parent))
    return out, registry


def _extract_json(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(s[start:end + 1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_canonicalize.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/canonicalize.py tests/test_signals_canonicalize.py
git commit -m "feat(signal): agentic fit-or-mint canonicalization with slug fallback"
```

---

## Task 9: Leakage audit (adapt soul_audit to flat schema)

Verify every citation quote is a verbatim substring of the in-window corpus and dated ≤ T. Pure deterministic check; reuses the substring logic from `doppelganger/soul_audit.py`.

**Files:**
- Create: `signals/audit.py`
- Test: `tests/test_signals_audit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_audit.py
from datetime import date
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal
from signals.audit import audit_period

def _period(citations):
    item = SignalItem(item="zk", item_type="sector", parent_sector=None, stance="bullish",
                      conviction=80, horizon="structural", rationale="r",
                      provenance="grounded", age_note=None, citations=tuple(citations))
    return PeriodSignal(as_of="2024-06-30", approach="A1", items=(item,),
                        risk_regime=RiskRegime("risk_on", 60, "w", "grounded"))

CORPUS = "[2024-02-01] (transcript) zk is the endgame for scaling and i am very bullish"

def test_grounded_citation_passes():
    rep = audit_period(_period([Citation("2024-02-01", "zk is the endgame for scaling")]),
                       CORPUS, t=date(2024, 6, 30))
    assert rep.ok
    assert rep.matched == 1 and not rep.hallucinated

def test_hallucinated_quote_flagged():
    rep = audit_period(_period([Citation("2024-02-01", "solana will flip ethereum by 2025")]),
                       CORPUS, t=date(2024, 6, 30))
    assert not rep.ok
    assert len(rep.hallucinated) == 1

def test_post_t_quote_flagged_as_leaked():
    rep = audit_period(_period([Citation("2025-01-01", "zk is the endgame for scaling")]),
                       CORPUS, t=date(2024, 6, 30))
    assert not rep.ok
    assert len(rep.leaked) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_audit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.audit'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/audit.py
"""Leakage firewall for a PeriodSignal: every citation quote must be a verbatim
substring of the in-window corpus and dated <= T. Adapts doppelganger/soul_audit.py
to the flat signal schema. The distillation being EXTRACTIVE is what lets this work."""
from __future__ import annotations
import difflib
from dataclasses import dataclass
from datetime import date

from signals.schema import PeriodSignal, Citation

_MATCH_THRESHOLD = 0.85


@dataclass(frozen=True)
class AuditReport:
    checked: int
    matched: int
    hallucinated: list[Citation]
    leaked: list[Citation]

    @property
    def ok(self) -> bool:
        return not self.hallucinated and not self.leaked


def _quote_in_corpus(quote: str, corpus_norm: str) -> bool:
    q = " ".join(quote.lower().split())
    if q in corpus_norm:
        return True
    # fuzzy fallback (mirror soul_audit): best window similarity
    return difflib.SequenceMatcher(None, q, corpus_norm).quick_ratio() >= _MATCH_THRESHOLD \
        and any(difflib.SequenceMatcher(None, q, corpus_norm[i:i + len(q) + 10]).ratio() >= _MATCH_THRESHOLD
                for i in range(0, max(1, len(corpus_norm) - len(q)), max(1, len(q) // 2)))


def _citations(period: PeriodSignal):
    for it in period.items:
        for c in it.citations:
            yield c


def audit_period(period: PeriodSignal, corpus_text: str, t: date) -> AuditReport:
    corpus_norm = " ".join(corpus_text.lower().split())
    checked = matched = 0
    hallucinated: list[Citation] = []
    leaked: list[Citation] = []
    for c in _citations(period):
        checked += 1
        # leakage: claimed date after T
        try:
            cdate = date.fromisoformat(c.date)
        except ValueError:
            cdate = None
        if cdate and cdate > t:
            leaked.append(c)
            continue
        if _quote_in_corpus(c.quote, corpus_norm):
            matched += 1
        else:
            hallucinated.append(c)
    return AuditReport(checked=checked, matched=matched,
                       hallucinated=hallucinated, leaked=leaked)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_audit.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/audit.py tests/test_signals_audit.py
git commit -m "feat(signal): leakage audit adapted to flat signal schema"
```

---

## Task 10: Orchestration, CLI, and the full-vs-distilled validation gate

Wire the stages into `build_panel`, expose a CLI (`distill` / `panel` / `validate`), and implement the validation gate that compares full-text vs distilled extraction on a tractable early slice (spec stage 2).

**Files:**
- Create: `signals/run.py`
- Test: `tests/test_signals_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals_run.py
from datetime import date
from unittest.mock import patch
import json
import pandas as pd
from signals.run import build_panel, rebalance_dates

def test_rebalance_dates_quarterly():
    ds = rebalance_dates(date(2023, 1, 1), date(2023, 12, 31), "quarterly")
    assert ds == [date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30), date(2023, 12, 31)]

def test_rebalance_dates_monthly_count():
    ds = rebalance_dates(date(2023, 1, 1), date(2023, 12, 31), "monthly")
    assert len(ds) == 12

def test_build_panel_end_to_end_with_mocked_llm(tmp_path):
    # one sector, two quarters: NEW then SUSTAINED
    extraction = json.dumps({
        "sectors_excited": [{"name": "zk rollups", "why": "w", "conviction": 80,
            "horizon": "structural", "provenance": "grounded", "age_note": None,
            "citations": [{"date": "2023-01-15", "quote": "zk good"}]}],
        "risk_regime": {"stance": "risk_on", "conviction": 60, "why": "w", "provenance": "grounded"},
    })
    canon = json.dumps({"mapping": [{"raw": "zk rollups", "canonical": "zk",
        "item_type": "sector", "parent_sector": None, "is_new": False}]})
    arts = pd.DataFrame({"post_date": ["2023-01-15"], "extracted_text": ["zk good"],
                         "permalink": ["p1"], "object_id": ["o1"]})

    def fake_llm(system, user, **kw):
        return canon if "registry" in user else extraction
    with patch("signals.extract.run_claude", side_effect=fake_llm), \
         patch("signals.canonicalize.run_claude", side_effect=fake_llm):
        df = build_panel(date(2023, 1, 1), date(2023, 6, 30), "quarterly",
                         window_months=18, twitter_paths=[], articles=arts,
                         distillates={}, out_dir=tmp_path)
    zk = df[df["item"] == "zk"].sort_values("as_of").reset_index(drop=True)
    assert list(zk["lifecycle_state"]) == ["NEW", "SUSTAINED"]
    assert (tmp_path / "signal_panel.parquet").exists()
    assert (tmp_path / "registry.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'signals.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/run.py
"""Orchestrate the A1 signal pipeline and expose a CLI.

  python -m signals.run distill                      # one-time transcript cleaning
  python -m signals.run panel --start 2022-12-31 --end 2026-03-31 --interval quarterly
  python -m signals.run validate --t 2023-03-31      # full-vs-distilled extraction gate
"""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from signals import config
from signals.registry import load_registry, save_registry
from signals.extract import extract_a1
from signals.canonicalize import canonicalize_items
from signals.audit import audit_period
from signals.corpus import assemble_corpus
from signals.panel import derive_panel
from signals.distill import build_distillate_cache, load_distillates


def _month_end(d: date) -> date:
    return (d.replace(day=1) + relativedelta(months=1)) - relativedelta(days=1)


def rebalance_dates(start: date, end: date, interval: str) -> list[date]:
    step = 3 if interval == "quarterly" else 1
    out, cur = [], _month_end(start)
    while cur <= end:
        out.append(cur)
        cur = _month_end(cur + relativedelta(months=step))
    return out


def build_panel(start: date, end: date, interval: str, *, window_months: int,
                twitter_paths, articles, distillates, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir or config.SIGNAL_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(out_dir / "registry.json")
    periods = []
    audits = []
    for t in rebalance_dates(start, end, interval):
        raw_period = extract_a1(t, window_months=window_months, twitter_paths=twitter_paths,
                                articles=articles, distillates=distillates)
        canon_items, registry = canonicalize_items(list(raw_period.items), registry)
        period = raw_period.__class__(as_of=raw_period.as_of, approach="A1",
                                      items=tuple(canon_items),
                                      risk_regime=raw_period.risk_regime, notes=raw_period.notes)
        corpus = assemble_corpus(t=t, window_months=window_months, twitter_paths=twitter_paths,
                                 articles=articles, distillates=distillates)
        rep = audit_period(period, corpus, t)
        audits.append({"as_of": t.isoformat(), "checked": rep.checked, "matched": rep.matched,
                       "hallucinated": len(rep.hallucinated), "leaked": len(rep.leaked)})
        (out_dir / "periods").mkdir(exist_ok=True)
        (out_dir / "periods" / f"{t.isoformat()}.json").write_text(json.dumps(period.to_dict(), indent=2))
        periods.append(period)

    save_registry(registry, out_dir / "registry.json")
    df = derive_panel(periods)
    df.to_parquet(out_dir / "signal_panel.parquet")
    (out_dir / "audit.json").write_text(json.dumps(audits, indent=2))
    return df


def validate_distillation(t: date, *, window_months: int, twitter_paths, articles,
                          distillates) -> dict:
    """Compare full-text vs distilled extraction on one tractable early date. Reports
    item-set overlap (Jaccard) so divergence is visible before scaling distillation."""
    full = extract_a1(t, window_months=window_months, twitter_paths=twitter_paths,
                      articles=articles, distillates={})            # transcripts excluded == full text path TBD
    dist = extract_a1(t, window_months=window_months, twitter_paths=twitter_paths,
                      articles=articles, distillates=distillates)
    a = {i.item for i in full.items}
    b = {i.item for i in dist.items}
    jac = len(a & b) / len(a | b) if (a | b) else 1.0
    return {"as_of": t.isoformat(), "full_items": sorted(a), "distilled_items": sorted(b),
            "jaccard": jac}


def _tracked_twitter_paths() -> list[Path]:
    import yaml
    people = yaml.safe_load(Path(config.TRACKED_PEOPLE).read_text())["people"]
    paths = []
    for p in people:
        h = p.get("x_handle")
        if h and (config.TWITTER_DIR / f"{h}.parquet").exists():
            paths.append(config.TWITTER_DIR / f"{h}.parquet")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("distill")
    pp = sub.add_parser("panel")
    pp.add_argument("--start", required=True); pp.add_argument("--end", required=True)
    pp.add_argument("--interval", default="quarterly", choices=["quarterly", "monthly"])
    pp.add_argument("--window-months", type=int, default=config.DEFAULT_WINDOW_MONTHS)
    vp = sub.add_parser("validate"); vp.add_argument("--t", required=True)
    vp.add_argument("--window-months", type=int, default=config.DEFAULT_WINDOW_MONTHS)
    args = ap.parse_args()

    if args.cmd == "distill":
        build_distillate_cache(config.TRANSCRIPTS, config.RESEARCH_ARTICLES)
        return
    arts = pd.read_parquet(config.RESEARCH_ARTICLES)
    dist = load_distillates()
    tw = _tracked_twitter_paths()
    if args.cmd == "panel":
        build_panel(date.fromisoformat(args.start), date.fromisoformat(args.end),
                    args.interval, window_months=args.window_months,
                    twitter_paths=tw, articles=arts, distillates=dist)
    elif args.cmd == "validate":
        out = validate_distillation(date.fromisoformat(args.t), window_months=args.window_months,
                                    twitter_paths=tw, articles=arts, distillates=dist)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

> **Note for the implementer:** `validate_distillation`'s "full-text" arm currently re-uses the distillate-excluded path as a stand-in. Before running the gate for real, add a `raw_transcripts=True` mode to `assemble_corpus` that injects raw (undistilled) transcript text for in-window docs, so the comparison is genuinely full-vs-distilled. This is a deliberate follow-up step, not a silent gap — flagged here so it is not missed. Wire it as Task 10b when picking up the validation gate.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals_run.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full signal test suite**

Run: `pytest tests/test_signals_*.py -v`
Expected: PASS (all signal tests green)

- [ ] **Step 6: Commit**

```bash
git add signals/run.py tests/test_signals_run.py
git commit -m "feat(signal): build_panel orchestration + CLI + validation gate"
```

---

## Task 11: README + STATUS handoff

**Files:**
- Create: `signals/README.md`
- Modify: `docs/STATUS.md`

- [ ] **Step 1: Write `signals/README.md`**

Document: the pipeline stages, the CLI (`distill` → `panel` → `validate`), the trailing-window + distillation design and *why* (link the spec), the `signal_panel.parquet` schema (the `PANEL_COLUMNS`), the registry/distillate caches, and the open follow-ups (validation-gate full-text arm = Task 10b; 1b market data; 1c strategy/backtest). State plainly that this is A1 only.

- [ ] **Step 2: Update `docs/STATUS.md`**

Add a section under active work: A1 signal panel built (Sprint 1a), branch `signals/corpus-strategy`, what it produces, what is deferred (validation-gate full-text arm, 1b, 1c), and that A2a/A2b are unbuilt.

- [ ] **Step 3: Commit**

```bash
git add signals/README.md docs/STATUS.md
git commit -m "docs(signal): README + STATUS handoff for A1 signal panel"
```

---

## Definition of done

- [ ] All `tests/test_signals_*.py` pass (run `pytest tests/test_signals_*.py -v`).
- [ ] `python -m signals.run distill` produces `data/signal/transcript_distillates.jsonl` (real LLM — run once, manually verified on a few rows that passages are verbatim substrings of their source transcripts).
- [ ] `python -m signals.run panel --start 2022-12-31 --end <recent> --interval quarterly` produces `signal_panel.parquet`, `registry.json`, `periods/*.json`, `audit.json`.
- [ ] `audit.json` shows zero `leaked` across all periods (the firewall holds). Any non-zero leak is a stop-and-investigate.
- [ ] Validation gate (Task 10b wired) run on ≥1 early date; full-vs-distilled Jaccard recorded in the findings. Decision logged: trust distillation or fall back.
- [ ] Token-count headroom checked at the largest T; final `window_months` chosen and recorded.

## Deferred to later sprints (not this plan)

- **Task 10b** — full-text arm of the validation gate (`raw_transcripts` mode in `assemble_corpus`).
- **1b** — Coinglass daily price/OI/funding panel + BTC common-factor beta model.
- **1c** — strategy layer (directional / regime-hedge / dispersion), walk-forward backtest with costs/funding/TWAP, pipeline informativeness + leakage eval, A1-vs-A2 comparison.
- **A2a / A2b** — per-member extraction, dispersion, consensus collapse, discussion variant.
