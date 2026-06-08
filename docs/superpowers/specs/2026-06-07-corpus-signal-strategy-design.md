# Corpus → Signal → Strategy — Design

_Date: 2026-06-07 · Branch: `signal/corpus-strategy` · Status: approved design, pre-plan_

## Purpose

Turn a large timestamped corpus of crypto-research material (a16z research posts,
podcast/video transcripts, team-member Twitter, bios) into a **signal timeseries**, and
build a trading-strategy mechanism around it. This is a **research project / case study**:
the deliverable is the *pipeline* — unstructured corpus → structured features → consensus
signal → strategy → backtest — with every stage clean, inspectable, swappable, and strictly
point-in-time. P&L is a sanity check on whether the signal is harvestable, **not** the grade.

**The thing we are actually testing:** can we add structure to unstructured data and extract
forward-looking signal from it? Profitability is secondary; the science is *informativeness +
leakage-purity* of the signal.

## Non-goals

- Not chasing a deployable Sharpe. N is small; we characterize signal, we don't claim an edge.
- No leverage beyond the hedge dial. No per-token discretionary overrides. No grid-searching
  dozens of configs against a tiny sample. These explode the parameter space and invite
  self-deception.

## Core principles (carried from the doppelganger engine)

1. **Feed the model inputs, don't script its cognition.** LLM does the judgment; deterministic
   code does only bookkeeping.
2. **Strict point-in-time.** At rebalance date T, every input (corpus, prices, OI, funding,
   betas) uses only data with timestamp ≤ T. Forward returns realize *after* T.
3. **Grounded + cited extraction** as the anti-leakage enforcement: no citation, no signal.
4. **Extracted vs derived** fields kept hard-separated, so the panel stays auditable.
5. **All LLM calls: Opus, effort `high`, uniformly.** Uniform effort removes the max/high
   confound that contaminated the prior eval.

---

## The central threat: pretraining lookahead

The model already knows the future. Asked "as Eddy in Q1 2023, what excites you?", the weights
know FTX collapsed, which L2s won, the 2024 memecoin run, which bets died. That is hindsight
laundered through a persona, not corpus signal. The corpus time-gating (frozen soul, citation
firewall) does **not** touch the model's own memory.

**Mitigations:**
- Grounded-and-cited extraction (corpus refs required per signal).
- Systematic **leakage audit** (eval stage B) — does the as-of-T signal reference things not
  knowable at T?
- Probe evidence so far (2026-06-07): pre-FTX Eddy persona showed no awareness of FTX. One
  probe, encouraging, not a full audit.

We treat reported numbers as an **upper bound** until the leakage audit constrains residual
contamination.

---

## Pipeline architecture — six stages

Each stage is a swappable unit with a clean interface. Stages 1–3 reuse and reframe the
existing doppelganger extraction engine; stages 4–6 are new.

### Stage 1 — Corpus assembly (as-of-T)
For each rebalance date T, gather all corpus material with `publish_ts ≤ T`.
- **A1:** whole corpus, members blended.
- **A2a:** partitioned by member.
Reuses ingestion + time-gating from the walk-forward engine.

### Stage 2 — Feature extraction (LLM, free-form)
The LLM reads the as-of-T corpus and emits the **extracted signal schema** (below) **in its
own words** — it does NOT receive the sector taxonomy (avoids force-fit priming). Grounded and
cited to corpus refs.
- **A1:** one extraction over the blended corpus. The extraction *is* the consensus.
- **A2a:** one extraction *per member* → N member signals.

### Stage 3 — Canonicalization + consensus
**Canonicalization (agentic, shared by A1 and A2a):** a separate LLM pass takes each raw item +
rationale + the **current registry** (seed taxonomy + everything minted so far) and judges, by
**semantic fit not string match**, whether it maps to an existing sector/token or mints a new
one. Handles token tickers and `parent_sector` assignment too. The only deterministic part is
bookkeeping (appending minted entries).

Rationale for decoupling extraction from canonicalization:
- **Fidelity** — raw extraction stays uncolored by our bucket scheme.
- **Auditability** — we keep both the raw phrase and the fitted sector; bad mappings are visible.
- **Cheap iteration** — extraction is the expensive pass; revising the taxonomy re-runs only
  canonicalization against existing raw extractions.

**Consensus:**
- **A1:** none needed — extraction already is the consensus.
- **A2a:** collapse N member signals → consensus + dispersion.
  - **Baseline = deterministic collapse:** conviction-and-coverage-weighted aggregation. Keeps
    the dispersion math honest and the panel inspectable — which is the entire reason A2a exists
    over A1.
  - **Variants (tested, not default):** LLM-aggregator collapse (can reason about *why* members
    disagree, e.g. defer to the domain expert); expertise-weighting (by self-report or corpus
    volume in sector).

### Stage 4 — Signal panel (the deliverable)
Append each period's consensus into a timeseries panel (item × date). **Deterministically
derive** the change features here — never LLM-touched:
- `lifecycle_state` ∈ {NEW, SUSTAINED, EXITED, FLIPPED, ABSENT}
- `delta_stance`, `delta_conviction` vs prior period
- `age` — periods since entry (the *realized* horizon; lets us check whether "structural" calls
  actually persisted)

The lifecycle is the spine of the signal — the tradeable event is the *change*, not the level
(prior walk-forward found `missed_changes` was the live metric while `confirm_rate` saturated):
- **NEW** (unmentioned → bullish) = open
- **SUSTAINED** (stays in) = hold; horizon accumulates
- **EXITED** (drops out) = close
- **FLIPPED** (bullish → concern) = strongest event = reverse

### Stage 5 — Strategy layer
Consumes signal panel + point-in-time price/OI/funding panel → target positions. Plug-in
strategies sharing `signal_panel + price_panel + params → target_weights`. (Detailed below.)

### Stage 6 — Backtest + eval
Walk-forward execution with costs; two distinct evaluations (detailed below).

---

## Signal schema

### Extracted fields (LLM emits; per-member in A2a)

Per **item** (an item is a sector *or* a token):
- `item` — canonical id (e.g. `zk-proofs`, `HYPE`)
- `item_type` ∈ {sector, token}
- `parent_sector` — for tokens (enables intra-sector dispersion trades)
- `stance` ∈ {bullish, neutral, bearish/concerned}
- `conviction` — 0–100 (intensity, NOT calibrated probability)
- `horizon` ∈ {tactical, structural}
- `rationale` + `citations` — short text tied to corpus refs (object_ids / tweet ids).
  **Citations are the anti-leakage enforcement.**

Per **period**:
- `risk_regime` ∈ {on, off, neutral} + `risk_conviction` + cited rationale

### A2a-only, added at consensus
- `dispersion` — cross-member disagreement per item (stance entropy + conviction spread)
- `coverage` — how many members hold a view (1-of-N → thin signal, discounted)

### Derived fields (deterministic, in the panel)
`lifecycle_state`, `delta_stance`, `delta_conviction`, `age` (see Stage 4).

### Sector taxonomy — seed + mintable registry
- **Seed taxonomy:** precise, predefined (e.g. liquid staking, gaming, perp DEX, L2, PoS L1,
  PoW L1, …).
- The LLM **fits to an existing sector or mints a new one** by semantic judgment (e.g. "ZK"
  becomes its own sector when nothing fits).
- **Registry + resolver:** every proposed sector is matched against the seed list *and all
  previously-minted entries* before a new one is created; once registered, reused verbatim.
  Vocabulary **grows but does not drift** ("ZK" / "zero-knowledge" / "validity proofs" resolve
  to one entry). When something graduates out of an emerging bucket into its own sector, that is
  itself a signal.
- Tokens identified by ticker, still canonicalized.

---

## The two approaches

| | A1 (baseline) | A2a |
|---|---|---|
| Corpus | whole, blended | partitioned by member |
| Extraction | one pass = consensus | one pass per member |
| Consensus | n/a (extraction is it) | deterministic collapse (baseline) |
| Unique feature | — | **dispersion** (cross-member disagreement) |
| Cost | cheap | N× extraction |
| Start date | earlier (no per-member density needed; just enough corpus to form a view) | gated on per-member corpus density (~2022 for most) |

**Sequencing:** A1 first (cheap, reuses the engine, honest baseline). A2a second — its payoff is
**not** a better point-estimate consensus; it's that **disagreement across members is itself a
feature** (unanimous house vs split house = a confidence/risk signal A1 cannot see).

**Shelved:**
- **A2b (doppelgangers debate to consensus)** — LLM persona debates converge artificially toward
  agreement/sycophancy; would measure a groupthink artifact. Revisit only if A2a proves the
  decomposition adds signal.
- **A2-v2 (news injection)** — feed period news to the doppelgangers and have them react. Powerful
  but changes the experiment from "what does the frozen persona believe" to "how does it react to
  events," and drags in a second strictly-as-of-T news pipeline. After plain A2a proves out.

---

## Strategy layer

Shared interface: `signal_panel + price_panel + params → target_weights`.

### Strategies (sequenced by signal demand)

1. **Directional (build first).** An *item* (sector OR token) bet on/against directionally.
   Driven by Δ-stance/lifecycle, not raw level (NEW = open, FLIPPED = reverse, EXITED = close).
   **Bet sign is a parameter** — `momentum` (follow excitement) vs `fade` (excitement = local
   top). The data decides; we do not assume the sign.

2. **Regime hedge (build second).** Modulates net exposure via the risk signal. Three sources,
   tested **individually and combined:**
   - **Quant regime** — naive market-state from price (trend/vol filter). No LLM.
   - **Consensus concern** — `risk_regime` + aggregate concern from the panel.
   - **Combined.**
   Output is a net-exposure dial: full → reduced → neutral → (optionally) net-short.
   **No-hedge is one setting of the dial** — long-only is this strategy with the dial pinned.

3. **Intra-sector dispersion (build third).** Within a bullish sector, long the standout token,
   short the laggards (e.g. long HYPE / short ASTER, LIGHTER). Market-neutral by construction.
   Needs token-level resolution + a liquid short leg, so gated on richer signal + narrower
   universe → last.

### Hedging: beta neutrality
- **Fully hedged = fully beta-hedged**, not dollar-matched.
- **Common-factor beta to BTC:** estimate each item's beta to BTC (point-in-time, rolling),
  construct the book to **net-zero portfolio beta**. The L1/L2 example (long 1 L1 / short 0.7 L2)
  falls out as a special case; scales to any mix without pairwise sector betas.
- **Honest caveat:** beta-neutral is a *target* on an *estimated* beta. Crypto betas are unstable
  and jump at regime shifts. We achieve estimated net-zero; realized beta will drift. We **report
  realized portfolio beta**, not assume the hedge held.

### Execution / cost model
- **Capital:** $10mm AUM (fixed assumption).
- **Liquidity cap:** trade at most **5% of OI per day**; larger positions **TWAP over following
  day(s)**. Rarely binds at our cadence (~60 trading days/quarter, ~20/month) and the liquidity
  filter removes thin names — modeled for honesty, not over-engineered.
- **Funding** baked into returns (perp funding, point-in-time).
- **Fees + slippage** modeled. Quarterly/monthly rebalance = low turnover, small but present.
- **Venue:** Binance primary perp; Hyperliquid secondary (only for HYPE if traded). **All market
  data (price/OI/funding, both venues) pulled from Coinglass at daily granularity** and resampled
  to the rebalance interval — daily queries are fast, parquet storage cheap.
- **Universe filter (guardrail, non-negotiable):** only items with a point-in-time *liquid* market
  at T. Existence ≠ tradeable. Keeps the backtest honest.
- **Sizing:** conviction-weighted, capped. Dispersion (A2a) modulates — split house → down-weight.
- **Always-deployed (preference):** the book is never empty; net exposure can be ~0 when fully
  hedged ("always positioned," not "always net-long" — reconciles with risk-off). Adjust position
  size to make room for new positions.

These guardrails (universe, sizing, costs, beta target) are **fixed**, not parameters to sweep.

---

## Backtest + evaluation

Two distinct evaluations for two distinct deliverables. **Pipeline eval (B) is the headline;
the strategy backtest (A) is the demonstration that the signal is harvestable.** On a tiny
sample the backtest is illustrative; the informativeness measurement is the science.

### A. Strategy backtest (trading layer)
- **Engine:** walk-forward over the panel. At each T: read signal → beta-neutralize vs BTC →
  apply universe/liquidity filter → conviction-size → simulate fill (5%-OI TWAP) → carry funding
  + fees → mark forward to next rebalance. No lookahead at any step.
- **Benchmarks:** BTC buy-hold **and** equal-weight liquid-universe basket. Report **risk-adjusted**
  (Sharpe, max DD, **realized portfolio beta**) — not raw return. A long book beating BTC on raw
  return in a bull market proves nothing; beating on Sharpe, or making money beta-neutral in a
  drawdown, is the claim.
- **Configs pre-committed:** bet-sign × hedge-source × horizon-filter — a *handful*, named before
  results are seen. Everything else logged exploratory.

### B. Pipeline eval (research deliverable — headline)
- **Leakage audit** — systematic extension of the FTX probe: does as-of-T signal reference
  not-yet-knowable things? Cited-grounding enforcement + spot-checks. Makes the whole thing
  credible.
- **Signal informativeness** — does Δ-stance/lifecycle predict forward sector/token returns,
  *before* any strategy wrapper? Rank-correlation of signal vs forward return. Isolates "is there
  signal" from "did the strategy harvest it."
- **A1 vs A2a** — does per-member decomposition + dispersion add informativeness over the blended
  baseline? The core scientific comparison.

### Honesty rails
- **N is small** but larger than feared: data available 2020→2026 → **~24 quarterly / ~72 monthly**
  points. Monthly is worth the compute (72 supports rank-correlation; 24 is thin). Binding start
  constraint is **corpus density**, not price data — panel starts sparse (~2022 for member-gated
  A2a; A1 can start earlier once there's enough corpus to form a view) and densifies.
- **Pre-commit configs**; log the rest as exploratory. On this sample size, that discipline is the
  difference between a finding and self-deception.
- **Train/test:** extraction does not "train" (not a fitted model) — the panel is OOS by
  construction. Only **strategy parameter selection** needs a split. Crypto regime structure
  (2021 bull / 2022 bear / 2023 recovery / 2024–25 bull) breaks any static split, so:
  - **Walk-forward, expanding window** for strategy selection (choose config using only data ≤ T).
  - Plus a **final untouched holdout** (~last 6–9 months) never looked at during development — one
    clean OOS read at the end.
- **Both rebalance intervals** reported, to show results aren't an artifact of one cadence.

---

## Reuse map

- **Reused from `doppelganger/engine-design`:** ingestion, time-gating, soul/memory,
  per-subject extraction (`respond`), the walk-forward harness, `doppelganger/llm.py`
  (`claude -p --model opus --effort high`). Stages 1–3 are a *reframing* of this engine's output
  as a signal panel.
- **New:** signal panel + derived lifecycle/delta features (Stage 4), strategy layer (Stage 5),
  backtest + cost/beta/execution model and pipeline eval (Stage 6), Coinglass daily
  price/OI/funding panel, common-factor beta model, canonicalization registry/resolver.

## Open items for the plan (not design gaps — sequencing)

- Seed sector taxonomy: enumerate the starting list.
- Coinglass daily collection sprint (price/OI/funding for the universe) — parallelizable with
  corpus work.
- Pre-committed strategy config set: enumerate the handful before any backtest runs.
- Final-holdout window: fix the exact dates and quarantine them.
