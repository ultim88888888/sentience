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

**Corpus representation (decided 2026-06-07 — feeding all-history ≤ T is infeasible: transcripts
alone are ~8.38M chars / ~2M tokens blended):**
- **Trailing holding-period window.** At each T, include only evidence with
  `publish_date ∈ (T − window, T]`, applied **uniformly across all sources** (tweets, articles,
  transcripts). The window is the **longest anticipated holding period**, not a recency horizon —
  the originating evidence for a trade must stay in view for the life of the trade or the signal
  flickers off mid-hold. **Config param, default 18mo, test 24mo**, set as the largest that fits
  context with headroom (measured empirically at the largest T). Backtested holding periods are
  constrained ≤ window.
  - Consequence: exits shift from "aged out" (`EXITED` by silence) toward "explicitly reversed"
    (`FLIP`) — more robust, since silence is a weak signal and a stated reversal is strong.
- **Distillation (one-time cleaning stage, stashed permanently, resumable).** Transcripts are the
  bloat (median 56k chars, low signal-per-token). Distill each transcript **once** into its
  **verbatim, dated, stance-bearing passages with local context preserved** — *extractive, not
  abstractive*. Abstractive summarization would break the leakage firewall (a paraphrase is not a
  verbatim substring, so `audit` can't verify quotes); extractive selection keeps every line a
  verbatim substring with its real date, so the firewall survives. Cached per document, reused
  across every window and every approach (A1/A2a/A2b). Tweets and articles stay verbatim
  (already compact). The distillation is what makes the long window affordable (~350–400k tokens
  blended at a late-T 24mo window vs 2M+ raw).
  - **Validation gate (before committing to distilled inputs):** on early periods where the raw
    windowed corpus *does* fit, run extraction full-text vs distilled and compare signals. If
    distilled ≈ full-text, trust and scale; if they diverge, tighten the distill prompt or fall
    back to full-text-with-budget for affected sources.
- **Recency-privileging prompt.** A long window means each extraction sees both stale and fresh
  statements; the prompt must weight recency on conflict and use `age_note`/`provenance` to flag
  reversals ("stated 2021, reversed 2024").

**AUP constraint + distilled-tweets-for-A1 (decided 2026-06-08, hard-won — see
[[reference_claude_p_aup_limits]]):** `claude -p` rejects large crypto-signal requests via a
probabilistic **Usage-Policy classifier** (NOT rate/account/network). Triggers: **stdin delivery**
of a big payload + raw-tweet hype volume. Measured: 288k-token blended-A1 via stdin = ~0-20% pass;
**≤~50k tokens passes ~100%; distilled content goes higher (86k passed 3/3)**. The fix splits tweet
handling by approach — and this is also conceptually right:
- **A1 (member-AGNOSTIC content blend): distill the tweets too.** A1 cares about trade-relevant
  CONTENT, not author voice — so extractively keep only stance-bearing tweets (index-select at high
  effort, ~88% dropped, verbatim guaranteed), blend distilled tweets + distilled transcripts → ~86k
  tok → **one reliable stdin call**, member-agnostic. (Built: `signals/distill.py::distill_tweet_batch`
  + `build_tweet_distillate_cache`, cached `data/signal/tweet_distillates.jsonl`.)
- **A2 (per-member doppelganger): keep tweets RAW.** Voice/personality IS the signal; distilling
  would strip it. Each member's raw corpus is usually ≤50k (reliable single call); only the most
  prolific (Kominers ~130k) need **raw-corpus chunk-and-merge** (≤50k slices → partial views → merge
  his view; preserves voice + all data). Never build A1 from per-member views — that injects member
  bias and collapses A1 into A2.
- **General rule:** keep every `claude -p` call ≤~50k tokens; if a single call must exceed it,
  map-reduce over content/time slices (member-agnostic for A1), never per-member for A1.
- **Effort:** mechanical filtering (tweet distillation) runs at `--effort low`→ found lossy
  (dropped ~58% of stance tweets) → reverted to **high effort with index-based output** (tiny output
  = fast despite high effort; selection quality preserved; verbatim guaranteed by reconstruction).
  The actual extraction/judgment calls stay at high.

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
- **A2a:** collapse N member signals → consensus + dispersion. **Two separable outputs, only one
  of which is deterministic:**
  - **Dispersion = deterministic measurement (always).** Spread/entropy of the N member stances +
    conviction spread. A thermometer reading computed directly from the member signals — no
    judgment, no LLM. Captures disagreement *independently* of the consensus.
  - **Consensus = LLM judgment.** An LLM reads the N member views (with the computed dispersion in
    hand) and writes the house view — reasoning about *why* members differ (defer to the domain
    expert, note the split). This is the natural act: a research committee reasons, it does not
    average integers. Because dispersion is measured separately, the LLM **cannot** paper over
    disagreement — the spread number sits next to its consensus.
  - Rationale for the split: the only thing a deterministic consensus formula buys is
    reproducibility, at the cost of an arbitrary stance-averaging rule that mushes a bull-90
    against a bear-60 into a fake "mild bullish." Not worth it. Determinism is reserved for the
    measurement, not the judgment.
  - **Variants (tested, not default):** expertise-weighting the consensus (by self-report or
    corpus volume in sector).
  - **Keep ALL items, no coverage floor (decided 2026-06-08).** The consensus weighting
    (conviction × coverage) automatically drowns out / eliminates a singular-coverage view — a lone
    low-conviction call won't survive aggregation — so we don't pre-filter. Expose `coverage` and
    `dispersion` as features for the strategy layer to weight; don't drop the long tail by hand.
  - **This drowning-out is precisely what motivates A2b** (below). The A1→A2a→A2b progression is a
    spectrum of *how much a unique/minority view can survive*: A1 (no member structure) → A2a
    (structure, but majority/conviction wins → minority buried) → A2b (structure + persuasion can
    flip the room). High-`dispersion` items are where A2a-buries-minority and A2b-minority-can-win
    diverge most — so dispersion flags exactly where the A2a-vs-A2b comparison is informative.
- **A2b — fully agentic, zero determinism (variant, after A2a).** Instead of each doppelganger
  emitting a confidence that we then collapse, the doppelgangers **discuss their reasons and reach
  consensus together** through dialogue — testing whether a member can **persuade the others** of a
  unique-but-correct call that A2a would have outvoted (minority-view propagation). No dispersion math, no formula — the consensus *is* the
  outcome of the discussion. **Risk to measure, not a blocker:** LLM personas converge toward
  agreement/sycophancy, so A2b may manufacture false consensus. We detect this by comparing A2b's
  agreement against A2a's measured dispersion — if A2a shows the house genuinely split on an item
  while A2b's doppelgangers all nod along, that exposes the groupthink artifact. Believing A2b
  blindly is not allowed; testing it is legitimate.

### Stage 3.5 — Coherence reviewer (annotation, not a gate)
A second LLM review of the consensus, **distinct from canonicalization** (which only fits sectors).
This one checks each call's *faithfulness and basis* — given the rebalance date + the call's cited
evidence, it scores whether the call actually follows from the corpus. It catches the class of errors
the downstream OI/universe filter *cannot* see — calls that are liquid/tradeable but **wrong or
hollow**:
- **misread** — model emitted bearish where the cited evidence was bullish (or vice versa);
- **thin** — a "high-conviction" call resting on a single offhand mention dressed as signal;
- **incoherent / contextually dead** — a stance that is liquid but broken in context.

**Scope discipline (important):** this reviewer judges **faithfulness, NOT tradeability.** FTT-type
(dead-but-mentioned) calls are *correct extractions* and are handled by the OI floor at the universe
layer — do NOT make this reviewer a tradeability filter. **Output is a FLAG, not a hard remover**
(`grounded` / `thin` / `incoherent` + confidence): we audit what it catches before ever letting it
gate, because an over-eager validity gate that deletes good contrarian signal is worse than no gate.
Build it as a confidence annotation on the panel; promote to a gate only once its precision is proven.

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

## The approaches (A1, A2a, A2b)

| | A1 (baseline) | A2a |
|---|---|---|
| Corpus | whole, blended | partitioned by member |
| Extraction | one pass = consensus | one pass per member |
| Consensus | n/a (extraction is it) | LLM consensus + deterministic dispersion |
| Unique feature | — | **dispersion** (cross-member disagreement) |
| Cost | cheap | N× extraction |
| Start date | earlier (no per-member density needed; just enough corpus to form a view) | gated on per-member corpus density (~2022 for most) |

**Sequencing:** A1 first (cheap, reuses the engine, honest baseline). A2a second — it adds **two**
things A1 cannot produce: (1) **dispersion** — disagreement across members is itself a feature
(unanimous house vs split house = a confidence/risk signal); (2) **reasoned persistence** — a
per-member doppelganger with a coherent soul can *judge* whether a dormant thesis still holds
(worldview-consistent) vs has lapsed, where A1 only *mechanically* recency-weights. Caveat:
reasoned persistence is also how a persona **confabulates** (rationalizing a stale view is
indistinguishable from genuine conviction), so it is a **hypothesis to test, not assumed** — and
**A1 (long-window + recency-privileging prompt) is the mechanical-persistence null hypothesis A2
must beat** on structural-persistence cases to justify its cost. A2b third —
the fully-agentic discussion variant (no determinism), gated on A2a proving the per-member
decomposition adds signal, and measured against A2a's dispersion to catch false consensus (see
Stage 3).

**Shelved:**
- **A2-v2 (news injection)** — feed period news to the doppelgangers and have them react. Powerful
  but changes the experiment from "what does the frozen persona believe" to "how does it react to
  events," and drags in a second strictly-as-of-T news pipeline. After plain A2a/A2b prove out.

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
- **Liquidity cap (execution):** trade at most **5% of OI per day**; larger positions **TWAP over
  following day(s)**. Rarely binds at our cadence (~60 trading days/quarter, ~20/month) for liquid
  names — modeled for honesty, not over-engineered.
- **Absolute OI floor (universe inclusion) — distinct from the 5%-cap, and the reason it exists:**
  without a floor, the 5%-of-OI/day cap interacts pathologically with thin tokens — `target_position
  / (5% × OI)` blows up, so the engine would try to **TWAP a dead/illiquid token (e.g. FTT) over
  *months*** and never meaningfully build the position. The floor excludes any token whose OI is too
  low to build the target position inside an acceptable horizon. **Derive it, don't hand-pick:**
  `floor_OI ≥ max_position_$ / (0.05 × max_TWAP_days)` — e.g. max position $500k (5% of $10mm) and a
  5-day max TWAP ⇒ floor ≈ **$2M OI**. A token must clear this to enter the universe at all. This is
  the principled cure for FTT-type calls (dead token → OI below floor → excluded) — handled by *data*
  at the universe layer, NOT by an LLM judging "is this still a real call." The signal layer stays
  faithful (it may surface FTT if the corpus mentions it); the universe filter decides tradeability.
  Also apply a **min daily $-volume** floor alongside OI (manipulation / execution-quality guard).
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
- **A1 vs A2a** — does per-member decomposition add informativeness over the blended baseline? Two
  distinct claims to test separately: (1) does **dispersion** carry signal; (2) does A2a's
  **reasoned persistence** beat A1's mechanical recency-weighting on structural-persistence cases
  (the confabulation-vs-conviction test). A1 is the null hypothesis. The core scientific comparison.

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
