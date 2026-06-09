# A3 — Market-Aware Member Deliberation (design)

**Date:** 2026-06-09 (written autonomously per Jax's overnight brief; Jax asleep ~7h, pre-approved build)
**Branch:** `signal/corpus-strategy` (worktree `signal-strategy`)

## Thesis (Jax)
The member doppelgangers were built not to regurgitate stance but to **think with the member's framework**.
A2 only extracts their corpus stance. A3 gives each member a **lookahead-safe digest** of what happened
*during* the interval (key events + market performance, strictly ≤ eval date) and asks them to make calls
*reasoning like the member*. Hypothesis: reacting to in-interval context (esp. monthly) lifts signal
materially vs A2's static corpus stance.

## Two new artifacts + two new consensus variants
1. **Period digest** (per interval, quarterly AND monthly), lookahead-safe:
   - **Market-performance block** (deterministic, from Coinglass OHLCV): trailing returns 7D/1M/3M/6M/1Y/2Y
     per token and per sector basket, as of eval date T (all *backward*-looking ⇒ no lookahead). Plus simple
     vol/drawdown. This is the quantitative half.
   - **Events/news block** (GDELT 2.0 DOC API, `enddatetime ≤ T` ⇒ lookahead-safe by construction):
     crypto headlines in (T−interval, T], Sonnet-summarized into a dense bullet digest. CryptoCompare
     requires a key we don't have; GDELT is free, keyless, timestamped, covers 2017+.
   - Sonnet (haiku/sonnet, no deep thinking) builds the digest — aggregation/summary only.
2. **A3a** = market-aware per-member calls → mechanical consensus (like A2a).
   **A3b** = market-aware per-member calls → council deliberation (like A2b).

## A3 member-call prompt (the new reasoning step)
Input: member persona/framework (their A2 corpus view = stance+framework) + the period digest.
Output (same SignalItem schema PLUS):
- excited/concerned sectors & tokens with conviction, horizon, **reason** (mandatory, cited to digest/corpus).
- **risk appetite BY HORIZON**: short-term (≤1–3M), medium (3–12M), long (>12M), each {risk_on|neutral|risk_off}
  + reason. (Jax: a VC is structurally long-bullish; asking short-term separately may flip it — that's the
  hedge signal we actually want.)
- The prompt is hard-anchored: "It is T. The future has not happened. Everything in the digest is ≤ T."

## Lookahead — the existential control (Jax flagged twice)
1. **Construction guarantee (primary):** digest market block uses only prices ≤ T; news block uses only
   `published ≤ T` (GDELT enddatetime=T). Hard date filters, not prompt-trust.
2. **Reason audit (secondary):** a Sonnet pass reads every call's `reason` and flags any that references
   information that could only be known after T (a result, a later price move, a consequence). Flagged calls
   are dropped/quarantined before consensus. Auditing reasons is why reasons are mandatory.

## SCIENTIFIC RISK I must flag (cofounder duty)
Feeding trailing market performance risks turning A3 into a **momentum strategy wearing a sentiment mask**:
if the member sees "ZK +180% trailing 1Y" they may chase it, and if past returns predict forward returns
(momentum factor), A3's "edge" is just momentum, not corpus-sentiment alpha. **Mitigation/measurement:**
build a pure price-momentum baseline (rank sectors by trailing return, no LLM) and report A3's IC *vs* and
*orthogonalized to* momentum. If A3 only matches momentum, we've learned the digest adds nothing beyond it.
This is a measurement, not a blocker — built into the A3 eval.

## Universe expansion (Jax: "search Coinglass for tokens to fit each sector")
Classify the FULL `coins_markets()` liquid-perp list (not just signal-named) into sectors → deeper baskets
⇒ more sectors tradeable. NB: the truly un-tradeable "bearish sectors" (regulation, quantum-threat,
snark-security) are risk *narratives*, not asset classes — expansion won't (and shouldn't) make those
shortable. Reconciles the IC paradox: IC was computed only on tradeable baskets (empty→dropped), so it
already measured the tradeable, long-tilted universe.

## Execution order (value-if-interrupted)
1. A2b dynamic-knob sweep (deterministic, quick; Jax asked). 
2. Universe expansion (parallel subagent).
3. Digest builder + **coverage/accuracy validation gate** (Jax: only run A3 consensus if broad & accurate).
4. A3 member-call prompt + reason-audit.
5. A3a/A3b consensus code; kick off **quarterly** A3 first (14 periods, fast), monthly after validation.
   A3 runs concurrently with the still-grinding monthly A2.
6. When A3 data lands: IC significance + sweep + momentum-orthogonalization, same rigor as A2b.

## Reuse
consensus.py (A3a), council.py (A3b — same minority-propagation deliberation), extract.py schema,
ic_significance.py + sweep_a2b.py + backtest_compare.py (eval). New: signals/digest.py, signals/a3.py.
