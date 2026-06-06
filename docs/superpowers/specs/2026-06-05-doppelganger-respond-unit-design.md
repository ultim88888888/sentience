# Corpus Doppelganger — Unit ❹ Doppelganger (Respond) — Design Spec

_Design spec · 2026-06-05 · status: draft-pending-review_

## 0. One-paragraph summary

The **doppelganger** is the reasoner that combines the frozen soul card + the time-gated memory feed to answer a market-view query **as the subject, in the present tense, at date T**. One Opus pass (`claude -p`, no API cost) reads the soul (who he is) + all ≤T evidence (what he's said) + the query, and emits a **structured JSON market view** — sectors and tokens he's excited/concerned about, a risk-regime stance, the "why" for each — every claim carrying a provenance tag (`grounded`/`persisted`/`extrapolated`) and dated citations. Optionality and abstention are first-class: any axis may be empty, and that is a complete answer. Scope is **one answer for one date**; running it across the walk-forward and scoring it are Unit 5 (eval).

## 1. Purpose & scope

### Goal
`respond(slug, t0, *, query=None) -> dict` — the subject's market view at T0, as a validated JSON object written to `data/doppelganger/<slug>/views/<t0>.json`.

### The two threats this unit must respect
1. **Data leakage** — handled upstream: the soul is frozen ≤T0 and `load_memory` firewalls evidence to ≤T0.
2. **Model leakage (the residual)** — Opus's *weights* know the post-T future; data-gating can't remove that. Mitigations here: (a) **immersive present-tense framing** ("You ARE Eddy. It is {date}. The future has not happened.") — places the model inside the moment to suppress hindsight, and reads as real-time rather than retrospective; (b) **provenance as the exposure gauge** — `grounded` claims are his real ≤T views (low exposure), `extrapolated` claims are where model-hindsight can creep in (high exposure), so the tags tell the eval which claims to distrust. Elimination is impossible short of an older-cutoff model; the eval (Unit 5) **measures** the residual via an ablation (with-corpus vs. without-corpus). Every walk-forward result carries that caveat.

### Scope (v1)
- One query, one date → one JSON market view. Same isolated `claude -p --model opus --effort max` as the soul unit.
- The query is a single standing **survey** ("what are your current market views"), fixed across all walk-forward steps and both subjects for comparability. `query` override is the seam for topic-targeted probes later.

### Non-goals (v1)
- No walk-forward loop, no scoring, no ablation run (all Unit 5). No retrieval (memory is feed-all). No multi-doppelganger council.

### Success criteria
Produces a schema-valid JSON market view; its citations pass the grounding/leakage audit (real ≤T0 quotes); and Eddy's view is recognizably different from Ali's at the same date (distinct market minds, not a shared template).

## 2. Output contract (the market-view schema)

JSON. The model reasons in `--effort max` thinking; the **output** is the structured verdict (structuring the output is not scripting the reasoning). Every item across the excited/concerned arrays shares one shape:

```json
{
  "as_of": "2022-12-31",
  "subject": "eddy-lazzarin",
  "abstained": false,
  "sectors_excited":   [ { "name": "...", "why": "...", "provenance": "grounded|persisted|extrapolated", "age_note": "stated 2021-06, not revisited", "citations": [ { "date": "YYYY-MM-DD", "quote": "verbatim ≤T0" } ] } ],
  "sectors_concerned": [ /* same item shape */ ],
  "tokens_excited":    [ /* same item shape */ ],
  "tokens_concerned":  [ /* same item shape */ ],
  "risk_regime": { "stance": "risk_on|risk_off|neutral|no_view", "why": "...", "provenance": "grounded|persisted|extrapolated" },
  "notes": "anything that doesn't fit the buckets"
}
```

- **Per-axis independence is mandatory and instructed.** Excited about a sector with no specific token is complete; a token concern with no broad sector view is complete; any array empty is expected, not a gap. **Never manufacture an item to fill a bucket.** `risk_regime.stance: "no_view"` and `abstained: true` (nothing at all) are valid, faithful answers.
- **`age_note`** present on `persisted` items (an older standing view carried forward — see the persistence rule: silence ≈ unchanged for a slow-moving investor).
- **`citations`** are dated verbatim quotes; they let the eval run the same grounding/leakage audit on answers that we run on souls. `grounded`/`persisted` items must cite; `extrapolated` items have no citation (they're inferred from the soul) and are the model-leakage-exposed set.

## 3. Architecture

```
soul.md (frozen@T0) ──┐
                      ├─► build_query_prompt ──► claude -p (isolated) ──► JSON ──► parse/validate ──► views/<t0>.json
load_memory(slug,t0) ─┘                          (system: instructions+soul;
   (≤T evidence feed)                             stdin: memory feed + query)
```

- **Prompt assembly:** `--system-prompt` carries the doppelganger instructions (role, present-tense framing, the schema, the partial-is-correct rule, provenance definitions) **+ the soul card** ("this is who you are"). **stdin** carries the ≤T memory feed (`MemoryView.text`) + the survey query (the large payload). Mirrors the soul unit.
- **Parsing:** strip ```json fences if present, `json.loads`, normalize to the schema (default any missing array to `[]`, missing `risk_regime` to `{"stance":"no_view"}`). Lenient — the LLM produces it; we don't reject on a missing optional key.
- **Reuse / small refactors (DRY):**
  - Extract the `claude -p` wrapper from `soul.py` into `doppelganger/llm.py` (`run_claude(system, user, ...)`); `soul.py` imports it from there (behavior identical, soul tests still pass).
  - Extract the citation-matching core from `soul_audit.py` into `audit_citations(cites: list[Citation], evidence_path, t0) -> AuditReport`; `audit_soul` calls it, and a new `audit_answer(view, evidence_path, t0)` (pulls `{date,quote}` from the JSON's `citations`) calls it too.

## 4. Module layout

```
doppelganger/
  llm.py           # run_claude(system, user, *, workdir=None, timeout=...) — moved from soul.py
  respond.py       # build_query_prompt, respond
  soul.py          # MODIFY: import run_claude from llm
  soul_audit.py    # MODIFY: extract audit_citations core; add audit_answer
  run.py           # add `respond` subcommand
tests/
  test_doppelganger_respond.py
```

## 5. Testing strategy

- **Plumbing (TDD, mocked LLM):** `build_query_prompt` — system contains the present-tense framing, the schema keys, the partial-is-correct rule, and the soul text; stdin contains the memory feed + query. `respond` — with `run_claude` monkeypatched to a canned JSON string (incl. one wrapped in ```json fences), parses + normalizes + writes `views/<t0>.json`, returns the dict; empty arrays / abstention parse cleanly.
- **Audit (TDD):** `audit_answer` flags a hallucinated citation and a post-T0 leaked citation, passes a clean answer — reusing the soul audit's fuzzy matcher. Existing soul-audit tests still green after the `audit_citations` extraction.
- **Real-data gate (not TDD):** generate Eddy's T0 answer; eyeball for sane, Eddy-flavored, partial-where-appropriate views; run `audit_answer` (citations real ≤T0, 0 leaked); generate Ali's answer to the same query and confirm the two hold **different** market views (discrimination). A mini ablation eyeball (answer with vs. without the memory feed) is noted here but the rigorous ablation is Unit 5.

## 6. Open questions for planning

1. **Module/function name** — `respond.py`/`respond()` working name (the doppelganger "responds"); finalize at plan time.
2. **Schema normalization strictness** — exact defaults for missing/partial keys (lean: lenient, default-to-empty, never raise on a missing optional field; only raise if output isn't parseable JSON at all).

---

_Next: on approval, `writing-plans`. Then Unit ❺ (Eval) runs the walk-forward + the ablation + held-out scoring over `respond()`._
