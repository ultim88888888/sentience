# Corpus Doppelganger — Unit ❷ Soul — Design Spec

_Design spec · 2026-06-05 · status: draft-pending-review_

## 0. One-paragraph summary

The **soul** is a frozen-at-T0 **characterization** of the subject — the view-generating apparatus the doppelganger reasons through. A single Opus-max extraction agent reads the subject's bio + all evidence dated ≤ T0 and writes a **lightly-structured Markdown "soul card"**: how he thinks, what he believes, what he attends to, his contradictions, and (least priority) how he talks — every claim backed by a dated quote from the corpus. The card is the artifact. It is validated, before any downstream unit exists, by an automated grounding+leakage audit and a side-by-side discrimination eyeball against a second subject. First subject: **Eddy Lazzarin**, T0 = 2022-12-31.

## 1. Purpose & scope

### Goal
`soul = extract_soul(subject, t0)` — read `identity_as_of(t0)` + `evidence ≤ t0`, produce a human-readable, evidence-grounded characterization that captures **how the subject generates and selects market views**, frozen to what he knew at T0.

### Why it exists
The doppelganger (Unit 4) reasons *through* the soul. The soul does **not** store his specific dated opinions — those live in memory (Unit 3) and are retrieved per query. The soul holds the **stable apparatus that produces views**: frameworks, priors, reasoning moves, attention. "Get his views right" is precisely the soul's job, delivered through the cognitive machinery rather than by storing opinions.

### Relationship to the engine spec §4 ❷
The engine spec called for "a human-readable, lightly-structured soul document … not a NEO-PI-R scored schema." This spec **concretizes** that as a Markdown soul card, and **refines two leanings from the engine brainstorm** (which were never hard-committed in the engine spec):
1. **Format → Markdown soul card** (the brainstorm floated "structured JSON"). Rationale: the *feed-inputs-don't-script-cognition* principle; rendering-to-prompt becomes free; the card is human-readable (serves the docs/presentable goal) and gives clean git diffs for the soul-evolution visualization. Nothing in the design needs the soul to be machine-queryable.
2. **Extraction → one holistic call** (the brainstorm floated "modular parallel extractors"). Rationale: the soul-card sections are *interdependent* (contradictions emerge from the whole; frameworks shape attention; voice colors belief) — splitting them throws away the coherence that makes a characterization good. The modular rationale (depth + independent tuning) was premised on scale and a scored-schema; the ≤T0 corpus is ~40k tokens, a single Opus pass.

All other §4 ❷ decisions carry over: bio-rooted + corpus-individuated (behavior wins conflicts), evidence-backed, contradictions preserved, frozen + time-gated at T0 (incl. bio truncation), diffable across walk-forward steps.

### Scope (v1)
- One subject (Eddy), one T0 (2022-12-31), single-pass extraction.
- Produces one soul card. The **walk-forward-soul treatment** (re-extract per step) is just calling this extractor with different `t0` — built into the extractor's signature, exercised by the walk-forward harness (later unit), not here.

### Non-goals (v1)
- No JSON/scored schema. No modular extractors. No fine-tuning.
- No system-prompt rendering (Unit 4's job — trivial given the card).
- No quantitative style/prediction scoring (Unit 5 / eval).

### Success criteria
A generated soul card (a) passes the automated grounding+leakage audit (every cited quote exists in ≤T0 evidence and is dated ≤ T0), and (b) is **analytically distinct** from a second subject's card built the same way (Eddy ≠ Ali Yahya as *minds*, not just prose).

## 2. Input / output

**Inputs (Unit 1 artifacts):**
- `data/doppelganger/<slug>/identity.json` → loaded and truncated via `IdentityProfile.as_of(t0)`.
- `data/doppelganger/<slug>/evidence.parquet` → filtered to `timestamp ≤ t0`.

**Output:**
- `data/doppelganger/<slug>/soul.md` — the soul card (YAML frontmatter + Markdown body).
- (audit report printed/logged; see §5.)

## 3. The soul card

A Markdown document. **YAML frontmatter** holds metadata:
```yaml
subject: eddy-lazzarin
name: Eddy Lazzarin
t0: 2022-12-31
built_from: { evidence_items: 329, span: "2019-11-17..2022-12-27", model: claude-opus-4-x }
```

**Body = fixed H2 sections, in priority order (views first; voice last).** Each section is prose characterization plus specific bulleted claims; **every claim carries an inline dated quote** (the evidence the audit checks). Sections:

1. **`## Bio Lens`** — the generative root: how his background (PNP/behavioral-econ → data science → engineering → investor) shapes his analytical lens. Notes inline where corpus evidence **confirmed vs. overrode** the bio-only expectation.
2. **`## How He Thinks`** — reasoning *moves* (e.g. "asks who the marginal buyer is"), epistemic style (how he updates, hedges, calibrates certainty), and his named **frameworks / mental models**. The engine of extrapolation. *(Voice-as-cognition signal folds in here: phrasing/hedging patterns that betray how he reasons.)*
3. **`## What He Believes`** — durable, recurring **priors/convictions** only (stable across the ≤T0 span). Volatile, specific positions are deliberately excluded — they live in memory.
4. **`## What He Attends To`** — what he fixates on vs. dismisses; his attention map across topics/sectors.
5. **`## Open Contradictions`** — genuine tensions in his thinking, preserved, never averaged away.
6. **`## How He Talks`** — *least priority, but kept.* A brief characterization of his register/voice. Zero effort on cosmetic tics for their own sake; included because thought shapes speech and it rounds out the persona.

**Discipline:** claims must be specific and corpus-grounded, never generic-pundit filler. A claim with no dated quote is a defect (the audit fails it).

## 4. Extraction

A single **Opus-max** agent, one call. Prompt instructs it to:
1. Read `identity_as_of(t0)` → form expectations about his lens (priors/hypotheses from bio).
2. Read **all** evidence ≤ t0 → confirm / sharpen / **override** those expectations with dated behavioral evidence; **behavior wins conflicts**. Weight by `attribution_confidence` (solo X/podcast > firm research).
3. Write the soul card per §3: views-first, every claim carrying a real dated quote drawn from the provided evidence, contradictions preserved, bio-confirm/override noted inline.

**Sizing (measured):** Eddy ≤T0 = 329 items / ~40k tokens; full corpus ~200k tokens. Both fit a single Opus context comfortably — no map-reduce, no retrieval, the whole ≤T0 corpus goes in the prompt.

**Model invocation (decided):** **`claude -p` subprocess on the Max subscription — no API cost.** Pattern from the original doppelganger project (`src/eval/doppelganger-runner.ts`): `claude -p --model opus --effort max --no-session-persistence`, run from an **isolated working dir** (e.g. a temp dir) so the extraction does NOT inherit this repo's / Fushi's `CLAUDE.md`. The bio + ≤T0 evidence + extraction instructions are passed as the prompt; stdout is the soul card. The plan nails the exact flags/invocation (consult `claude-api` for the current `--model` value and the doppelganger runner for the established subprocess pattern). **Not** the Anthropic SDK — chosen specifically to avoid API cost given the walk-forward fans this call out ~40× per subject.

## 5. Validation (runs now, before Units 3–5 exist)

Two gates, both cheap, neither needing the doppelganger:

**Gate 1 — Grounding + leakage audit (automated, the anti-bullshit gate).**
Parse the inline dated quotes from the soul card. For each: verify (a) the quote text matches an item in the subject's ≤T0 evidence (fuzzy/substring match — exact matching strategy decided at plan time), and (b) that item's timestamp is ≤ T0. **Any unmatched quote = hallucinated evidence; any post-T0 quote = leakage.** Either fails the audit. Output a report (claims checked, matched, unmatched, leaked).

**Gate 2 — Discrimination eyeball (the make-or-break).**
Generate the T0 soul card for **Eddy and for Ali Yahya** (the engine-spec's ceiling foil). Read side by side: do they present as two **distinct analytical minds** — different frameworks, priors, attention — or two generic crypto-VC templates? Distinctness must be in the *thinking*, not the prose; two cards that differ only in writing style = extraction failure. (Human read, but the bar is analytical, not stylistic.)

**Deferred to the eval unit (5):** quantitative style-match and held-out prediction — they need scoring machinery and overlap eval's job.

## 6. Module layout

```
doppelganger/
  soul.py            # extract_soul(slug, t0, *, evidence_path=None, identity_path=None,
                     #             out_dir=None, model=...) -> Path (writes soul.md)
                     # + the extraction prompt construction
  soul_audit.py      # audit_soul(soul_md_path, evidence_path, t0) -> AuditReport (Gate 1)
  run.py             # add `soul` subcommand: python -m doppelganger.run soul --subject eddy-lazzarin --t0 2022-12-31
```
Consumes Unit 1's `identity.py`/`schema.py`/`config.py`. The LLM call is isolated in `soul.py` behind a thin function so it can be mocked in tests (the extraction itself isn't unit-tested for content; the *plumbing* — input gating, prompt assembly, output writing, the audit — is).

## 7. Testing strategy

- **Plumbing (TDD, mocked LLM):** input time-gating (evidence filtered ≤ t0; identity truncated), prompt assembly includes the right items, output written to the right path with valid frontmatter, `soul` CLI wiring.
- **Audit (TDD, real logic):** `audit_soul` correctly flags a hallucinated quote, a post-T0 (leaked) quote, and passes a clean card — against small fixtures.
- **Real-data gate (not TDD):** generate Eddy's actual soul card, run the audit on it, and produce Eddy + Ali cards for the discrimination eyeball.

## 8. Open questions for planning

1. ~~Model invocation~~ → **decided: `claude -p --model opus --effort max` subprocess, isolated dir, no API cost** (§4). Plan confirms exact `--model` value via `claude-api` + the doppelganger runner pattern.
2. **Evidence-citation format in the card** — how quotes are marked so the audit can parse them reliably (e.g. a trailing `— [<source_type> <date>] "quote"` convention).
3. **Audit match strictness** — substring vs. fuzzy threshold for matching a cited quote back to an evidence item (quotes may be lightly trimmed/elided in the card).
4. **Prompt content** — the actual extraction instructions (the plan writes them; this spec fixes what they must achieve).

---

_Next: on approval, `writing-plans` for the soul unit._
