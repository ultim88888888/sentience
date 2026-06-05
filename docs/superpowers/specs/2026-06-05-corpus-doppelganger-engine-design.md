# Corpus Doppelganger Engine — Design Spec

_Design spec · 2026-06-05 · status: draft-pending-review_

## 0. One-paragraph summary

Build an engine that constructs a **digital doppelganger of a person purely from their public corpus** (X posts, research, speaker-attributed podcast turns, bio/LinkedIn) — no interviews, no elicitation, no human validation loop. The doppelganger is used as an **analytical lens**: given a date T, it answers questions ("what sectors/tokens is he excited or concerned about, and why") **as that person would, knowing only what they knew at T**. We **walk it forward** month by month over years. First subject: **Eddy Lazzarin** (a16z crypto GP). This is the corpus-only sibling of the interview-based `projects/doppelganger` self-clone — same lineage, different data source and a new time axis.

## 1. Purpose, scope, success

### Goal
A reusable engine: `opinion = doppelganger(subject, as_of_date, query)`, built from `corpus(subject)`, where the answer is faithful to who the subject was **at that date**.

### Why
Doppelgangers become lenses to review market data and construct market views — a corpus-grounded "expert agent." The walk-forward design also serves a research question: can a person's evolving market stance be reconstructed and (because the corpus is timestamped) **validated** against what they actually said next?

### v1 scope — the smallest honest engine
- **One subject** (Eddy), all available sources, pure-corpus.
- **Frozen-soul control** as the default walk-forward mode (see §5); walk-forward-soul is the treatment.
- **Structured market-view output** with first-class abstention.
- **Leakage-controlled** and **eval-instrumented** from day one.

### Non-goals (v1)
- No council/multi-doppelganger consensus (deferred — §8).
- No fine-tuning / weights-level training (§8).
- No associative entity/framework **graph** retrieval (§8) — lightweight indexing only.
- No external data fed to the doppelganger (no other authors, news, prices) — pure-corpus.
- No claim of market alpha. We measure **fidelity to the person**, not correctness of their views.

### Success criteria
The engine clears its own eval (§7): the doppelganger's walk-forward answers (a) **predict** the subject's actual held-out subsequent statements above baseline, (b) are **discriminable** from another subject's doppelganger, (c) **match the subject's style** against held-out real text, and (d) **abstain** appropriately. Leakage is measured and bounded, not assumed.

## 2. Core design principle

> **We feed the model inputs; we don't script its cognition.**

Anything that controls *what data the model sees* is in scope (time-gating, retrieval, the persona characterization). Anything that scripts *how it reasons or selects* is out — that's the LLM's job, and it does it better than a hand-tuned formula we'd have no ground truth to fit. This principle is why the engine is deliberately thin (see §9, "What we cut").

The doppelganger and the extraction agent both run **Opus at max thinking/effort** — a hard requirement, never downgraded for latency or cost.

## 3. Architecture — five units

Five isolated units. Each has one purpose and a well-defined interface; downstream units consume artifacts, never raw sources. The dotted line is the time-gate that makes walk-forward honest.

```
SOURCES                 INGESTION                 CONSTRUCTION            QUERY
────────                ─────────                 ────────────           ─────
LinkedIn ─┐
a16z bio ─┴─► identity profile (static) ──┐
                                           ├─► ❷ SOUL ───────────────┐
X ────────┐                                │   (bio-rooted,          │
research ─┼─► ❶ evidence stream ───────────┘    corpus-grounded,     ▼
podcast ──┘    (dated, normalized,              frozen@T0)      ❹ DOPPELGANGER
                attribution-tagged)                             soul + retrieved
                      │                                         memory ≤T → reasons
                      └─► ❸ MEMORY (time-gated RAG) ··≤T··────► in-character, answers
                                                                structured query w/
                                                                per-claim provenance
                                                                      │
                                       ❺ EVAL ◄──── walk T forward ───┘
                                       held-out prediction · discrimination · style · coverage
```

| # | Unit | Contract |
|---|------|----------|
| ❶ | **Ingestion** | raw sources → (a) merged *identity profile* (static, time-truncatable), (b) normalized *evidence stream* of dated utterances |
| ❷ | **Soul** | identity profile + corpus ≤T → a *characterization document* of how the subject thinks/talks/frames. Frozen at T0 (control) or re-derived per T (treatment) |
| ❸ | **Memory** | evidence stream → time-gated retrieval; a query at date T returns only utterances ≤ T |
| ❹ | **Doppelganger** | soul + retrieved memory ≤T → an Opus-max reasoner that answers the structured query in-character, marking each claim's provenance |
| ❺ | **Evaluation** | walks T forward, scores the doppelganger against the subject's actual held-out words |

## 4. Unit specifications

### ❶ Ingestion

Each source is an **adapter** normalizing into shared schemas. Add a future source (e.g. Farcaster) by writing one adapter; nothing else changes.

**Output A — Identity profile (static, time-truncatable).** Merge LinkedIn (spine: full `experience[]`, `education[]`) + a16z team bio (firm positioning, current title, sections). Dedup overlap (they overlap by design). Expose `identity_as_of(T)` → drops experience/education entries started > T and clips current role to what was active at T. _(So the EOY-2022 soul knows Eddy as Head of Engineering, not GP — this is the bio time-gate; without it the soul leaks its own future.)_

**Output B — Evidence stream (dated utterances).** Uniform item:
```
{ id, subject, timestamp (UTC), source_type, text,
  speaker_slug, attribution_confidence, thread_id, context, engagement }
```
The unit is the **natural utterance, reassembled to stand alone** — not too granular (fragments retrieve badly), not too coarse (noise). Adapters:

- **X** (`data/twitter/<handle>.parquet`; Eddy: 1,637 items, 2021–2026, typed reply/quote/original/retweet):
  - Originals → keep (solo voice, highest fidelity).
  - Quotes → keep; text = his comment, quoted tweet attached as `context` (dangling `quoted_id` → flag `context_missing`).
  - Replies → **filter, don't blanket-drop**: logistics/noise out, substantive argument in (he reasons in replies). Cheap length+substance pass, threshold tuned against eval.
  - Self-threads → stitch consecutive tweets via `conversation_id`/`in_reply_to_id` into one opinion-unit.
  - Retweets → drop for v1 (negligible, no original text).
  - Engagement carried as metadata (possible salience signal later; not v1-critical).
  - All timestamps tz-aware UTC.
- **Research** (`data/a16z_research/articles.parquet`; Eddy: ~43 attributed posts):
  - Body from `acf_content`/`extracted_text`, paragraph-chunked, whole-doc referenced.
  - **Attribution split:** solo-authored → high `attribution_confidence`; co-authored firm pieces ("things we're excited about") → lower confidence, flagged `firm_voice` (he co-signs; not verbatim his words). Passage-level attribution within firm posts is **deferred**; v1 flags at post level.
- **Podcast** (`data/a16z_research/attributed_transcripts.jsonl` from the corpus-attribution pipeline; Eddy: 12+ diarized records, growing):
  - Keep `segments` where `slug == <subject>`, threshold on `confidence` (≈ ≥ 0.8). His turns = high-fidelity spoken voice.
  - Interlocutor turns attached as `context` (so a retrieved answer carries its question), never attributed to the subject.
  - Dated to episode `post_date` (no per-utterance timestamps; episode date is the gate).

Cross-source: dedupe near-identical cross-posts (a tweet quoting his own essay), sort by time → the stream.

### ❷ Soul (characterization, not a scored schema)

A **rich qualitative document** the doppelganger reads — distilled once so the doppelganger stays consistent across queries without re-deriving the persona each call, and so it freezes cleanly at T0. Built by an **Opus-max extraction agent** in two passes:

1. **Prior formation (bio):** read `identity_as_of(T)` → hypotheses about his lens (e.g. PNP/behavioral-econ + data-science + engineering + investor arc → quantitative, incentive/mechanism-oriented, systems-thinking, insider allocator). Marked as bio-hypothesis, low confidence.
2. **Corpus grounding:** read the evidence stream ≤T → **confirm / sharpen / override** the bio priors with dated behavioral evidence, weighted by `attribution_confidence` (solo X/podcast > firm research). **Behavior wins conflicts with bio.** Add what bio can't give: his voice, actual positions, recurring moves.

What the characterization captures (as **extraction guidance**, not enforced sub-stores — the consult's content/process insight kept as wisdom, not machinery):
- **How he talks** — written register (case, punctuation, hedging, tics, humor) from X; spoken cadence from podcast turns. _(The proven win from the original project: style is highly capturable.)_
- **How he thinks** — his reasoning *moves* (e.g. "asks who the marginal buyer is," how he updates, how he hedges) and named frameworks/mental models. This is the engine of extrapolation for novel questions.
- **What he believes** — durable convictions/priors, distilled only when **recurring and stable**; volatile/specific positions stay in memory and are superseded by later ones (the "GPT-5.5" rule below). **Note where and when his views changed** — a person who only accumulates convictions reads as eerily static.
- **What he attends to** — topics he fixates on vs. dismisses.

Discipline carried from the original: every characterization claim is **evidence-backed** (dated quote + source) and **contradictions are preserved**, never averaged away.

**Output:** a soul document (human-readable + lightly structured for diffing across walk-forward steps), rendered into the doppelganger's system context. Not a NEO-PI-R scored schema — the clinical-relational facets (attachment, neuroticism, family) are unrecoverable from a professional corpus and irrelevant to a market lens.

### ❸ Memory (time-gated retrieval)

The evidence stream, indexed for retrieval. Memory is **monotonic** — indexed once over the full stream; queries filter by date. No re-index per walk-forward step (only the soul re-derives in the treatment; memory just accretes).

- **The leakage firewall (non-negotiable):** the `timestamp ≤ T` filter is applied **at the index query, before ranking** — never post-hoc (post-hoc risks off-by-one and ranking leakage). Every retrieval is **logged** (items, dates, scores); that log is what the eval uses to *prove* nothing > T surfaced and to detect the base model "knowing" something not in the retrieved evidence.
- **Retrieval = plain hybrid RAG** (semantic + lexical). Lexical/BM25 is required for crypto vocab — tickers, "ZK" vs "zero knowledge," protocol names — that pure-vector misses. Assemble the relevant ≤T material into context; the **LLM does the selection and reasoning over it** (no activation-function scoring layer).
- **Lightweight organization:** items tagged by topic and by framework (framework tags double as a cheap cross-topic associative hop). All ≤T memory is accessible — the subject holds *many* views across time; the doppelganger sees the spread (current stance, evolution, genuine tensions), not a single latest take.
- **Diverse/contradictory views are surfaced, not flattened** — the doppelganger, reasoning over the retrieved spread, selects which take fits the query's framing (the multi-authentic-take selection problem is left to the model, given the material).

### ❹ Doppelganger (+ query/opinion)

An **Opus-max** reasoner. Reads: the soul (frozen@T0) + retrieved memory ≤T + the query. Produces the **structured market view**:
- sectors/tokens excited about, concerned about, risk-on/off regime bias, and the **"why"** behind each.
- **Optionality is first-class:** 0/1/many sectors; tokens may be absent; risk regime may be absent. **Abstention is a valid answer** — a faithful subject says "no view" when the real one wouldn't have one. Inventing an opinion is *less* faithful, not more.

**Per-claim provenance** (the doppelganger marks each claim; not a separate mechanical typing system):
- `grounded` — he actually said it ≤ T (cite the dated evidence).
- `persisted` — an earlier standing view, nothing new since; carried forward and **annotated with its age** (he's a slow-moving VC; silence usually means "position unchanged," not "no position").
- `extrapolated` — inferred from the soul when memory is silent; the model shows its derivation.
- `abstain` — genuinely no view.

Reconstruction over parroting: verbatim quotes are **citations under reconstructed claims**, never the claim itself.

### ❺ Evaluation

See §7.

## 5. Walk-forward: control vs. treatment

The walk-forward advances T (e.g. monthly from a chosen T0 to present), re-querying and logging the evolving view as a trajectory.

- **Control — frozen soul (v1 default):** extract the soul once at T0 (gated to ≤T0, bio truncated to ≤T0). Walk forward by advancing **memory** only; the soul is fixed. Cleanest attribution and the baseline we need.
- **Treatment — walk-forward soul:** re-extract the soul at each step T from corpus ≤T (bio re-truncated). Captures persona drift / belief revision over time, at the cost of murkier attribution and higher compute (only the soul re-derives; memory still just accretes). Run **after** the control works, to measure whether soul-evolution adds anything.

Both modes are always time-gated: **no soul ever sees its own future.**

## 6. Key decisions & rationale (resolved during design)

- **Approach B over A (single prompt) and C (fine-tune).** B = time-gated RAG + characterization + reasoning LLM. A can't hold years of dated utterances and lets the generic-pundit prior fill gaps; C is infeasible on this little data and makes time-gating a leakage nightmare (a model per cutoff). _If B saturates, revisit C._
- **Soul rooted in bio + experience, individuated by corpus.** Bio sets the priors (the generative root of his lens); the corpus is the evidence that grounds and individuates them. **Behavior wins conflicts.** Bio gives the category ("ex-FB-eng crypto GP"); the corpus gives the individual (Eddy, not the archetype).
- **Soul vs. memory split (the spine).** Soul = stable "how he thinks/talks." Memory = dated "what he's said." A surfaced view = the soul reasoning over time-gated memory. Worked example — *"Eddy prefers GPT-5.5 to Claude"*: a volatile, dated **memory** item (superseded if he switches), **not** a soul fact; the soul holds the disposition that *generates* such preferences. Baking it into a frozen soul would make it stale on walk-forward — concrete reason B's split beats A.
- **Persistence default.** No new utterance on a topic → carry the last stated view forward (`persisted`, age-annotated), not abstain. He's a slow-moving investor; silence ≈ unchanged.
- **All ≤T memory accessible, diversity preserved.** Not a "latest view" lookup; the subject holds many genuine, sometimes contradictory views, and the model selects per query framing.
- **Opus max-think** for doppelganger and extractor — hard requirement.

## 7. Evaluation strategy

The key insight that makes this tractable: **the corpus is the subject's held-out ground truth** — something the interview-based original never had. The walk-forward is therefore not just the use case; it is the eval methodology.

- **Held-out prediction (primary).** Build the doppelganger ≤ T, ask it about a topic, score its answer against what the subject **actually said at T+1**. Real temporal cross-validation.
- **Discrimination (cheap, make-or-break).** Can a judge tell this doppelganger's output from another subject's (Eddy vs. Kominers)? Can it tell doppelganger-Eddy from real held-out Eddy? If subjects converge to the same generic view, the engine captures no individual signal — **nothing else matters until this passes.**
- **Style match (measurable).** Generated text vs. held-out real text — register, function words, tics — quantified, not eyeballed.
- **Coverage diagnostic (optional).** Over a query set, the fraction `grounded / persisted / extrapolated / abstain`. Tells us empirically — for this subject — how much the engine is recalling vs. inferring, and therefore how load-bearing the "how he thinks" layer actually is. A diagnostic, **not** an enforced ratio.
- **Leakage check.** Cross-reference answers against the retrieval log: flag any claim relying on knowledge not in the ≤T evidence (base-model hindsight). Leakage is **measured**, not asserted.

Caveat to hold: the subject's *later public statements may sit in the base model's training data*, weakening pure prediction — which is exactly why discrimination + leakage checks run alongside it.

## 8. Explicitly deferred (out of v1 scope)

- **Council / multi-doppelganger consensus.** A layer *on* the engine, not part of it. Engine constraint it implies: doppelgangers must be independently queryable and composable; "do they see each other's takes first" is just an input-construction toggle. Defer the design.
- **Associative entity/framework graph + spreading-activation retrieval.** Powerful but a research subsystem; partly duplicates what the reasoning LLM already does natively. v1 ships lightweight topic+framework tagging. **Build the graph only if the eval shows the doppelganger failing specifically on cross-domain connections the lightweight index starved.**
- **Fine-tuning (approach C).** Revisit only if B saturates.
- **External-data analysis** (other authors, news, prices through the lens). Pure-corpus first.
- **Passage-level attribution** within co-authored firm posts. Post-level flag for v1.

## 9. What we cut, and why (deliberate — don't re-add without eval signal)

A memory-science consult proposed a sophisticated apparatus. We kept its **insights as extraction guidance** and cut its **mechanisms**, per §2 (feed inputs, don't script cognition):

- **Cut: ACT-R activation scoring** (base/recency/spread/fan weights) — let the model select which take fits; no coefficients to overfit with no ground truth.
- **Cut: consolidation thresholds / asymmetric promotion math** — the model reads "he believed X, shifted to Y in 2023" from the corpus directly.
- **Cut: enforced grounded/extrapolated ratios** — at most an optional diagnostic (§7), never a target.
- **Cut: formal semantic/procedural/episodic three-store architecture** — collapsed into the soul characterization (process insight) + memory (episodic).

These thinned units from the inside; they did not remove any. Re-add a mechanism **only** when the eval points at the specific failure it addresses.

## 10. Module location, naming, deliverables

- **Lives in the `sentience` repo** as a new module (working name `doppelganger/`; final name TBD to avoid confusion with the separate interview-based `projects/doppelganger`). No filesystem collision (different repos); conceptual distinction documented.
- **In-module docs/visual explainer is a first-class deliverable** — a `docs/` explainer with the architecture chart (Mermaid), the soul/memory model, and the walk-forward + eval flow, presentable to Jax and others. The soul's light structure enables a **soul-diff visualization** across walk-forward steps (how the persona evolves) — a genuine research artifact, not just docs.

## 11. Open questions for planning

1. **T0 choice and step size** for Eddy's walk-forward (data starts 2021; X is the density driver). Monthly? From when?
2. **Query bank** for the walk-forward — fixed question set re-asked each step, or topic-targeted from what was live that month? (Fixed set makes the trajectory and discrimination cleaner.)
3. **Discrimination foil** — second subject to build (Kominers has 20 research posts + a large X corpus; a strong contrast).
4. **Eval scoring** — LLM-judge rubric for held-out prediction; how to score "predicted his stance" without overfitting to wording.
5. **Reply-filter threshold** and podcast `confidence` threshold — initial values, then tuned against eval.
6. **Soul rendering** — exact format of the characterization → system context.

---

_Next: on approval, `writing-plans` to produce the implementation plan._
