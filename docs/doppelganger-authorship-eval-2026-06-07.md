# Doppelganger evaluation — blind authorship attribution (2026-06-07)

**Question.** Does the corpus produce genuinely *distinct minds*, or do the doppelgangers
blur into a generic "a16z GP"? The walk-forward `confirm_rate` saturated (1.0 everywhere) and
couldn't answer this. This eval answers it as a number.

**Subjects.** Eddy Lazzarin, Ali Yahya (a16z crypto GPs). 14 quarterly views each,
2022-12-31 → 2026-03-31.

**Code.** `doppelganger/authorship.py` (run: `python -m doppelganger.authorship`). Pure logic
covered by `tests/test_doppelganger_authorship.py`. This run was executed 2026-06-07 at
`--effort high`.

---

## Method

A judge (`claude -p`, blind to all identity) is shown a **stance-only** view — sectors/tokens
the author is excited/concerned about, each with its `why` reasoning and `conviction`, plus
risk posture and notes. **Stripped before judging:** subject, date (`as_of`), citations
(verbatim quotes), provenance, age_note. The judge must attribute it to `eddy-lazzarin` or
`ali-yahya` (forced choice) using the view's substance + its own knowledge of the two real
people, and report a confidence (0–100).

The test runs as a **three-rung ladder** that varies how much identity the *generator* had:

| rung | identity given to the generator (when the view was produced) |
|---|---|
| **FULL** | frozen soul card + time-gated corpus memory |
| **NAMED-ABLATION** | identity stub (name + one-line headline) only — no soul, no memory |
| **ANON-ABLATION** | nothing — a generic "a crypto venture investor" stub, no name |

FULL and NAMED-ABLATION reuse the committed walk-forward views (`views/`, `views_ablation/`).
ANON-ABLATION generates 14 fresh generic views (one per quarter; no memory feed, so cheap) to
a scratch dir, then judges them. Because anon views have no true author, their metric is the
**split** (≈50/50 would mean no identity signal) and **confidence**, not accuracy.

Why a ladder: it isolates the confound that these are **public figures the base model already
knows**. A single ablation can't tell "the corpus made them distinct" from "the model
remembers Eddy and Ali from pretraining." The ladder separates corpus signal from name-knowledge
from base-model default.

---

## Results

| rung | accuracy | per-subject | mean judge confidence |
|---|---|---|---|
| **FULL** | **100%** (28/28) | Eddy 14/14, Ali 14/14 | **86** |
| **NAMED-ABLATION** | **82%** (23/28) | Eddy 10/14, Ali 13/14 | **78** |
| **ANON-ABLATION** | — (no true label) | split: **Eddy 10/14, Ali 4/14** | **61** |

**NAMED-ABLATION misses (5):** 4 are Eddy→Ali, all with the same tell — soul-less Eddy drifts
to an *"AI×crypto convergence / machine-to-machine payments / distributed-systems"* framing that
reads as Ali. 1 is Ali→Eddy (2022-12-31, his earliest/thinnest quarter) on a ZK/DePIN-infra tell.

**ANON-ABLATION skew:** the generic, identity-free base-model default skews **Eddy 10 / Ali 4**.
Every Eddy-labeled generic view leans ZK-proving / rollup / L2 fee-economics / "revenue not
narrative" — i.e. the base model's untethered "crypto GP" voice is **Eddy-shaped**. The 4
Ali-labeled ones are exactly the quarters where the generic view foregrounds AI×crypto / DePIN
compute / agentic payments.

---

## Interpretation (honest — this revises the synthesis memo)

1. **No collapse to chance.** The earlier synthesis memo
   (`docs/doppelganger-synthesis-2026-06-07.md`) claimed that stripping the corpus makes the two
   subjects "nearly the same document / near-interchangeable." **That is overstated.** Even
   name-only views are 82% attributable, and even *generic* views carry residual identity signal.
   The corpus does not rescue them from chance — they were never at chance.

2. **The base-model default is Eddy-shaped.** "Eddy" (ZK/infra/data-mechanics) overlaps heavily
   with what Opus produces by default for a crypto GP, so he stays identifiable with little/no
   corpus. **Ali is the distinctive one** — his AI×crypto/coordination/Google-Brain lens is *not*
   the default, so soul-less and generic views rarely read as Ali.

3. **The corpus's measurable contribution is the confidence ladder: 61 → 78 → 86**, plus pushing
   accuracy to a 100% ceiling and (per the misses) **repairing Eddy's drift** into the generic AI
   thesis. Accuracy understates the lift because of the ceiling and the high name-knowledge floor;
   **confidence is the cleaner signal** here.

**Bottom line:** the corpus genuinely sharpens identity (monotonic confidence gain, ceiling
accuracy, drift repair), but the "two distinct minds" story is carried as much by the base
model's pretrained knowledge of these public figures as by the corpus. To claim the corpus
*creates* distinctiveness, we'd need subjects the base model does **not** already know.

---

## Caveats

- **N=2**, both well-known public a16z GPs → high name-knowledge floor. Not a published result.
- Single judge; forced binary choice; the judge's own model-knowledge is the ground-truth proxy.
- Eval-generation effort confound: FULL/NAMED views were generated at `--effort max` (≤2026-06-06);
  ANON views at `--effort high` (2026-06-07).
- ANON has no true label, so it contributes a split/confidence reading, not an accuracy.

## Next steps

1. **Unknown-subject control.** Re-run with a subject the base model does *not* know well
   (e.g. a pseudonymous writer) → the ANON/NAMED floors should drop toward chance, exposing the
   corpus lift uncompressed. This is the clean way to prove corpus-*created* distinctiveness.
2. **Pairwise same-quarter discrimination.** Show the judge Eddy's and Ali's view for the *same*
   quarter side by side and ask which is which — removes era cues, isolates persona.
3. Fold confidence (not just accuracy) into the headline metric set alongside change-recall +
   groundedness.

## Reproduce

```bash
python -m doppelganger.authorship --out data/doppelganger/authorship_eval.json
```
Raw per-view judgments from this run: `data/doppelganger/authorship_eval_2026-06-07.json` (untracked).
