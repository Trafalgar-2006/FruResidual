# Phase 2 at cell-type resolution: the coarse metric was hiding most of the signal

Run: 2026-09-28, via `scripts/run_phase2_celltype.py`.

Phase 2's wiring-difference target compares partner profiles over **10
superclass buckets**. That is coarse enough to be blind to the most
obvious form of rewiring — *same broad partner category, entirely
different specific partners* — and it was the largest acknowledged
methodological weakness left in the project. This rebuilds the identical
target over partner **cell types** and reruns the ablation on it.

Run as a parallel analysis rather than a replacement. A conclusion that
holds at both resolutions is worth more than one that only holds at the
resolution it was designed at — and as it turns out, one of them does
not hold at both.

## Building a shared vocabulary

The difficulty is that MaleCNS partner types (`CB1078`, `DNp48`) and
FlyWire cell types are different naming systems. They are bridged
through MaleCNS's own `flywireType` field, which maps **86.4%** of
MaleCNS partner rows (193,117 of 223,500) into FlyWire's vocabulary.

Both sides are then restricted to the **intersection**: 5,338 shared
cell types, from 6,050 reachable on the MaleCNS side and 7,286 on the
FlyWire side. That symmetry is not optional. Leaving categories in the
comparison vector that only one dataset can express is precisely the bug
that made the original coarse target ~76% artifact
(`docs/phase2_results.md`), and the same failure at this resolution
would be far harder to spot.

FlyWire-side profiles: 123,315 rows across 8,692 distinct partner types,
fetched live at 8 samples per type over 352 types.

## The target separates cases from controls far better

| | cases | controls | ratio |
|---|---|---|---|
| mean | **0.2605** | 0.1115 | 2.34 |
| median | **0.1826** | 0.0855 | 2.14 |

Mann-Whitney p = 3.7e-92, rank-biserial effect size **+0.659**.

Against the coarse target's effect size of **+0.259**, this is a
substantially better measurement of the same underlying quantity. The
coarse metric was not wrong so much as blunt: collapsing thousands of
partner identities into 10 buckets discarded most of what distinguishes
a rewired neuron from an unchanged one.

## The ablation conclusion is resolution-dependent — and that is the finding

Phase 2's headline was that *fru*/*dsx* adds nothing once structure is in
the model. At cell-type resolution that is no longer straightforwardly
true.

| resolution | model | R² without | R² with | Δ | folds improving | Wilcoxon p |
|---|---|---|---|---|---|---|
| coarse | ridge | 0.090 | 0.092 | +0.002 | 3/5 | 0.625 |
| coarse | gradient boosting | 0.219 | 0.216 | −0.003 | 2/5 | 0.812 |
| **fine** | **ridge** | **0.168** | **0.243** | **+0.075** | **5/5** | **0.062** |
| fine | gradient boosting | 0.409 | 0.439 | +0.030 | 3/5 | 0.312 |

Tested paired across folds, because both models see the same splits and
the reported `r2_std` is variability *across* folds rather than the
standard error of the difference — eyeballing overlapping error bars
would have been the wrong comparison. The paired test is computed by
`fruresidual.models.baselines.ablation_fold_test` and recorded under
`ablation_fold_test` in both phases' summary JSON, so every number in
the table above regenerates from a run. (It was originally computed
ad hoc and was not reproducible from the committed pipeline; the
2026-10-13 audit closed that gap and confirmed all four rows unchanged.)

**Ridge: a real effect.** Every one of the five folds improves, by
+0.060 to +0.094. With five folds the smallest p-value Wilcoxon can
return is 0.0625, so p=0.062 *is* the floor — it means the direction was
unanimous, which is the strongest evidence this fold count can produce.

**Gradient boosting: not distinguishable from noise.** +0.030 on
average, but two of five folds get *worse* (−0.034, −0.033).

The natural reading is that at fine resolution *fru*/*dsx* carries
genuine information the **linear** model cannot otherwise reconstruct,
while the tree model already recovers it from interactions among
superclass, connectivity and neurotransmitter. That is consistent with
Phase 0/1: superclass is the strongest correlate of *fru*/*dsx* status,
and a tree can exploit that where a linear model cannot.

## The GNN verdict holds at this resolution too

Phase 2's GNN was negative, but it was measured against a target we now
know was largely artifact, so it was worth refitting rather than
inheriting the verdict. Gradient boosting went 0.219 → 0.409 on the
finer target; the graph model deserved the same second look.

Same protocol as Phase 2 (5-fold CV × 3 seeds = 15 fits), so the number
is directly comparable to the cross-validated baselines rather than a
single lucky split:

| | coarse target | fine target |
|---|---|---|
| best baseline (no *fru*/*dsx*) | 0.219 | 0.409 |
| GNN | −0.388 ± 0.665 | **0.290 ± 0.239** |
| GNN folds beating that baseline | 1/15 | 3/15 |
| beats it on the mean | no | **no** |

The GNN improves a great deal in absolute terms — from clearly broken
to merely unhelpful — and its fold-to-fold variance drops by nearly
two-thirds (±0.665 → ±0.239). But it still does not beat a gradient
boosting model that never sees the graph, on the mean or on more than
3 of 15 folds. The conclusion is unchanged and now rests on a better
target: **1,268 nodes is not enough data to train this GNN
competitively**, and the project plan's pre-registered response ("if the
GNN doesn't beat gradient boosting, report that") still applies.

## The "controls that look rewired" lead closes

`docs/phase2_results.md` left a surviving thread: at coarse resolution,
**5 isomorphic-labelled controls exceeded the 99th percentile of the
case distribution**, i.e. a handful of supposedly unchanged neurons
looked genuinely rewired. That was already the weakened remnant of a
larger effect that turned out to be the category-asymmetry artifact.

Re-examined on a target that separates cases from controls 2.5× better
(effect size +0.659 vs +0.259):

**0 of 634 controls** exceed the 99th percentile of the case
distribution (0.7605).

The lead does not survive the better measurement. The most likely
reading is that it was the residue of the same coarse-metric noise that
produced the original effect — at 10 buckets, a control with an unusual
profile in one bucket can land high in the distribution, and at 5,338
types it cannot. It is closed, not merely smaller.

## What this changes

- **Structure still carries most of the signal.** Gradient boosting
  reaches R²=0.409 *without* ever seeing *fru*/*dsx*, on a target that
  separates cases from controls at effect size +0.659. The project's
  central framing survives.
- **"*fru*/*dsx* adds nothing" does not survive as stated.** It holds at
  superclass resolution for both models and at cell-type resolution for
  the tree model, but fails for the linear model at cell-type
  resolution, consistently across every fold. The honest claim is
  narrower: *fru*/*dsx* adds no **incremental** value to a model that
  can already exploit structural interactions — not that it is
  uninformative.
- **The coarse metric's limitation was real and is now measured.**
  Everything computed on it remains valid for what it measures; it
  simply measures less than the finer version does.

## Caveats

- Five folds is very few for a paired rank test. Ridge's unanimity is
  the strongest available signal at that fold count, not a small
  p-value in the usual sense.
- The FlyWire side samples 8 neurons per type (down from Phase 2's 15,
  to bound a live connection that hung twice mid-fetch), so per-type
  profiles are noisier than the coarse run's.
- The `flywireType` bridge covers 86.4% of MaleCNS partner rows. The
  missing 14% are partners with no clean FlyWire counterpart and are
  dropped from both sides, which is the symmetric choice but does lose
  information.

## Reproduce

```
python scripts/run_phase2_celltype.py
```

Outputs `phase2ct_wiring_targets.feather`, `phase2ct_ablation_table.csv`,
`phase2ct_residual_scores.feather` and `phase2ct_summary.json` to
`data_processed/`. The FlyWire fetch is checkpointed per type, so an
interrupted run resumes where it stopped; the GNN step needs a live
neuPrint connection to pull population-internal edges.
