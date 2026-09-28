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
would have been the wrong comparison.

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

Outputs `phase2ct_wiring_targets.feather`, `phase2ct_ablation_table.csv`
and `phase2ct_summary.json` to `data_processed/`. The FlyWire fetch is
checkpointed per type, so an interrupted run resumes where it stopped.
