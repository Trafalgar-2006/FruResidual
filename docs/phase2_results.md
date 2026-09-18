# Phase 2: predicting the form of rewiring

Run: 2026-09-18, via `scripts/run_phase2_data.py` then `scripts/run_phase2.py`.

## Population and target

Per `src/fruresidual/features/rewiring.py`: a **case-control design**, not
the full traced population. Cases are the 634 "sexually dimorphic" /
"potentially sexually dimorphic" MaleCNS neurons that (a) have a
non-comma flywireType, (b) that flywireType has a real match in FlyWire's
own `cell_type` vocabulary (~91-93% of nominally-matched neurons do — the
rest were silently checked and excluded, not assumed), and (c) sit in a
"brain-like" superclass (excludes `vnc_*`/`ascending_neuron`/
`descending_neuron`, since FlyWire is brain-only and can't fairly score
wiring differences whose defining feature is VNC connectivity). Controls:
634 isomorphic neurons from the same eligible pool, sampled to match
cases' superclass distribution.

**male-specific/potentially-male-specific neurons are entirely excluded**
from this phase — confirmed 0% FlyWire type coverage (they have no
female counterpart by definition), so there's no valid "rewiring form"
target to compute for them. They remain relevant to the overall project
(Phase 0/1), just not to this specific matched-pair comparison.

**Target**: for each case/control neuron, a partner-superclass
connectivity profile is built two ways — MaleCNS side via a targeted
neuPrint query (server-side aggregated, ~42ms/neuron), FlyWire side by
sampling up to 15 neurons of the matched type and averaging their
connectivity (`fafbseg.flywire.get_connectivity`, ~1s/neuron, 371
distinct types queried, checkpointed every 5 types after an earlier
un-checkpointed run risked losing ~20 min of progress to this
connection's known flakiness). Both profiles are restricted to the 10
superclass categories FlyWire can express (see rewiring.py — VNC-only
categories are dropped from both sides, not zero-filled, so a data-
coverage gap can't masquerade as a wiring difference). The target is
`1 - cosine_similarity` between the two profiles.

### The target was rebuilt on 2026-09-19 — a category asymmetry inflated it ~4x

`SHARED_CATEGORIES` listed `ascending` and `descending`, and **no MaleCNS
superclass mapped to either**, so the male profile vector was
structurally zero in those two dimensions while the FlyWire vector was
not. FlyWire carried 104,092 weight (3.6%) there, across **293 of 371
types**. Every bit of it inflated cosine distance one-sidedly — precisely
the "dataset coverage gap masquerading as wiring difference" that
`rewiring.py`'s own docstring claimed to prevent. It accounted for ~76%
of the measured signal.

The fix is a *mapping*, not a deletion: MaleCNS does have
`ascending_neuron` (55,808 weight) and `descending_neuron` (117,265) as
partner superclasses, they were simply being dropped as "VNC partners."
Being excluded from the *case population* (a neuron whose defining
connectivity leaves the brain can't be scored fairly against brain-only
FlyWire) is a separate question from being a valid *partner category* —
an ascending neuron is a perfectly real partner of a brain neuron, and
both datasets can see it. `rewiring.py` now asserts at import that every
shared category is reachable from both sides, so this cannot return.

Deleting the two categories instead — the first fix attempted — made
cases score *lower* than controls, which is what made it clear that
deletion was the wrong remedy rather than a real finding.

| | cases | controls | ratio |
|---|---|---|---|
| old (inflated) mean | 0.01461 | 0.00663 | 2.20 |
| **corrected mean** | **0.00339** | **0.00322** | 1.05 |
| corrected median | 0.000724 | 0.000297 | 2.44 |

Sanity check (regression-pinned in `test_phase2_integration.py`): cases
still show higher wiring difference than controls, Mann-Whitney
p=8.0e-16, rank-biserial effect size +0.259 (down from the inflated
+0.301). The test now uses that rank test rather than a bare mean
comparison — the distribution is heavily right-skewed with a fatter
control tail, so the two *means* sit close together even though the rank
separation is large. Spearman correlation between old and corrected
targets is 0.90: the bug inflated magnitudes roughly fourfold but largely
preserved rank order, which is why the finding survives.

`cosine_distance` also used to return 1.0 (maximum difference) for an
empty profile, asserting maximal rewiring from zero evidence. It now
returns NaN and `run_phase2_data.py` drops those rows. No neuron in the
current population hits this path — it was a latent trap, not an active
bug.

## The ablation: fru/dsx adds ~nothing on top of structure

Numbers below are on the corrected target, so they are **not** comparable
to the pre-2026-09-19 figures (R²=0.507 etc.), which were fitted to a
target that was ~76% artifact.

| model | R² without fru/dsx | R² with fru/dsx | difference |
|---|---|---|---|
| ridge | 0.090 ± 0.154 | 0.092 ± 0.154 | +0.002 |
| gradient_boosting | **0.219 ± 0.138** | 0.216 ± 0.132 | -0.003 |

Predictability dropped sharply along with the target (gradient boosting
0.507 → 0.219), and that is the expected direction: the artifact was
driven by *ascending/descending partner weight*, which correlates
strongly with superclass — a feature the model had. The model was
substantially predicting the artifact, and the artifact was structural,
hence easy to predict. What remains is the real, and much more modest,
predictable component.

**The headline conclusion is unchanged**: both ablation differences are
far inside one cross-validation standard deviation, so fru/dsx still adds
nothing once structure is in the model.

The ridge baseline was also mis-specified twice over, fixed in the same
pass: nominal categoricals (`superclass`, `subclass`, `somaSide`,
`consensus_nt`) were integer-coded with `.cat.codes`, asserting a false
ordering to a *linear* model, and features spanning thousands
(`total_connectivity_weight`) down to [0,3] (`connectivity_entropy`) were
fed to an L2 penalty unscaled, so regularization hit them unequally. Now
one-hot encoded, with `StandardScaler` inside the CV pipeline (so the
scaler refits per fold rather than leaking test-fold statistics). Tree
baselines are invariant to both problems, which is exactly why this was
invisible in the gradient-boosting numbers.

Both differences are far inside one cross-validation standard deviation.
**Adding a fru/dsx feature does not improve wiring-difference prediction
once superclass, connectivity structure, and neurotransmitter are already
in the model.** This is a real, if not fully surprising, result: Phase
0/1 already showed superclass (especially `ol_intrinsic`) is the
strongest correlate of fru/dsx-unexplained status, so a model that
already has superclass as a feature has little room left for fru/dsx to
add. Read together with Phase 0/1, this reinforces rather than
contradicts the project's core hypothesis — regional/structural identity
looks like a stronger predictor of wiring difference than the genetic
switch itself, consistent with "some other factor, correlated with
region, is doing real work here."

## GNN does NOT reliably beat the baselines (corrected 2026-09-19)

| model | test R² | test MAE |
|---|---|---|
| ridge (without fru/dsx) | 0.090 ± 0.154 (CV mean) | 0.005 |
| gradient_boosting (without fru/dsx) | **0.219 ± 0.138** (CV mean) | 0.004 |
| GNN (population-internal subgraph, 7,441 edges) | **-0.592 ± 1.000** (10-seed mean) | 0.004 |

**This section previously claimed the GNN "clearly beats both baselines"
at R²=0.735. That was wrong, and the way it was wrong is worth
recording.** Two compounding errors:

1. **`train_gnn` was not actually seeded.** `random_state` seeded the
   numpy train/val/test split but not PyTorch's global RNG, so model
   weight init differed every run. The same call returned R²=0.735, then
   0.702, then 0.615 on successive reruns. Found during a repo-wide
   audit by diffing documented numbers against the regenerated
   `phase2_summary.json`; fixed by seeding `torch.manual_seed` inside
   `train_gnn`, with two regression tests (`test_gnn_is_reproducible_across_runs`
   pins determinism, `test_gnn_different_seeds_give_different_results`
   stops an over-aggressive seed fix from making the first test vacuous).
2. **One GNN run was compared against cross-validated baselines** —
   apples to oranges. The baselines were already reported as means over
   CV folds; the GNN was a single 15% test split (~190 nodes).

Reporting the GNN the same way the baselines are reported — swept over
10 seeds (`GNN_SEEDS` in `scripts/run_phase2.py`), which is
repeated-random-subsampling cross-validation, each seed drawing a fresh
train/val/test split as well as a fresh initialization — gives
R² = **-0.592 ± 1.000** on the corrected target, spanning **-2.324 to
+0.556**. It beats the gradient-boosting baseline on **3 of 10 seeds**,
and its mean is far below it. (On the old inflated target the same sweep
gave 0.399 ± 0.478 and 6/10; the conclusion was already negative and the
corrected target makes it more so.)

So the honest answer to the plan's pre-registered question ("if the GNN
doesn't beat gradient boosting, report that") is: **it doesn't, not
reliably.** On this population the GNN is far too seed-unstable to claim
a win. The instability itself is the finding — with only 1,268 nodes,
7,441 internal edges and a ~190-node test split, there isn't enough data
to train a GNN stably here, and the single lucky run that produced 0.735
should never have been reported as a result. A defensible GNN comparison
would need k-fold cross-validation over the graph (not one split), and
probably a larger population than this case-control design provides.

This does **not** affect any other Phase 2 result: the ablation table,
the wiring-difference target, and the per-neuron residual scores all
come from the scale-invariant, deterministic sklearn baselines, not the
GNN. Nothing downstream (Phase 3, Phase 4) consumed the GNN's output.

One real bug caught and fixed before this result was trustworthy: the
first GNN run diverged catastrophically (R² = -7250, MAE = 2.58, against
a target that never exceeds ~0.7) because raw, unstandardized features
(e.g. a connectivity-weight sum in the thousands) were fed directly into
a neural net. Tree-based baselines are scale-invariant, which is exactly
why the same raw features worked fine for them and masked the bug there.
Fixed by z-score standardizing features inside `build_graph`, with a
regression test (`test_gnn_stable_with_unscaled_large_magnitude_features`)
pinning the failure mode so it can't silently recur.

## Per-neuron residual scores (the Phase 3 handoff)

Saved to `data_processed/phase2_residual_scores.feather`: out-of-fold
predictions from the without-fru/dsx gradient boosting model, and each
neuron's residual (actual - predicted). Most residuals cluster near
zero (most neurons are well-explained by structure alone); a small tail
extends to |residual| > 0.25.

An unplanned observation was flagged here for Phase 3: several
isomorphic-labelled **controls** with surprisingly high actual wiring
difference (reported as 0.48-0.67). **Checked against the corrected
target on 2026-09-19: that specific lead was mostly the category-asymmetry
artifact.** Zero controls now reach 0.48; the largest control value is
0.357.

A weaker version does survive and is worth keeping on the list: 5
controls still exceed the 99th percentile of the case distribution, and
the three largest model residuals in the whole population are controls
(bodyIds 47081, 87905, 157110 — actual 0.357/0.185/0.111 against
predictions near 0.02). So "a handful of isomorphic-labelled neurons look
genuinely rewired" remains a real, if much smaller, thread — either
mislabelled, or wired differently with no dimorphism annotation at all.
It is no longer the striking effect the original number implied.

## Limitations, stated plainly

- **Coarse metric**: the wiring-difference target compares profiles over
  only 10 superclass-level categories, not fine partner identity. It
  can miss a real "same broad category, different specific partners"
  rewiring, and it can't be made finer without a much larger, slower
  connectivity pull.
- **Type-level, not neuron-level, FlyWire profiles**: individual
  male↔female neuron correspondence isn't defined across two different
  animals, so the FlyWire side is a same-type average over ≤15 sampled
  neurons, not an exact match.
- **Case-control, not full-population**: 634+634 neurons, not all ~135K
  eligible ones — a deliberate scope decision for tractability (see
  rewiring.py), not an attempt to cover the whole connectome.
- **GNN trained on a population-internal subgraph** (edges only between
  the 1,268 case/control neurons), not the full 165K-neuron connectome —
  a legitimate, common scoping choice (ego-network modeling), not a
  claim of graph completeness.
- Brain-only FlyWire comparison scope carries forward the same
  limitation flagged in `rewiring.py` and `phase1_results.md`: any
  genuinely VNC-driven rewiring is structurally invisible to this
  analysis. BANC (whole-CNS) is the planned fix, in Phase 4.

## What this unblocks

- The per-neuron residual scores are ready for Phase 3's enrichment
  analysis (neurotransmitter, hemilineage, connectivity motifs) on the
  *model-residual* population, complementing Phase 0/1's raw-annotation
  residual.
- The unplanned "surprisingly-high-difference isomorphic neurons" lead
  is worth a dedicated look before or alongside Phase 3's main analysis.
