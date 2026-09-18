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

Sanity check (regression-pinned in `test_phase2_integration.py`): cases'
mean wiring-difference (0.0146) is higher than controls' (0.0066) — the
metric points the right direction before any model touches it.

## The ablation: fru/dsx adds ~nothing on top of structure

| model | R² without fru/dsx | R² with fru/dsx | difference |
|---|---|---|---|
| ridge | 0.327 ± 0.176 | 0.334 ± 0.172 | +0.007 |
| gradient_boosting | **0.507 ± 0.178** | 0.499 ± 0.182 | -0.008 |

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

## GNN beats the baselines

| model | test R² | test MAE |
|---|---|---|
| ridge (without fru/dsx) | 0.327 (CV mean) | 0.020 |
| gradient_boosting (without fru/dsx) | 0.507 (CV mean) | 0.009 |
| **GNN** (population-internal subgraph, 7,441 edges) | **0.735** | 0.010 |

The GNN — a 2-layer GraphSAGE trained full-batch (population is only
1,268 nodes, easily fits a 4GB GPU, no neighbor sampling needed) on the
634 dimorphic + 634 isomorphic neurons and the real connectome edges
*among* them — clearly beats both baselines. This makes sense: the
baselines see only per-neuron features, while the GNN also sees who
each neuron actually talks to within the population. Per the plan's own
risk mitigation ("if the GNN doesn't beat gradient boosting, report
that") — here it does, honestly, not by construction.

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

An unplanned observation worth flagging for Phase 3: several of the
largest-residual neurons are **controls** (labeled isomorphic) whose
*actual* wiring-difference is surprisingly high (0.48-0.67) despite the
model predicting near-zero. This wasn't the residual signal Phase 0/1
was built around (that one is about fru/dsx-negative *dimorphic*
neurons), but it's a distinct, concrete lead: either a handful of
isomorphic-labeled neurons are mislabeled, or they're genuinely
wired differently despite lacking any dimorphism annotation at all — a
second, independent thread pointing at the same kind of gap the whole
project is chasing.

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
