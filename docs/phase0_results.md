# Phase 0 kill-gate — result: ALIVE

Run: 2026-09-17, via `scripts/run_phase0.py`, on
`body-annotations-male-cns-v1.0-minconf-0.5.feather` (Traced neurons only,
n=165,122).

## What we expected vs. what we found

The plan (docs/project_plan.md section e, risk #2) anticipated a hard
type-vs-neuron granularity join between MaleCNS dimorphism calls and
fru/dsx annotations. That join turned out to already be done: the bulk
MaleCNS annotation file ships `dimorphism` and `fruDsx` per neuron
(bodyId), not per type. No custom rollup was needed — just a decision on
how to treat `fruDsx`'s confidence suffix (`_high`/`_low`), which is the
`any_positive` vs `high_confidence_only` rule in
`src/fruresidual/features/fru_dsx_join.py`.

## The gate

| Rule | dimorphic/male-specific neurons | unexplained (no fru/dsx) | % unexplained | top-2 superclass concentration |
|---|---|---|---|---|
| any_positive | 2,368 | 921 | 38.9% | 76.5% (ol_intrinsic + cb_intrinsic) |
| high_confidence_only | 2,368 | 1,079 | 45.6% | 79.6% (ol_intrinsic + cb_intrinsic) |

Both rules pass the gate criterion (residual is both non-trivial in size
and structured, not spread evenly across superclasses) — **verdict: ALIVE**,
robust to the aggregation-rule sensitivity check the plan called for.

Sanity check against the published number: the *Cell* 2026 paper reports
39.7% of dimorphic neurons are fru−/dsx− centrally; our `any_positive`
figure (38.9%) lands almost exactly on that, which is a good sign the
pipeline is measuring the same thing the paper measured.

## An early, unplanned lead

The residual isn't spread the way the paper's overall dimorphic-neuron
population is (concentrated in higher-order centers per the plan's Phase-1
expectations). Instead the *unexplained* subset is dominated by
`ol_intrinsic` (optic lobe intrinsic — visual system) at ~50% of the
residual, with `cb_intrinsic` a distant second (~20-25%). This wasn't
predicted by the plan and is worth treating as the first real Phase-1
question: is the fru/dsx-blind residual specifically a visual-system
phenomenon? Not a conclusion yet — no enrichment stats run, just where the
raw counts land — but a concrete thread to pull in Phase 1.

## What this unblocks

- Phase 1 (atlas) and Phase 2 (predictive model) can proceed.
- `fruDsx` and `dimorphism` are usable directly per-neuron from this one
  file — no FlyWire cross-match needed just to get these two labels for
  MaleCNS neurons. FlyWire's own `flywire_annotations` table (also
  fetched, 139,248 neurons, same-shaped `dimorphism`/`fru_dsx` columns) is
  still needed for the actual male<->female wiring comparison in Phase 2,
  and for Phase 4's BANC cross-validation.
