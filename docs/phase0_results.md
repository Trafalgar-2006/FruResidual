# Phase 0 kill-gate — result: ALIVE

Run: 2026-09-17, via `scripts/run_phase0.py`, on
`body-annotations-male-cns-v1.0-minconf-0.5.feather` (Traced neurons only,
n=165,122). Verified by `tests/test_phase0_integration.py` (skipped if the
data file isn't downloaded).

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

| Rule | dimorphic/male-specific neurons | unexplained (no fru/dsx) | % unexplained |
|---|---|---|---|
| any_positive | 2,368 | 921 | 38.9% |
| high_confidence_only | 2,368 | 1,079 | 45.6% |

**Structure is now decided by a real statistical test**, not a heuristic
threshold: Fisher's exact test per superclass (residual vs. the full
dimorphic/male-specific population it's drawn from), Benjamini-Hochberg
corrected across superclasses, requiring both significance (p_adj < 0.05)
and a non-trivial effect size (odds ratio >= 2) —
`fruresidual.eval.enrichment.category_enrichment`, the same method the
plan specifies for Phase 3, applied at a lighter scope here. Result
(any_positive rule):

| superclass | % of residual | odds ratio | p_adj |
|---|---|---|---|
| ol_intrinsic | 53.2% | 3,291 | 1.6e-247 |
| descending_neuron | 9.7% | 9.3 | 2.5e-22 |
| ascending_neuron | 9.1% | 3.0 | 5.6e-9 |
| sensory_descending | 0.7% | 20.6 | 8.6e-3 |

`ol_intrinsic`'s odds ratio (~3,291, Haldane-Anscombe corrected since the
raw table has a near-zero cell) reflects that essentially **all**
optic-lobe-intrinsic dimorphic neurons lack a fru/dsx annotation — not a
mild skew, a near-total one. `cb_intrinsic` — despite being the
second-largest raw count in the residual — is **not** statistically
enriched; its share of the residual is proportional to its share of the
overall dimorphic population, not concentrated beyond that. (An earlier
version of this doc used a naive "top-2 categories by raw count" rule and
wrongly implied cb_intrinsic was part of the structure — the real test
corrects that.)

Both rules pass the gate criterion (residual is non-trivial AND
statistically structured) — **verdict: ALIVE**, robust to the
aggregation-rule sensitivity check the plan called for.

### Sanity check against the published number — investigated, not fully resolved

The paper (Berg et al., *Cell* 2026) reports two separate central-brain
statistics: 90.4% of male-specific neurons and 60.3% of dimorphic neurons
are fru+/dsx+. Restricting our data to central-brain superclasses only
(`cb_*`: 1,553 of 2,368 dimorphic/male-specific neurons):

| category | our result (per-neuron) | our result (per-type) | paper |
|---|---|---|---|
| male-specific | **90.4% positive / 9.6% negative — exact match** | 87.2% / 12.8% | 90.4% / 9.6% |
| sexually dimorphic | 79.0% / 21.0% | 67.6% / 32.4% | 60.3% / 39.7% |

Independent cross-checks confirm the pipeline itself is correct:
- Central-brain fru+ rate (10.05% any-confidence, 5.31% high-confidence)
  closely matches the paper's cited 10.6%/~5% for *all* central-brain
  neurons (not just dimorphic ones) — confirming our `cb_*` superclass
  filter is the right definition of "central brain."
- Absolute fru/dsx counts match the paper exactly once co-expressing
  neurons are counted toward both totals: 4,600 fru-only + 258 coexpress
  = **4,858** (paper: 4,858); 154 dsx-only + 258 coexpress = **412**
  (paper: 412); same exact match at high-confidence (2,804, 331). Pinned
  as a regression test in `test_phase0_integration.py`.
- **male-specific** matches the paper exactly at the per-neuron level
  (90.4%/9.6%, not just close — identical to one decimal place).

So the pipeline, the central-brain definition, and the male-specific
statistic are all independently confirmed correct. The gap is isolated
specifically to the "sexually dimorphic" (cross-sex-matched) category,
and neither per-neuron (79.0%) nor per-type (67.6%) accounting reproduces
the paper's 60.3% exactly — ruling out "it's just a type-vs-neuron
counting difference" as the complete explanation, though type-level
accounting moves in the right direction. The most likely remaining
explanation: our downloaded `male-cns:v1.0` release (June 2026) may be a
different annotation snapshot than whatever internal revision fed the
published Sept 2026 figures — the plan itself already documents that the
Oct 2025 bioRxiv preprint and the final *Cell* paper report different
total counts (114 vs 138 dimorphic types), so a third, slightly different
snapshot in the public v1.0 data release is plausible. Unresolved, but
narrowed from "unknown" to a specific, evidenced hypothesis — and it does
not affect the gate verdict, which doesn't depend on matching the paper's
number.

## An early, unplanned lead

The unexplained residual is dominated by `ol_intrinsic` (optic lobe
intrinsic neurons) — statistically confirmed above, not just a raw-count
artifact. This wasn't predicted by the plan (which expected dimorphism
generally to concentrate in higher-order centers) and is worth treating
as the first real Phase-1 question: is the fru/dsx-blind residual
specifically a visual-system phenomenon? The `descending_neuron` and
`ascending_neuron` enrichment (both VNC-brain connector types) is a
second, smaller lead worth carrying into Phase 1.

## What this unblocks

- Phase 1 (atlas) and Phase 2 (predictive model) can proceed.
- `fruDsx` and `dimorphism` are usable directly per-neuron from this one
  file — no FlyWire cross-match needed just to get these two labels for
  MaleCNS neurons. FlyWire's own `flywire_annotations` table (also
  fetched, 139,248 neurons, same-shaped `dimorphism`/`fru_dsx` columns) is
  still needed for the actual male<->female wiring comparison in Phase 2,
  and for Phase 4's BANC cross-validation.
- `fruresidual.eval.enrichment.category_enrichment` is a reusable,
  general-purpose enrichment-testing utility — Phase 3's planned
  enrichment analysis (neurotransmitter, hemilineage, connectivity
  motifs) can call the same function.
