# The optic-lobe gap: three routes tried, three closed

The project's evidence has one hole, and it is the biggest one.

Phase 6 (`docs/annotation_crosscheck_results.md`) established that the
residual is not merely unannotated — but only for the central brain and
descending neurons. It was explicitly silent about the **optic lobe**,
where **372 of the 576** matched residual neurons sit, and where every
annotation source has near-zero coverage (Codex 0.01%, curated 0.08%).

Three routes into that gap were tried. All three are now closed, each
for a different and independently verified reason. This document exists
so none of them is attempted again.

## Route 1 — Özel optic-lobe scRNA-seq atlas: closed

Assessed in full in `docs/scrnaseq_route_assessment.md`. Three
independent failures: `fru` is not measurable by 3′ 10x (its promoters
differ at the 5′ end, and only P1 is the sex-specific transcript); the
only `dsx`+ clusters are **glia** (`repo`/`wrapper`/`alrm`/`moody`, zero
neuronal markers), and glia are not in the connectome; and the
cluster→cell-type mapping is not in any public file.

## Route 2 — Fly Cell Atlas head: closed, with a useful by-product

`scripts/run_fca_crosscheck.py`, run 2026-09-23 against
`v2_fca_biohub_head_10x_raw.h5ad` (100,527 cells × 13,056 genes).

FCA is a better candidate than Özel on its face: it annotates cells with
strings that **embed connectome type names** ("medullary intrinsic
neuron Mi1", "transmedullary neuron Tm9"), so the join can be made and
then validated against the connectome's own vocabulary rather than
assumed. Of 83 FCA annotations, **36 resolve to a real connectome cell
type** — `Mi1/4/9/15`, `Tm1/2/9/20/29`, `L1`–`L5`, `T2/T3/T4/T5`,
`Dm3/8/9/10/11/12`, `Pm2/4`, `TmY4/5a/8/14`, `LC10/12/17`, `R7`, `R8`.
Both sexes are present (49,105 female / 47,409 male).

**It still cannot answer the question, because the overlap is zero.**
All 36 matched types are `not_dimorphic` — none of them is in the
2,368-neuron dimorphic population at all. FCA samples the abundant,
canonical optic types; the dimorphic optic types are different ones
(`Cm26`, `Mi20`, `Tm26` among the unmatched). The join works; it just
lands entirely beside the population of interest.

**The by-product is worth keeping**, because it independently
corroborates the isoform argument that closed Route 1, in a separate
dataset:

| gene | mean % of cells expressing, across 36 optic types |
|---|---|
| `fru` | **42.9%** (median 43.6%, max 89.0%) |
| `dsx` | **0.8%** (median 0.4%, max 9.8%) |

`fru` transcript is near-ubiquitous in the optic lobe — reproducing
Özel's ~48% baseline in independent data — while `dsx` is genuinely
rare. Against a connectome annotation putting *fru* at 0.01% in the
optic lobe, that gap is not a discrepancy to explain; it is confirmation
that 3′ *fru* counts and connectome *fru* calls are different
quantities.

The sex-differential check points the same way. If total *fru* counts
tracked FruM, male cells should read consistently higher. They do not:
the mean male−female difference is **−6.0** percentage points (slightly
*higher in females*), with large swings in both directions (`Pm4` −28,
`Mi1` +22) that look like sampling noise rather than a sex-specific
signal. `dsx` shows +0.3 points, essentially nothing.

## Route 3 — MAOL cross-dataset type table: closed

`MAOL_olmatching.txt` maps optic types across
`OL_type ↔ Matsliah ↔ Schlegel ↔ hemibrain`, so it looked like a way to
rescue optic neurons that fail to match FlyWire. It does not, and
checking why corrected a mistaken premise.

**The attrition is not where it was assumed to be.** Of 1,603 unmatched
dimorphic neurons:

| superclass | unmatched | share of attrition | no `flywireType` at all |
|---|---|---|---|
| `cb_intrinsic` | 1,293 | **81%** | 1,161 |
| `ol_intrinsic` | 118 | 7% | 118 |
| `ascending_neuron` | 101 | 6% | 101 |
| `vnc_intrinsic` | 66 | 4% | 66 |

The optic lobe is 7% of the problem, not the bulk of it. And more
decisively: **every unmatched optic neuron has no `flywireType`
whatsoever** — the bridge is absent, not misspelled. A type-translation
table can only repair naming mismatches; it cannot invent a
cross-match that was never made.

Confirmed directly against MAOL's own table: the three unmatched optic
types (`Cm26`, `Mi20`, `Tm26`) are all present in it and all three are
marked **`matched_as = "unmatched"`**. MAOL could not match them to
FlyWire either. Neurons rescuable by this route: **0**.

## What is left

Nothing available. The optic-lobe portion of the residual has no
independent *fru*/*dsx* evidence obtainable from any public dataset
identified so far, and the reason is structural rather than a matter of
effort:

- 3′ scRNA-seq measures the wrong *fru* transcript (Routes 1 and 2).
- The connectome's optic dimorphic types are not sampled by the
  available atlases (Route 2).
- They have no female cross-match to compare against (Route 3).

What would work is unchanged from the earlier assessment: **5′ or
full-length scRNA-seq** able to resolve the *fru* promoter, a published
***fru*-P1-specific analysis**, or **fru-P1-GAL4 driver-line expression
data** — which would at least distinguish "not expressed there" from
"never examined there."

Phase 6's conclusion stands exactly as written: it covers the central
brain and descending neurons, and says nothing about the optic lobe.
That silence is now a measured boundary rather than an untested gap.
