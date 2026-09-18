# Phase 4: cross-dataset validation against BANC

Run: 2026-09-19, via `scripts/run_phase4.py`.

## The plan changed on contact with a real permission wall

The plan (docs/project_plan.md section d, PHASE 4) called for re-deriving
Phase 2's wiring-difference metric using BANC's connectivity instead of
FlyWire's — the same cosine-distance-over-partner-profiles approach,
swapping the comparison dataset. That needs live queries against BANC's
CAVE API. Confirmed directly: our CAVE token (already working for
FlyWire) authenticates fine against BANC's datastack but gets
`403: Missing permission: view for dataset BANC` — an account-level
grant neither this project nor a retry can self-serve; BANC access is
gated separately from FlyWire's.

Rather than block the whole phase on that, this uses what BANC **does**
expose with no auth at all: a public bulk metadata table
(`banc_888_meta.feather`, confirmed at
`gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/`,
188,508 neurons × 81 columns) that ships **direct neuron-level
`malecns_match` cross-references** to MaleCNS bodyIds and **BANC's own
independent `sexually_dimorphic` classification**. That's not the same
validation the plan specified (connectivity-based, not annotation-based),
but it validates something arguably more fundamental: does an
independent research group's independent female reconstruction agree
that the neurons *we* called dimorphic actually are? The plan's actual
question — "do the same neurons/types remain unexplained" — is
answerable this way even without connectivity access.

## Matching

`malecns_match` gives 23,608 raw BANC-to-MaleCNS links within the
165,122-neuron traced population; 13,942 are already unambiguous 1:1.

The remaining ~19% are one MaleCNS neuron claimed by several BANC
fragments. Picking a winner among them arbitrarily would be
unjustifiable, but there is a subset where no winner needs picking:
when every competing fragment carries the **same** `sexually_dimorphic`
call, the ambiguity is real but cannot change the answer. 3,264 of 3,349
ambiguous neurons have unanimous fragments and are recovered; the 85
whose fragments disagree are still dropped. Total: **17,206 matches**,
of which **148 are MaleCNS-dimorphic** (up from 134 — a 10% power gain,
not the large one first guessed).

## Recall is the number that answers the Phase 4 question; precision answers a different one

| metric | value | what it means |
|---|---|---|
| recall (MaleCNS-dimorphic confirmed by BANC) | **87.2%** | of neurons *we* call dimorphic, how many does BANC also call dimorphic |
| precision (BANC-dimorphic matches ours) | 16.5% | of neurons *BANC* calls dimorphic, how many did we also call dimorphic |
| Jaccard overlap | 16.1% | low, but expected — see below |

Recall is high: **BANC independently confirms 87% of the neurons MaleCNS
calls dimorphic.** Precision is low because BANC's own survey flags far
more neurons as dimorphic overall (783 vs. 148 in this matched set) —
a real asymmetry in classification sensitivity between two independent
research groups' pipelines, not a problem with this analysis. The raw
Jaccard number alone would misleadingly read as "poor agreement" — it's
mechanically low for any rare trait (dimorphism is ~1% of neurons here)
once *either* side calls extra positives the other has no matched
counterpart for. Recall is the metric that actually answers "does BANC
confirm our calls," and it's high.

## The actual Phase 4 finding: residual is not disproportionately unconfirmed

| population | n matched | % confirmed dimorphic by BANC |
|---|---|---|
| residual (fru/dsx-unexplained) | 115 | 87.8% |
| explained (fru/dsx-positive) | 33 | 84.8% |
| **difference** | | **p = 0.768 (not significant)** |

This is the result that matters: the residual set — the neurons Phase 0
found are sex-different but not explained by fru/dsx — is confirmed
dimorphic by an independent female reconstruction at a rate
**statistically indistinguishable** from the explained set. If the
residual were mostly FlyWire-matching artifacts or MaleCNS-side
annotation noise, it should show up as *less* reliably confirmed than
the explained set. It doesn't. This directly supports the project's
central premise: the fru/dsx-unexplained residual reflects real,
independently-replicable sex differences, not an artifact specific to
the FlyWire comparison Phase 0-3 were built on.

## Two robustness checks on the headline number (added on audit, 2026-09-18)

BANC's `sexually_dimorphic` column has five states, and two of them are
easy to conflate: `isomorphic` (assessed, judged not dimorphic) and
`None` (never assessed — 70,121 of BANC's 188,508 neurons). The analysis
counts both as "not confirmed", which would understate concordance if
many matched neurons were simply unassessed. Checked directly: of the
148 matched MaleCNS-dimorphic neurons, **129 are confirmed dimorphic, 18
are genuine disagreements (BANC says isomorphic), and only 1 is
unassessed.** Restricting to assessed-only neurons moves recall from
87.2% to 87.8% — immaterial, so the headline number stands as reported.

Second: BANC ships a `proofread` flag, and unproofread reconstructions
carry less reliable annotations. **All 148 of the MaleCNS-dimorphic
matches the finding rests on are `proofread=TRUE`** — so no
reconstruction-quality caveat applies to the core result. Splitting
residual vs. explained on assessed-only neurons leaves the conclusion
unchanged (still not significantly different).

## Per-superclass breakdown — directionally consistent, but underpowered

| superclass | n residual matched | % confirmed | n explained matched | % confirmed | p |
|---|---|---|---|---|---|
| ascending_neuron | 29 | 89.7% | 12 | 100.0% | 0.543 |
| cb_intrinsic | 10 | 100.0% | 13 | 84.6% | 0.486 |
| vnc_intrinsic | 9 | 11.1% | 2 | 0.0% | 1.0 |

None reach significance (expected — n=9-29 per cell has essentially no
power), and none contradict the overall finding except `vnc_intrinsic`'s
low absolute confirmation rate, which is itself built on only 11 total
matches — too few to draw anything from. `ol_intrinsic` had fewer than 5
matches entirely and isn't reported. These are shown in
`reports/figures/phase4/concordance_bar.png` for transparency, not as
independent findings — read the overall row as the real result.

## Honest scope statement

**What this validates**: the core dimorphism classification from Phase 0
— including specifically the residual/unexplained subset — replicates
in an independently reconstructed, independently annotated female CNS.

**What this does not validate**: Phase 2's specific wiring-difference
metric (cosine distance over connectivity profiles) or Phase 3's
specific motif/cluster findings, since neither could be re-run against
BANC without connectivity access. The `ascending_neuron` stratum's
residual/explained BANC-concordance was checked here (89.7% vs 100.0%,
n too small to be meaningful) as the closest available proxy, not a
direct test of Phase 3's clustering result.

The plan also asks for "rank correlation of residual scores" between
datasets — not computed, and not silently skipped: BANC's public table
gives a categorical call (`sexually_dimorphic`: isomorphic/dimorphic/
female-specific/male-specific), not a continuous per-neuron score
comparable to Phase 2's wiring-difference values, so there's no
meaningful rank correlation to compute from what's actually available.
The categorical concordance analysis above (recall/precision/Jaccard,
residual-vs-explained comparison) is the correct analog given this
constraint, not a partial version of the originally-planned metric.

## What this unblocks

- `fruresidual.features.cross_validation` and
  `fruresidual.data.banc_client` are reusable if BANC dataset access is
  ever granted to this account — the connectivity-based re-derivation
  the plan originally specified becomes straightforward to add on top of
  this matching infrastructure.
- The project's four data-driven phases (0-4) are now complete, each
  with an independent line of evidence: Phase 0 established the residual
  exists and is structured; Phase 1 mapped where; Phase 2 showed
  structural features predict it about as well as fru/dsx does; Phase 3
  found one non-confounded circuit-level candidate signature; Phase 4
  shows the underlying classification replicates in an independent
  female dataset.
