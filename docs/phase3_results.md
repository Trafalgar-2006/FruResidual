# Phase 3: turning the residual into a hypothesis

Run: 2026-09-18, via `scripts/run_phase3_data.py` (fetches
`data_interim/phase3_partner_type_profile.feather` and
`phase3_full_superclass_profile.feather` over the full 2,368-neuron
dimorphic population — ~1 minute total, both server-side aggregated;
skips already-cached files) then `scripts/run_phase3.py`.

## This phase found and fixed three real methodological gaps before any result could be trusted

**1. Regional confound (tautology).** The first version of this analysis
compared the residual set (921 neurons) against the whole dimorphic
population (2,368) directly. Result: 3,450/8,115 partner types "significant."
That number was wrong in spirit even where technically computed: **all
490 `ol_intrinsic` dimorphic neurons are residual — zero explained
neurons in that superclass.** Any partner type an optic-lobe neuron
happens to touch will look "residual-enriched" purely because being in
that superclass already implies being residual — not because of any
circuit-specific signal. Fix: `identify_mixed_strata` restricts analysis
to superclasses with ≥20 of *both* residual and explained neurons —
`ascending_neuron`, `cb_intrinsic`, `vnc_intrinsic`. `ol_intrinsic` and
four smaller superclasses are excluded (see table below), not analyzed
as if they had a real internal comparison to make.

**2. Multi-label statistical bug.** Even after stratifying, one category
(`DNg96` in `cb_intrinsic`) reported "0.04% of subset" for a partner
type actually touched by 29/215 = 13.5% of residual neurons. Cause: the
first `partner_type_enrichment` reused `fruresidual.eval.enrichment.category_enrichment`,
which assumes one category value per entity (correct for superclass,
hemilineage, Leiden cluster). Partner type is **multi-label** — one
neuron connects to dozens of types simultaneously, contributing many
rows — so `len(series)` silently counted *edges* (~80,000) instead of
*neurons* (215), corrupting every odds ratio and p-value. Fixed with a
dedicated neuron-set-membership implementation
(`partner_type_enrichment`, no longer built on `category_enrichment`);
regression-tested (`test_partner_type_enrichment_pct_is_neuron_level_not_edge_level`).

**3. Unvalidated clustering resolution.** Found during a full-project
audit (2026-09-18): `clustering.py`'s own docstring said not to trust a
single Leiden resolution blindly on real data and to run a sensitivity
sweep, but the script had drifted to a single fixed `resolution=0.1`
picked without ever validating it against the real population — a
documentation/implementation mismatch, and exactly the kind of thing
that could have silently made the headline `ascending_neuron` finding an
artifact of one arbitrary parameter choice. Fixed by sweeping
`[0.02, 0.05, 0.1, 0.2, 0.3]` per stratum and only trusting a cluster
result that's significant at a majority of swept resolutions (≥3/5).
Result of actually doing this: it **confirms** the `ascending_neuron`
finding (significant at 3/5, a stable 2-cluster split from resolution
0.1 upward — not fragile) and **correctly excludes** `cb_intrinsic`
(0/5 — a clean null, not a resolution artifact) and `vnc_intrinsic` (1/5,
only at the extreme edge of the swept range) from the shortlist.

## Stratified population

| superclass | n | residual | explained | % residual |
|---|---|---|---|---|
| ascending_neuron | 131 | 84 | 47 | 64.1% |
| cb_intrinsic | 1,551 | 215 | 1,336 | 13.9% |
| vnc_intrinsic | 66 | 33 | 33 | 50.0% |

Excluded (saturated — no internal residual/explained variance — or too
small): `ol_intrinsic`, `descending_neuron`, `cb_motor`,
`sensory_descending`, `visual_centrifugal`, `visual_projection`,
`vnc_motor`.

## A third confound, found and reported (not corrected)

Checking whether residual and explained neurons differ in raw
connectivity intensity before trusting the motif counts:

| superclass | residual mean degree | explained mean degree | p (Mann-Whitney) | confounded? |
|---|---|---|---|---|
| **ascending_neuron** | 425 | 389 | 0.855 | **no** |
| cb_intrinsic | 399 | 322 | 0.0075 | **yes** |
| vnc_intrinsic | 518 | 285 | 1.7e-6 | **yes** |

In `cb_intrinsic` and `vnc_intrinsic`, residual neurons are
systematically more connected overall (not just to specific partners) —
a higher-degree neuron is more likely to touch any given partner type by
chance, which can inflate apparent motif enrichment broadly without
reflecting a specific circuit. This was **not** corrected (would need
degree-matched sampling or a covariate-adjusted model, e.g. logistic
regression instead of marginal Fisher tests per category — left as
future work) — it is reported and every downstream result from those two
strata is flagged with it, rather than presented as clean.

## Results by stratum

### ascending_neuron — the cleanest result (no detected confound)

68/1,902 partner types significant after BH correction; strongest hits
are specific interneuron types (`INXXX230`, `INXXX058`, `INXXX306`,
odds ratios 30-45) — see
`reports/figures/phase3/motif_enrichment_unconfounded_only.png`. Leiden
clustering, swept across 5 resolutions, finds a **robust** (significant
at 3/5, stable from resolution 0.1 upward) 2-cluster split; residual
neurons are significantly concentrated in one of them (60.7% of residual
vs 33% of background, odds ratio 2.68, p_adj=0.012 at resolution 0.1 —
see `reports/figures/phase3/cluster_scatter_ascending_neuron.png` and
`data_processed/phase3_cluster_resolution_sweep.csv` for the full sweep).
**This is the strongest candidate finding from Phase 3**: fru/dsx-negative
ascending neurons connect to a distinguishable set of specific
interneuron partners and cluster together connectivity-wise, independent
of the degree confound found elsewhere, and the cluster split isn't an
artifact of one arbitrary resolution choice.

### cb_intrinsic — largest stratum, but degree-confounded

1,397/4,860 partner types nominally significant — read with the degree
caveat above. Top hits (`DNg96`, `SMP204/361`, `APL`, `MBON33`, several
`LAL*`/`SMP*`/`CB*` central-complex-and-mushroom-body-adjacent types)
span diverse central-brain neuropils, not one obviously redundant
region — more plausible as partially real than the original
`ol_intrinsic` tautology, but not separable from the degree effect with
this analysis alone. Clustering here is a clean null, and confirmed as
such by the sweep: 0/5 resolutions found any significant cluster (at
resolution 0.3 it splits into 6 clusters and still none are enriched for
residual) — this isn't a case of "the right resolution wasn't tried,"
there's genuinely no cluster-level structure here to find.

### vnc_intrinsic — small, also degree-confounded

306/1,019 significant, same caveat, even more pronounced (residual mean
degree 518 vs 285 — nearly 2x). Clustering: significant at only 1/5 swept
resolutions (0.3, where it first splits into 2 clusters — n=66 is small
enough that this could easily be a small-sample artifact of that specific
split) — excluded from the shortlist for not being robust, not because
nothing was found at any resolution.

## Consolidated shortlist

`data_processed/phase3_candidate_shortlist.csv` (1,802 rows) pools:
region-stratified motifs, region-stratified clusters (only from strata
where the cluster finding was robust across the resolution sweep — just
`ascending_neuron`, 1 row), and Phase 1's already-computed marginal
(unstratified) hemilineage/neurotransmitter/neuromere/side dimensions.
The `degree_confounded` column is `True`/
`False` for motifs (directly checked), `None` for clusters (different
statistical basis — L2-normalized profiles partially mitigate but don't
rule out a degree effect) and for the folded-in Phase 1 dimensions
(**not** stratification-checked at all — Phase 1 predates discovering
the `ol_intrinsic` saturation issue, so those entries carry the same
un-resolved regional-confound risk this phase fixed for motifs/clusters).
Read the shortlist in tiers, not as one flat ranked list:

1. **Trust most**: `ascending_neuron` motifs/cluster — stratified, not
   degree-confounded.
2. **Directionally interesting, confound-caveated**: `cb_intrinsic`/
   `vnc_intrinsic` motifs — stratified but degree-confounded.
3. **Least resolved**: `phase1_marginal_*` entries — neither
   stratification- nor degree-checked; included for completeness, not
   because they're more trustworthy than their p-values alone suggest.

## Honest bottom line

Per the plan's own framing ("if nothing clusters, report that
honestly — still a result"): something *does* cluster, but only cleanly
in one of three testable strata, and that one held up under every check
this phase could throw at it — not degree-confounded, and the cluster
split is stable across a resolution sweep rather than an artifact of one
parameter choice. The `ascending_neuron` finding is a real,
non-tautological, non-degree-confounded, resolution-robust candidate
second-driver signature — specific interneuron partners plus a
distinguishable connectivity cluster, both statistically defensible. The
larger `cb_intrinsic` result is suggestive at the motif level but shows
no cluster-level structure at all (a clean null across every swept
resolution) and isn't separable from a genuine connectivity-intensity
confound with the tools built here. This phase's
deliverable is exactly what PHASE 3 asks for — "a ranked shortlist of
candidate signatures with effect sizes and corrected p-values" for
further investigation — not a confirmed biological mechanism.

## What this unblocks

- The `ascending_neuron` candidate signature is concrete enough to take
  into Phase 4 (BANC cross-validation) or targeted literature/anatomical
  follow-up on the specific `INXXX*`/`DNpe030` partner types.
- `fruresidual.features.motifs.identify_mixed_strata` and
  `check_degree_confound` are reusable for any future stratified
  enrichment work in this project, including revisiting Phase 1's
  un-stratified marginal dimensions.
