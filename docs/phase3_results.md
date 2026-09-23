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

## A third confound — now CORRECTED (2026-09-19), not just reported

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
reflecting a specific circuit.

This is now corrected rather than merely flagged.
`partner_type_enrichment_degree_adjusted` bins neurons into quartiles of
partner-type degree and runs a **Cochran-Mantel-Haenszel** test — each
partner type is tested within degree bins, comparing residual against
explained neurons of comparable connectivity intensity, never across
degree levels. Only motifs that survive this reach the shortlist.

CMH rather than logistic regression, for a reason found empirically: a
per-type `is_residual ~ connects + log(degree)` logit was tried first and
is unusable here. Many partner types are touched by nearly all residual
and nearly no explained neurons within a stratum — perfect separation —
where an unpenalized fit will not converge and a weakly-penalized one
drives the log-likelihood to zero, manufacturing p-values near 0 and odds
ratios above 1e5. It reported *more* "significant" motifs than the
uncorrected test (353 vs 68 on `ascending_neuron`), inverting the whole
point of the correction. CMH is conditional, stays finite on sparse
tables, and degrades to "not significant" rather than to false
confidence. The pooled odds ratio carries a Haldane-Anscombe +0.5
correction so that completely-separated types get a large finite estimate
instead of an infinity that would be silently discarded.

### What survived

| superclass | degree-confounded? | marginal | degree-adjusted | survived |
|---|---|---|---|---|
| ascending_neuron | no | 68 | **25** | 37% |
| cb_intrinsic | yes | 1,397 | **1,429** | 102% |
| vnc_intrinsic | yes | 306 | **44** | 14% |

Three genuinely different outcomes:

- **`vnc_intrinsic` was mostly artifact.** 86% of its motifs vanish once
  degree is held constant — consistent with it having the worst confound
  (residual mean degree 518 vs 285). This is the result the old caveat
  correctly suspected.
- **`ascending_neuron` narrows but holds.** 25 of 68 survive, and the
  headline hits (`INXXX230`, `INXXX058`, `INXXX306`, `DNpe030`) are among
  them, now with credible odds ratios of 6-23 rather than the marginal
  test's 30-45. This stratum had no detected degree confound, so a modest
  reduction from a stricter conditional test is what you would expect.
- **`cb_intrinsic` does NOT collapse.** Its motifs survive at 102% —
  conditioning on degree slightly *increases* detections, because
  removing degree as a source of variance sharpens the within-bin
  comparison in a stratum with n=1,551. **The old "read `cb_intrinsic`
  with a degree caveat" warning is therefore resolved: degree does not
  explain these associations.** What remains notable is their sheer
  breadth — 1,429 of 4,860 tested types (29%) — which reads more like a
  broad population-level connectivity difference than a specific circuit
  motif, and should be described that way.

### Empirical null for the motif counts (added 2026-09-23)

BH controls the false-discovery rate under assumptions about how the
test statistics co-vary. Those assumptions are shaky here: partner types
are heavily correlated — neurons that touch one type tend to touch
related ones — so the effective number of independent tests is well
below the nominal count, and BH could be optimistic by an unknown
margin. The cluster result got an empirical null; the much larger motif
table did not, which left the biggest result carrying the weakest
statistical backing.

`motif_permutation_null` fixes that by asking directly: hold the
connectivity matrix and the degree bins fixed, shuffle which neurons
carry the residual label (preserving the residual set's size), and count
how many "significant" motifs the identical machinery produces from
noise. 1,000 shuffles per stratum.

| stratum | observed | expected by chance | null 95th pct | null max | empirical p |
|---|---|---|---|---|---|
| ascending_neuron | **25** | 0.0 | 0 | 2 | 0.001 |
| cb_intrinsic | **1,429** | 0.7 | 4 | 24 | 0.001 |
| vnc_intrinsic | **44** | 0.1 | 0 | 54 | 0.002 |

The motif tables are not BH artifacts. Across all three strata the
expected number of false positives is **under one**, against observed
counts of 25 to 1,429. Whatever else is arguable about these results,
"this is just multiple testing" is not.

One honesty note on `vnc_intrinsic`: its null mean is 0.1 but its
maximum is 54, above the observed 44. The null is heavy-tailed there —
n=66 with a 33/33 split means an unlucky shuffle can occasionally
produce a large count. The empirical p of 0.002 already accounts for
this (2 of 1,000 shuffles reached 44), but the stratum is small enough
that its 44 motifs deserve less weight than the p-value alone suggests.

Making this feasible required rewriting CMH to run over all partner
types at once as matrix operations. The per-type statsmodels loop takes
about a minute on cb_intrinsic's 4,860 types, so 1,000 shuffles would
have been ~16 hours; vectorized it is ~25ms, and the whole Phase 3 run
including 3,000 permutations finishes in under six minutes. The
vectorized statistic is pinned equal to statsmodels'
(`test_vectorized_cmh_matches_statsmodels`) — which immediately caught
that statsmodels' `test_null_odds()` defaults to *no* continuity
correction, so an initially-corrected implementation would have been
measuring a slightly different, more conservative test than the one
being validated.

## Results by stratum

### ascending_neuron — the cleanest result (no detected confound)

68/1,902 partner types significant marginally, **25 after degree
adjustment**; strongest hits are specific interneuron types
(`INXXX230`, `INXXX058`, `INXXX306`, degree-adjusted odds ratios 17-23,
marginal 30-45) — see
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

**Empirical null (added 2026-09-20).** The BH p-value above assumes
cluster assignment is independent of the residual labels, which it is
not — the clusters were derived from the same connectivity data and the
resolution was picked by a sweep, so the parametric value is optimistic
by an unknown amount. `docs/project_plan.md` section (e) asked for a
degree-preserving null for exactly this reason and it had never been
built.

`label_permutation_null` holds the graph and its clustering **fixed**
and shuffles which neurons carry the residual label, 1,000 times. That
preserves every degree, every community and the residual set's size
exactly, and asks the right question — "do residual neurons concentrate
in one cluster more than chance", where chance means the same graph with
the labels scattered.

| | value |
|---|---|
| observed max odds ratio | **2.68** |
| null mean | 1.36 |
| null 95th percentile | 2.04 |
| shuffles reaching the observed value | **11 / 1,000** |
| **empirical p** | **0.012** |

The finding survives, and the empirical p (0.012) lands essentially on
top of the parametric p_adj (0.012) — so in this case the parametric
assumption was not inflating anything. That agreement is worth more than
either number alone, because it was not guaranteed.

### cb_intrinsic — largest stratum, and the degree caveat is now resolved

1,397/4,860 partner types significant marginally, **1,429 after degree
adjustment** — the associations survive holding connectivity intensity
constant, so degree does not explain them. Top hits (`DNg96`,
`SMP204/361`, `APL`, `MBON33`, several `LAL*`/`SMP*`/`CB*`
central-complex-and-mushroom-body-adjacent types) span diverse
central-brain neuropils, not one obviously redundant region.

The remaining caveat is breadth, not confounding: 29% of all tested
partner types come out significant, which is more consistent with
`cb_intrinsic` residual neurons differing broadly in connectivity than
with a specific circuit signature. Clustering here is a clean null, and confirmed as
such by the sweep: 0/5 resolutions found any significant cluster (at
resolution 0.3 it splits into 6 clusters and still none are enriched for
residual) — this isn't a case of "the right resolution wasn't tried,"
there's genuinely no cluster-level structure here to find.

### vnc_intrinsic — small, and mostly degree artifact

306/1,019 significant marginally, **44 after degree adjustment** — 86%
vanish. This stratum had the worst confound (residual mean degree 518 vs
285, nearly 2x), and the correction confirms that most of its apparent
motif enrichment was that confound rather than circuit-specific signal.
Clustering: significant at only 1/5 swept
resolutions (0.3, where it first splits into 2 clusters — n=66 is small
enough that this could easily be a small-sample artifact of that specific
split) — excluded from the shortlist for not being robust, not because
nothing was found at any resolution.

## Consolidated shortlist

`data_processed/phase3_candidate_shortlist.csv` (**1,531 rows**, was
1,802 before the degree correction) pools:

| component | rows |
|---|---|
| `connectivity_motif[cb_intrinsic]` | 1,429 |
| `connectivity_motif[vnc_intrinsic]` | 44 |
| `connectivity_motif[ascending_neuron]` | 25 |
| `annotation_*[stratum]` (hemilineage, neuromere, NT, side) | 32 |
| `connectivity_cluster[ascending_neuron, res=0.1]` | 1 |

**Every motif row on this list has passed the CMH degree adjustment**, so
`degree_confounded` is `False` for all 1,498 of them rather than `True`
for 1,703 as before. The 33 `None` entries are the clusters and the
annotation dimensions, where the degree check does not apply (clusters
use L2-normalized profiles and single-label enrichment; annotation labels
are not connectivity at all) — left unclaimed rather than asserted either
way.

Two further changes closed the old "least resolved" tier entirely:

- **Phase 1's annotation dimensions are no longer folded in marginally.**
  They were previously copied straight from Phase 1, which computed them
  over the whole dimorphic population and predates the `ol_intrinsic`
  saturation discovery — carrying exactly the regional confound this
  phase had fixed for motifs and clusters. They are now **re-tested
  within each stratum** and named `annotation_<dim>[<stratum>]`. That
  reduces them from 30 marginal hits to 32 stratified category-in-stratum
  hits, and the surviving ones (e.g. `FLAa3`, `LALa1_anterior`,
  `VLPl&p2_lateral` within `cb_intrinsic`) are now region-controlled.
- **Unassigned-lineage placeholders are excluded.**
  `docs/phase1_results.md` flagged that `itoleeHl`'s largest hit was
  `putative_primary` (n=74, "not yet confidently assigned a lineage") and
  `trumanHl`'s was literally `TBD` (n=44), and asked for a base-rate check
  before trusting them. A neuron missing both a fru/dsx call and a
  confident lineage call plausibly just means "under-annotated neuron",
  not "biologically distinct lineage", so these are now dropped from the
  shortlist outright (4 placeholder categories excluded), not ranked as
  biology.

Read in two tiers now rather than three:

1. **Best supported**: `ascending_neuron` motifs and its cluster —
   stratified, degree-adjusted, and resolution-robust.
2. **Real but broad**: `cb_intrinsic` motifs (degree-adjusted, so no
   longer confound-caveated, but 29% of all tested types — a
   population-level difference rather than a specific circuit) and the
   much smaller `vnc_intrinsic` remainder.

## Following the lead: what ARE the surviving ascending_neuron partners?

The 25 degree-adjusted partner types map to 123 MaleCNS neurons. Rather
than leave them as opaque type codes, characterized directly from the
annotations (2026-09-19):

- **71% are VNC local interneurons** (87/123 `vnc_intrinsic`, plus 32
  `ascending_neuron` and 4 `descending_neuron`). So fru/dsx-negative
  ascending neurons preferentially talk to VNC-local circuitry, not to
  brain targets.
- **Soma positions span the whole VNC**, thoracic (T1-T3, n=72) *and*
  abdominal (A1-A7, n=47). The abdominal component is the more
  interesting half: abdominal neuromeres house the circuitry for
  copulation and egg-laying, which is where sexual dimorphism is
  expected on prior biology.
- **The partners are themselves fru/dsx-unannotated — every single one.**
  0 of 123 carry a fru/dsx call, against 3.5 expected at the base rate
  for the same superclasses (2.86%), Fisher p=0.028. That is a real
  depletion, and it is the most suggestive thing in this phase: the
  residual is not a set of isolated neurons hanging off a
  fru/dsx-positive backbone, it connects into a sub-network that the
  genetic switch does not appear to touch either. **Read with the small
  expected count in mind** — p=0.028 on 3.5 expected events is
  suggestive, not strong, and would need a larger matched sample to
  firm up.
- **There is no neurotransmitter signature, and the obvious-looking one
  is an artifact of small numbers.** The strongest 8 partners skew
  GABAergic (5/8), which invites a tidy "inhibitory motif" story — but
  across all 123 partner neurons the mix is 50.4% acetylcholine / 48.0%
  GABA, essentially identical to the `vnc_intrinsic` background of
  42.2% / 40.4%. Checked precisely because the top-8 pattern looked
  compelling; it does not survive.

The remaining half of this lead — literature and anatomical follow-up on
the specific `INXXX*`/`DNpe030` types — is genuinely outside what the
connectome data alone can settle and is left as future work.

## Honest bottom line

Per the plan's own framing ("if nothing clusters, report that
honestly — still a result"): something *does* cluster, but only cleanly
in one of three testable strata, and that one held up under every check
this phase could throw at it — not degree-confounded, and the cluster
split is stable across a resolution sweep rather than an artifact of one
parameter choice. The `ascending_neuron` finding is a real,
non-tautological, degree-adjusted, resolution-robust candidate
second-driver signature — specific interneuron partners plus a
distinguishable connectivity cluster, both statistically defensible.

The larger `cb_intrinsic` result now also survives degree adjustment,
which is a meaningful upgrade from "not separable from a connectivity-
intensity confound" — but it shows no cluster-level structure at any
swept resolution and is very broad (29% of tested types), so it reads as
a diffuse population-level connectivity difference rather than a specific
circuit signature. `vnc_intrinsic`, by contrast, largely dissolved under
the same correction.

**This bears on the project's central open question** (biology vs.
annotation ascertainment bias, see `docs/phase1_results.md`).
`cb_intrinsic` sits in the central brain — the region that received ~75%
of the fru/dsx annotation effort, and therefore the region where
"residual = under-annotated" is *least* plausible. Finding that its
residual neurons have connectivity differences which survive both
regional stratification and degree adjustment is evidence that the
residual is not purely an annotation-coverage artifact. It does not
settle the question for the poorly-annotated periphery, where the
ambiguity remains wide open.

This phase's deliverable is exactly what PHASE 3 asks for — "a ranked
shortlist of candidate signatures with effect sizes and corrected
p-values" for further investigation — not a confirmed biological
mechanism.

## What this unblocks

- The `ascending_neuron` candidate signature is concrete enough to take
  into Phase 4 (BANC cross-validation) or targeted literature/anatomical
  follow-up on the specific `INXXX*`/`DNpe030` partner types.
- `fruresidual.features.motifs.identify_mixed_strata` and
  `check_degree_confound` are reusable for any future stratified
  enrichment work in this project, including revisiting Phase 1's
  un-stratified marginal dimensions.
