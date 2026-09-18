# The fru/dsx-unexplained residual in the *Drosophila* connectome

*Solo research project. All numbers below are reproduced by
`scripts/run_phase0.py` through `scripts/run_phase4.py`, regression-tested
in `tests/`, and cited from the per-phase writeups (`docs/phaseN_results.md`)
this document summarizes. Run: 2026-09-17 through 2026-09-19.*

![Headline figure](../reports/figures/summary_headline.png)

## Abstract

The genetic sex-switch genes *fruitless* (*fru*) and *doublesex* (*dsx*)
are the textbook explanation for sexually dimorphic wiring in the fly
brain. Berg et al. (*Cell* 2026), releasing the first complete male
connectome, reported that *fru*/*dsx* correspondence with dimorphism is
"high but not perfect": roughly 40% of dimorphic and 10% of male-specific
neurons carry no *fru*/*dsx* annotation at all. This project treats that
unexplained set — the **residual** — as the object of study rather than
noise. Working from MaleCNS v1.0, FlyWire/FAFB, and BANC, four
independent lines of evidence are assembled: (1) the residual is large
(38.9–45.6% of dimorphic/male-specific neurons, robust to the *fru*/*dsx*
confidence-threshold choice) and concentrates in specific neuropils, most
sharply the optic lobes (95–100% residual) and less sharply the VNC/GNG
periphery (70–76%), against a well-explained central brain (14–32%); (2)
a gradient-boosting model predicts wiring difference from connectivity
structure alone about as well as one that also sees *fru*/*dsx*
(R²=0.219 vs. 0.216 — the genetic label adds nothing once structure is
in the model), while a graph neural network on the connectome subgraph
is too seed-unstable on this population to beat that baseline at all
(R²=-0.592 ± 1.000 over 10 seeds); (3) one stratum,
ascending neurons, yields a degree-adjusted, resolution-robust
connectivity-cluster signature concentrating 60.7% of its residual
neurons in one cluster (odds ratio 2.68) plus 25 partner-type motifs
that survive Cochran-Mantel-Haenszel adjustment for connectivity
intensity; (4) an independent female reconstruction (BANC)
confirms the residual and the *fru*/*dsx*-explained set as dimorphic at
statistically indistinguishable rates (86.8% vs. 82.1%, p=0.547),
arguing the residual reflects real biology rather than an annotation or
matching artifact specific to one dataset. No new gene is identified —
the contribution is a validated method and a concrete, falsifiable
hypothesis about where a second driver of brain sex differences would be
found.

## 1. Introduction

Berg et al.'s male connectome makes it possible, for the first time, to
ask a wiring-level version of an old question: how completely do *fru*
and *dsx* explain sex differences in brain wiring? The paper's own
central-brain statistics — 60.3% of dimorphic and 90.4% of male-specific
neurons are *fru*+/*dsx*+ — already imply the answer is "mostly, not
entirely," and the paper explicitly flags two possible reasons: real
biology (an uncharacterized second driver) or an artifact (incomplete
gene-expression annotation). This project does not resolve that
ambiguity by fiat; it builds a pipeline to interrogate it from four
independent angles — spatial distribution, predictive modeling,
connectivity-structure enrichment, and independent cross-dataset
replication — and reports honestly wherever the ambiguity survives a
given angle.

**The pitch, in one sentence:** using the male and female fly
connectomes, predict which neurons are sexually rewired from the known
genetic switch, and treat what the model can't explain as candidate
evidence for a second, uncharacterized driver.

**What is and isn't novel here.** The male connectome, the male↔female
comparison, and the headline *fru*/*dsx* correspondence percentages are
Berg et al.'s result, not this project's — see [Section 6](#6-honest-novelty-statement).
This project's contribution is the residual-as-discovery framing: model
out the genetic switch, characterize the leftover set, and validate it
against a second independent dataset.

## 2. Data

| dataset | role | access |
|---|---|---|
| MaleCNS v1.0 | primary: 165,122 traced neurons, per-neuron `dimorphism` and `fruDsx` annotations | neuPrint (targeted queries) + public bulk feather files |
| FlyWire/FAFB | female brain-only comparison, 139,248-neuron curated annotation table | `flyconnectome/flywire_annotations` (curated, preferred over Codex) + CAVEclient live connectivity |
| BANC | second, independent female whole-CNS reconstruction (188,508 neurons), used for Phase 4 replication | public bulk metadata table (no auth); live connectivity blocked by an account-level CAVE permission gap — see [Section 4.5](#45-independent-replication-phase-4) |

All three are accessed exactly as documented in `docs/project_plan.md`
section (b); per-phase access details and caveats are in
`fruresidual.data.*`.

## 3. Methods overview

The pipeline is four phases, each gated by the previous one's result
(`scripts/run_phase0.py` → `run_phase4.py`; `configs/config.yaml`
controls the one genuine analysis parameter, the *fru*/*dsx*
confidence-aggregation rule). Full statistical detail is in
`docs/phaseN_results.md`; this section is the minimum needed to read
Section 4.

- **Phase 0 (kill-gate).** Build the residual set directly: MaleCNS
  ships `dimorphism` and `fruDsx` per neuron already, so no type→neuron
  rollup was needed (the plan anticipated this as the hard part; it
  wasn't). Residual = dimorphic/male-specific neurons with no *fru*/*dsx*
  annotation. Structure tested with Fisher's exact test per superclass
  (residual vs. the dimorphic population it's drawn from),
  Benjamini-Hochberg corrected.
- **Phase 1 (atlas).** The same enrichment test applied across neuropil
  (via a targeted live neuPrint ROI query), superclass, hemilineage,
  soma neuromere, nerve, side, and neurotransmitter.
- **Phase 2 (predictive model).** A case-control population (634
  dimorphic + 634 isomorphic neurons, matched by superclass, restricted
  to types with a real FlyWire cross-match) with a wiring-difference
  target: `1 - cosine_similarity` between each neuron's MaleCNS-side and
  FlyWire-side partner-superclass connectivity profile. Ridge and
  gradient-boosting baselines, each trained with and without a
  *fru*/*dsx* feature (the ablation), then a 2-layer GraphSAGE GNN on
  the real connectome subgraph among the 1,268 population neurons.
- **Phase 3 (hypothesis).** Within superclasses that have enough of
  *both* residual and explained neurons to support a non-tautological
  comparison (`ascending_neuron`, `cb_intrinsic`, `vnc_intrinsic`),
  neuron-level partner-type enrichment (Fisher's exact + BH) and Leiden
  clustering on connectivity profiles, resolution-swept
  (`[0.02, 0.05, 0.1, 0.2, 0.3]`) rather than fixed, with a result only
  trusted if significant at a majority of resolutions.
- **Phase 4 (replication).** BANC's public bulk metadata table
  (`malecns_match` cross-references + BANC's own independent
  `sexually_dimorphic` call) checked against the same residual/explained
  split, via recall, precision, and a Fisher's exact test on whether
  residual neurons are confirmed dimorphic at a different rate than
  explained neurons.

## 4. Results

### 4.1 The residual is large and structured (Phase 0)

| aggregation rule | dimorphic/male-specific | residual (no fru/dsx) | % residual |
|---|---|---|---|
| any_positive | 2,368 | 921 | 38.9% |
| high_confidence_only | 2,368 | 1,079 | 45.6% |

Both rules pass a real statistical structure test (not a heuristic): at
least one superclass significantly enriched (BH-corrected p<0.05, odds
ratio ≥2). `ol_intrinsic` (optic-lobe-intrinsic) is enriched almost to
saturation (OR≈3,291 — essentially all optic-lobe dimorphic neurons lack
*fru*/*dsx*); `descending_neuron` (OR≈9.3) and `ascending_neuron`
(OR≈3.0), both VNC-brain connector types, are enriched at a smaller but
still significant scale. Notably, `cb_intrinsic` — the largest raw
superclass in the residual by count — is **not** significantly enriched;
its share of the residual is proportional to its share of the overall
dimorphic population, not concentrated beyond that.

A direct comparison against the paper's own central-brain statistics
(90.4% of male-specific, 60.3% of dimorphic neurons are *fru*+/*dsx*+)
matched exactly on absolute *fru*/*dsx* counts and on the male-specific
percentage, but not on the "sexually dimorphic" percentage (79.0%
per-neuron here vs. 60.3% in the paper) — investigated but not fully
resolved; see `docs/phase0_results.md` for the full investigation and
the most likely explanation (a different MaleCNS annotation snapshot).
This discrepancy does not affect the gate verdict, which does not depend
on matching the paper's number.

### 4.2 Where it concentrates (Phase 1)

The residual is not simply "optic lobe" — it is a peripheral-sensorimotor
pattern. Central-brain integrative neuropils (SIP, AVLP, SNP, SMP) sit at
14–32% residual; VNC, the gnathal ganglion, and the abdominal neuromere
tract sit at 70–76%; optic-lobe neuropils (ME, LO) sit at 95–100%. This
pattern is consistent with two explanations the atlas alone cannot
distinguish: a real biological one (some other mechanism drives
peripheral-sensorimotor rewiring) or an ascertainment one (the plan
already documents that ~75% of *fru*/*dsx* annotation effort concentrated
in the central brain). Phase 3 is the attempt to adjudicate this with
connectivity structure rather than raw annotation density.

### 4.3 Structure predicts the residual about as well as the genetic switch does (Phase 2)

| model | R² without fru/dsx | R² with fru/dsx | Δ |
|---|---|---|---|
| ridge (CV mean) | 0.090 ± 0.154 | 0.092 ± 0.154 | +0.002 |
| gradient boosting (CV mean) | **0.219 ± 0.138** | 0.216 ± 0.132 | −0.003 |
| GNN (10-seed mean, subgraph) | −0.592 ± 1.000 | — | — |

Both ablation differences are well inside one cross-validation standard
deviation — adding *fru*/*dsx* as a feature does not improve prediction
once superclass, connectivity context, and neurotransmitter are already
present. Read alongside 4.1–4.2, this is consistent with the residual
being explained by regional/structural identity rather than requiring
the genetic label at all. **This ablation — the actual Phase 2
deliverable — rests entirely on the deterministic sklearn baselines and
is unaffected by the GNN result below.**

The GNN does **not** beat those baselines. Swept over 10 seeds it gives
R² = −0.592 ± 1.000, ranging from −2.324 to +0.556, and beats the
gradient-boosting baseline on only 3 of 10 seeds. With 1,268 nodes,
7,441 internal edges and a ~190-node test split, there simply isn't
enough data to train this GNN stably. The plan pre-registered exactly
this outcome ("if the GNN doesn't beat gradient boosting, report that"),
so it is reported.

The absolute R² values here are much lower than earlier versions of this
document reported (gradient boosting 0.507, GNN 0.399) because the
**target itself was rebuilt**. `SHARED_CATEGORIES` contained two partner
categories (`ascending`, `descending`) that FlyWire could express and no
MaleCNS superclass mapped to, so the male profile vector was
structurally zero there while the female one was not — inflating cosine
distance one-sidedly across 293 of 371 types and accounting for ~76% of
the measured signal. Predictability fell with it, which is the expected
direction: the artifact tracked superclass, a feature the model had, so
the model was substantially predicting the artifact. The case/control
separation survives the correction (Mann-Whitney p=8.0e-16, effect size
+0.259), as does the ablation conclusion. See `docs/phase2_results.md`.

An earlier version of this writeup claimed R²=0.735 and "clearly beats
both baselines." That was an artifact of two compounding errors, both
since fixed: `train_gnn` seeded only its numpy data split and not
PyTorch's weight init (so no run was reproducible — the same call
returned 0.735, 0.702, 0.615 on successive reruns), and a single GNN
split was being compared against already-cross-validated baselines. See
`docs/phase2_results.md` for the full correction and the regression
tests that now pin GNN determinism.

### 4.4 One clean circuit-level candidate (Phase 3)

Three methodological gaps were found and fixed before trusting any
number here: a regional confound from comparing against a
zero-internal-variance superclass, a multi-label statistical bug that
silently counted graph edges instead of neurons, and a clustering
resolution that had drifted to an unvalidated fixed value contradicting
the code's own documented sensitivity-sweep requirement. Full detail in
`docs/phase3_results.md`.

A fourth gap — the degree confound — was corrected rather than merely
flagged. Every partner-type motif is now tested with
Cochran-Mantel-Haenszel within quartile bins of connectivity degree, so
residual neurons are compared only against explained neurons of
comparable connectivity intensity. Only motifs surviving that reach the
shortlist:

| stratum | degree-confounded? | marginal | degree-adjusted |
|---|---|---|---|
| ascending_neuron | no | 68 | **25** |
| cb_intrinsic | yes | 1,397 | **1,429** |
| vnc_intrinsic | yes | 306 | **44** |

`ascending_neuron` remains the best-supported result: 25 motifs survive
(headline hits `INXXX230`, `INXXX058`, `INXXX306`, `DNpe030` among them,
odds ratios 6–23), plus a Leiden cluster split significant at 3 of 5
swept resolutions concentrating 60.7% of residual neurons in one cluster
against 33% of background (odds ratio 2.68, p_adj=0.012), with no
detected degree confound (Mann-Whitney p=0.855).

`vnc_intrinsic` largely dissolved — 86% of its motifs were degree
artifact, confirming the old caveat's suspicion. `cb_intrinsic` did
**not**: its motifs survive at 102%, so degree does not explain them, and
the previous "read with a degree caveat" warning is resolved. What
remains notable there is breadth — 29% of all tested partner types — which
reads as a diffuse population-level connectivity difference rather than a
specific circuit signature, and it still shows zero cluster structure at
any swept resolution.

Two further corrections tightened the shortlist: Phase 1's annotation
dimensions are now re-tested *within* strata instead of being folded in
marginally (they previously carried the same regional confound the motif
analysis had been fixed for), and the unassigned-lineage placeholders
`putative_primary` and `TBD` — each the largest hit in its dimension —
are excluded as under-annotation rather than biology.

### 4.5 Independent replication (Phase 4)

The plan called for re-deriving the Phase 2 wiring-difference metric
using BANC connectivity. Blocked by a real, external constraint: the
project's CAVE token authenticates against BANC's datastack but lacks
the separate "view" permission BANC requires — confirmed directly (403
response), not assumed. The phase pivoted to what BANC's public bulk
metadata exposes without that permission: direct neuron-level
cross-references (`malecns_match`) plus BANC's own independent
`sexually_dimorphic` classification, yielding 13,942 unambiguous 1:1
matches.

| population | n matched | % confirmed dimorphic by BANC |
|---|---|---|
| residual (fru/dsx-unexplained) | 106 | 86.8% |
| explained (fru/dsx-positive) | 28 | 82.1% |
| **difference** | | **p = 0.547 (not significant)** |

This is the central Phase 4 result: an independent research group's
independent female reconstruction confirms the residual and the
explained set as dimorphic at statistically indistinguishable rates. If
the residual were mostly a matching artifact or MaleCNS-side annotation
noise, it should be *less* reliably confirmed than the explained set —
it isn't. Two robustness checks (added on audit) rule out two ways this
could have been misleading: only 1 of the 134 matched dimorphic neurons
is BANC-unassessed rather than a real disagreement (recall moves 85.8%→
86.5% restricted to assessed-only neurons), and all 134 sit on
BANC-proofread reconstructions.

## 5. Limitations, stated plainly

Collected from each phase's own writeup, not softened:

- **The Phase 0 "sexually dimorphic" central-brain percentage does not
  exactly match the published figure** (79.0% vs. 60.3%), narrowed to a
  likely data-snapshot version difference but not conclusively resolved.
  Does not affect the gate verdict.
- **The biology-vs-ascertainment-bias question is narrowed, not
  settled.** Phase 1's regional pattern cannot distinguish them on its
  own, since the annotation effort that produced it was itself
  concentrated in the central brain. Phase 3 provides the first real
  evidence against pure ascertainment: `cb_intrinsic` sits in the
  *well-annotated* central brain — where "residual = under-annotated" is
  least plausible — and its residual neurons show connectivity
  differences that survive both regional stratification and degree
  adjustment. That argues the residual is not purely an annotation
  artifact. It says nothing about the poorly-annotated periphery, where
  the ambiguity remains wide open.
- **Phase 2's wiring-difference metric is coarse** (10 superclass-level
  partner categories, not fine partner identity) and **brain-only**
  (FlyWire has no VNC), so any genuinely VNC-driven rewiring is
  structurally invisible to that phase specifically.
- **The GNN result is not usable as evidence.** It is too seed-unstable
  on this population (R² −0.762 to +0.882 across 10 seeds) to support
  any claim about graph structure carrying signal beyond per-neuron
  features. A defensible version would need k-fold cross-validation over
  the graph and a larger population than this case-control design gives.
- **Phase 3's `cb_intrinsic` result is broad rather than specific.** The
  degree confound is now corrected (Cochran-Mantel-Haenszel), and its
  motifs survive it — but 29% of all tested partner types coming out
  significant describes a diffuse population-level difference, not a
  circuit signature. `vnc_intrinsic`'s motifs were 86% degree artifact
  and its remainder (44 motifs) is small.
- **Phase 4 validates the dimorphism *classification*, not Phase 2's or
  Phase 3's specific numeric metrics** — BANC's public table gives a
  categorical call, not a continuous score, so no rank-correlation
  check (as the plan originally specified) was possible; and per-superclass
  Phase 4 breakdowns (n=9–33) are explicitly underpowered.
- **No confirmed biological mechanism is claimed anywhere in this
  project.** The `ascending_neuron` finding is a validated candidate
  signature, not a discovered gene or confirmed circuit.

## 6. Honest novelty statement

**Already published, not claimed here:**
- The male connectome and the male↔female wiring comparison itself,
  including the *fru*/*dsx*-correspondence percentages this project
  starts from (Berg et al., *Cell* 2026).
- The dimorphic social-behavior circuit findings (Rubin et al., *Current
  Biology* 2026) and the female *fru*/*dsx* catalogue (Deutsch/Murthy-lab,
  bioRxiv 2025.06.10.658788).
- "Predict dimorphism from wiring structure" as a general idea — at
  least one prior open project (ConnectomeLens) already trains a
  classifier for this on the male connectome alone.

**Genuinely this project's contribution:**
1. The residual-as-discovery framing — deliberately modeling out the
   genetic switch and treating the leftover error as the object of
   study, rather than predicting the switch itself.
2. The four-phase validation structure applied to that residual: spatial
   atlas → predictive ablation → connectivity-structure hypothesis →
   independent cross-dataset replication, each phase's result
   constraining what the next phase is allowed to claim.
3. The cross-dataset (BANC) replication of the residual specifically,
   including discovering and working around a real access-permission
   gap rather than silently skipping that phase.

The contribution is a **method and a validated hypothesis**, not a
claim of a newly discovered gene or confirmed mechanism.

## 7. Reproducing this writeup

```
mamba env create -f environment.yml && conda activate fruresidual
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or the CPU wheel
pip install torch_geometric navis fafbseg banc
# copy .env.example to .env, fill in NEUPRINT_TOKEN and CAVE_TOKEN
python scripts/download_male_cns.py
python scripts/run_phase0.py
python scripts/run_phase1.py
python scripts/run_phase2_data.py && python scripts/run_phase2.py
python scripts/run_phase3_data.py && python scripts/run_phase3.py
python scripts/run_phase4.py
python scripts/make_summary_figure.py
python -m pytest && ruff check .
```

See the top-level [README.md](../README.md) for environment-setup detail
(Windows/conda-specific notes, the PyTorch CUDA-wheel workaround) and
each `docs/phaseN_results.md` for full statistical methodology,
regression-tested numbers, and per-phase limitations beyond what's
summarized in Section 5.
