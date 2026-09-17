# Phase 1: the unexplained-dimorphism atlas

Run: 2026-09-17, via `scripts/run_phase1.py`. Built on the same
residual/dimorphic split as Phase 0 (`any_positive` rule: 2,368
dimorphic/male-specific neurons, 921 residual). Two complementary views:
neuropil-level (live neuPrint ROI query, targeted to just those 2,368
neurons) and categorical (superclass, hemilineage, soma neuromere,
nerve, side, neurotransmitter — all via the same Fisher's-exact +
Benjamini-Hochberg enrichment test Phase 0 used).

## Headline: the residual is a peripheral-sensorimotor pattern, not just "optic lobe"

Ranking neuropils by total dimorphic-neuron involvement
(`reports/figures/phase1/roi_residual_by_involvement.png`):

| region | % residual |
|---|---|
| SIP (R/L) | 14.5 / 14.9 |
| AVLP (R/L) | 20.4 / 19.9 |
| SNP (R/L) | 21.8 / 22.5 |
| CentralBrain (aggregate) | 28.1 |
| SMP (R/L) | 30.9 / 31.8 |
| GNG | 69.9 |
| VNC (aggregate) | 71.5 |
| ANm | 75.7 |
| Optic (R/L) | 97.5 / 97.9 |
| ME (R/L) | 99.7 / 100.0 |

Central-brain integrative neuropils (SIP, AVLP, SNP, SMP — lateral
horn/protocerebrum association areas) sit at 14-32% residual, close to
the overall central-brain figure from Phase 0. Optic-lobe neuropils (ME,
LO, Optic) are **95-100% residual** — essentially every dimorphic
optic-lobe synapse belongs to a fru/dsx-unannotated neuron. VNC, the
gnathal ganglion (GNG, feeding-related), and the abdominal neuromere
tract (ANm) sit in between at 70-76%. This is a real, sharp regional
gradient, not an artifact of one specific `ol_intrinsic` label — see
`reports/figures/phase1/roi_residual_by_pct.png` for the full ranked
view (dominated by LOP/ME/Optic/mVAC/AbNT once the ~4,386 fine
sub-parcellations — optic-lobe columns, optic-lobe layers, antennal-lobe
glomeruli — are filtered out; see `roi_residual_summary` docstring).

## Categorical enrichment (Fisher's exact, BH-corrected per dimension)

| dimension | significant / tested | notable hits |
|---|---|---|
| superclass | 4/10 | `ol_intrinsic` (OR~3291), `descending_neuron` (OR~9.3), `ascending_neuron` (OR~3.0), `sensory_descending` (OR~20.6) |
| somaNeuromere | 7/19 | `LB` (labial neuromere of GNG, n=32, OR~21.8), VNC segments `T3`/`A1`/`A2`/`A4`/`A5`/`TC` |
| consensus_nt | 1/5 | `acetylcholine` (n=764, 83% of residual, OR~2.64) |
| somaSide | 1/3 | `M` (midline, unpaired neurons, n=9, OR~30.3) |
| class | 0/2 | none |
| subclass | 0/15 | none |
| entryNerve / exitNerve | 0/1, 0/2 | mostly null — sparse coverage |

Full tables: `data_processed/phase1_enrichment_<dimension>.csv`, figures:
`reports/figures/phase1/enrichment_<dimension>.png`.

### A caveat on hemilineage (itoleeHl/trumanHl) — read before citing these

`itoleeHl` (17/61 significant) and `trumanHl` (4/29 significant) both
show real signal, but **their single largest hits are placeholder
labels, not real hemilineages**: `itoleeHl`'s top category by count is
`putative_primary` (n=74, "not yet confidently assigned a lineage") and
`trumanHl`'s top category is literally `TBD` (n=44). Both are exactly
the kind of "genuinely unassigned" bucket that would inflate this
enrichment result independent of biology — a neuron missing a fru/dsx
call and missing a confident hemilineage call could just mean "under-
annotated neuron," not "biologically distinct lineage." The remaining
significant hemilineage hits (e.g. `LALa1_anterior`, `SMPpv2_dorsal`,
several `LB*` lineages consistent with the GNG/somaNeuromere finding
above) are named, real hemilineages and worth carrying forward, but
should be re-checked against the whole-population base rate of
"unassigned" labels before treating the ranked list at face value in
Phase 3.

## Interpretation — and the standing ambiguity

The pattern (central brain well-annotated, periphery poorly annotated)
is consistent with two different explanations that this data cannot yet
distinguish:
1. **Biological**: fru/dsx genuinely don't drive rewiring in peripheral
   sensorimotor circuits — some other mechanism does, concentrated there.
2. **Ascertainment bias**: the plan itself already notes (section b.5)
   that ~75% of fru/dsx annotation effort concentrated in the central
   brain — so peripheral regions may simply be under-annotated for
   fru/dsx, not under-expressing it.

This is exactly the ambiguity the original *Cell* 2026 paper flags
("likely results from incomplete gene expression data... however, other
genetic factors... may shape some dimorphisms") — Phase 1 sharpens
*where* the ambiguity lives spatially, it doesn't resolve which
explanation is right. That resolution is what Phase 3's proper
enrichment/connectivity-motif analysis is for.

## What this unblocks

- Phase 2 (predictive model) can proceed with a concrete, validated
  spatial/categorical picture of where the residual concentrates, useful
  for feature design and for interpreting model errors.
- The `fruresidual.data.neuprint_client.fetch_roi_info` +
  `fruresidual.features.atlas` pipeline is reusable for any future
  targeted ROI-level question without re-downloading the full connectome.
