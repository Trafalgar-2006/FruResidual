# The scRNA-seq route for the optic-lobe gap: assessed and rejected

Assessed 2026-09-20, before committing analysis time. **Verdict: this
route cannot answer the question it was proposed for.** All three of its
legs fail independently, and the failures are structural rather than
fixable with more data of the same kind.

## Why it was proposed

Phase 6 (`docs/annotation_crosscheck_results.md`) established that the
residual is not simply unannotated — but only in the central brain and
descending neurons. It could say nothing about the **optic lobe**, where
**372 of the 576** matched residual neurons sit and where both annotation
sources are blind (Codex 0.01% fru/dsx coverage, curated table 0.08%).

Single-cell RNA-seq looked like the answer: a different measurement
modality (transcript counts, not light-microscopy registration), dense
in exactly that region. The Özel et al. optic-lobe atlas
(GSE142787, ~250k cells) and the Fly Cell Atlas head dataset were the
two candidates.

## Leg 1 — `fru` cannot be measured by this technology

*fru* transcribes from four promoters. Only **P1** produces the
sex-specific FruM protein, and P1 is what the entire courtship /
dimorphism literature — and therefore every *fru* annotation in every
connectome — refers to. P2–P4 are common to both sexes and broadly
expressed.

The promoters differ at the **5′ end**. Both Özel and the Fly Cell Atlas
are **10x 3′** sequencing, which reads the opposite end of the
transcript. The data therefore cannot separate P1 from P2–P4.

The consequence is visible in the numbers rather than theoretical. In
`Cluster_markers_Adult`, *fru* appears as a marker of 32 of 199 clusters
with `pct.1` around 0.79–0.85 — but `pct.2` is **0.48**, meaning roughly
half of *all* adult optic-lobe cells carry detectable *fru*. Against a
connectome annotation that puts *fru* at 0.01% in the optic lobe, that
is not a discrepancy to explain; it is a different quantity.

**Taking that 48% at face value would have produced a headline finding
of catastrophic ascertainment bias that is purely an artefact of
comparing two different transcripts.**

## Leg 2 — the `dsx` signal is glial, and glia are not in the connectome

*dsx* looked salvageable: it is a marker of only 2 of 199 adult clusters
(199 and 201), `pct.1` 0.24/0.33 against `pct.2` 0.002 — sharp and
specific, exactly what leg 1 lacks.

Checked what those two clusters are, using canonical marker panels:

| cluster | glial markers present | neuronal markers present |
|---|---|---|
| 199 | `repo`, `wrapper`, `alrm`, `moody`, `gs2`, `Rh50` | **none** |
| 201 | `repo`, `wrapper`, `moody` | **none** |
| 1 (control) | none | `para`, `CadN` |
| 19 (control) | none | `syt1`, `VGlut` |

`repo` is the canonical pan-glial transcription factor; `wrapper`,
`alrm` and `moody` mark wrapping, astrocyte-like and subperineurial glia
respectively. Both *dsx*+ clusters are glial populations with no
neuronal markers at all, while control clusters show the opposite
pattern — so the panels are discriminating correctly.

**The connectome contains neurons.** Optic-lobe *dsx* expression
confined to two glial populations cannot be joined to any neuron in
MaleCNS or FlyWire, and *dsx* is detected in only 0.2% of all other
optic-lobe cells, so there is no neuronal signal underneath it either.

## Leg 3 — the cluster→cell-type mapping is not in the public files

Even setting legs 1 and 2 aside, `Cluster_markers.xlsx` identifies
clusters by **integer only** — no type names in any of its 12 sheets.
Nothing in the MAOL exports supplies the link either: both
`MAOL_olmatching.txt` and `MAOL_Nern_supplementary_table7.xlsx` are
connectome-to-connectome type-matching tables
(`OL_type ↔ Matsliah ↔ Schlegel ↔ hemibrain`) with **no cluster column
at all**.

The mapping exists, if anywhere, in the `.rds` Seurat object's
`meta.data` — an S4 R object that cannot be read reliably from Python.
So the R dependency that looked eliminated re-enters here, for a route
that legs 1 and 2 have already disqualified.

## What this predicts about the Fly Cell Atlas file

`v2_fca_biohub_head_10x_raw.h5ad` is also **10x 3′**, so leg 1 applies
to it unchanged — bigger file, identical blind spot. It is Python-native
(AnnData), which removes the R problem, and it is worth having for
central-brain sanity checks against results already in hand. It will not
rescue the optic lobe.

## What would actually work

- **5′ or full-length scRNA-seq** (Smart-seq2, PacBio/Nanopore long
  reads) of the fly brain, which can resolve the promoter and therefore
  *fru* P1.
- **A published *fru*-P1-specific analysis** — a literature question,
  not a download.
- **fru-P1-GAL4 driver-line expression data** via VFB/FlyLight. This is
  the primary source the connectome annotations are *derived from*, so
  it is not independent of them — but it would at least show directly
  whether anyone has ever looked at the optic lobe, which distinguishes
  "no expression" from "never examined".

## Standing conclusion

The optic-lobe portion of the residual — 372 of 576 matched residual
neurons, and the bulk of the residual overall — has **no independent
fru/dsx evidence available**, and none is obtainable from 3′ scRNA-seq.
Phase 6's finding stands as reported: it covers the central brain and
descending neurons, and is explicitly silent about the optic lobe.

This assessment cost roughly half an hour. Running the route to a
conclusion would have cost weeks and produced either a retracted
ascertainment-bias headline (leg 1) or an un-joinable glial result
(leg 2).
