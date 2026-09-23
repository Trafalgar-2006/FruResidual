"""Fly Cell Atlas (head, 10x) access -- an independent, transcriptomic
measurement of fru/dsx to set against the connectome annotations.

Why this dataset and not the Ozel optic-lobe atlas: Ozel identifies its
clusters by integer only, and the cluster->cell-type mapping is not in
any public file (docs/scrnaseq_route_assessment.md). FCA annotates cells
with descriptive strings that EMBED connectome cell-type names
("medullary intrinsic neuron Mi1", "transmedullary neuron Tm9"), so the
join can be made and then validated against the connectome's own
vocabulary rather than assumed.

The fru caveat from that assessment still applies in full and is not
solved here: FCA is 10x 3' sequencing, fru's four promoters differ at
the 5' end, and only P1 produces the sex-specific FruM that every
connectome fru annotation means. Total fru transcript is NOT the same
quantity. dsx carries no equivalent problem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FRU_DSX_GENES = ("fru", "dsx")


def load_fca_fru_dsx(
    h5ad_path: Path, connectome_types: set[str], genes: tuple[str, ...] = FRU_DSX_GENES
) -> pd.DataFrame:
    """Per (cell type, sex): what fraction of cells express fru / dsx.

    Returns long format (cell_type, sex, gene, pct_expressing, n_cells,
    mean_count).

    The cell type is taken as the LAST whitespace-separated token of
    FCA's `annotation` string and then kept only if it exists in
    `connectome_types`. That validation is the point: it turns a fuzzy
    string-parsing step into one that cannot silently invent a type, at
    the cost of dropping annotations whose names do not correspond to a
    connectome type at all ("Kenyon cell", "hemocyte", "unknown").

    Expression is reported as the fraction of cells with a non-zero
    count, not a mean expression level. On raw unnormalised counts that
    is the robust choice -- it does not require library-size correction
    and is directly comparable to how the Ozel marker tables report
    `pct.1`/`pct.2`.
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    present = [g for g in genes if g in set(adata.var_names.astype(str))]
    if not present:
        raise ValueError(
            f"None of {genes} are in the FCA gene list. Available names look like "
            f"{list(adata.var_names[:5])}"
        )

    obs = adata.obs
    ann = obs["annotation"].astype(str)
    candidate = ann.str.split().str[-1]
    valid = candidate.isin(connectome_types)

    # Only the gene columns are pulled into memory: n_cells x 2, not 5.3GB.
    sub = adata[:, present].to_memory()
    x = sub.X
    x = np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)

    frame = pd.DataFrame({
        "cell_type": candidate.where(valid),
        "sex": obs["sex"].astype(str).to_numpy(),
        "annotation": ann.to_numpy(),
    })
    for i, g in enumerate(present):
        frame[g] = x[:, i]

    frame = frame[frame["cell_type"].notna() & frame["sex"].isin(["male", "female"])]

    rows = []
    for (ct, sex), grp in frame.groupby(["cell_type", "sex"], observed=True):
        for g in present:
            v = grp[g].to_numpy(dtype=float)
            rows.append({
                "cell_type": ct, "sex": sex, "gene": g,
                "n_cells": len(v),
                "n_expressing": int((v > 0).sum()),
                "pct_expressing": float(100 * (v > 0).mean()),
                "mean_count": float(v.mean()),
            })
    out = pd.DataFrame(rows)
    out.attrs["n_annotations_total"] = int(ann.nunique())
    out.attrs["n_annotations_matched"] = int(candidate[valid].nunique())
    out.attrs["genes_found"] = present
    return out
