"""Transcriptomic cross-check: do optic-lobe types the connectome calls
fru/dsx-negative actually express fru/dsx?

This is the one angle on the OPTIC LOBE the project has. Phase 6's
annotation cross-check answered the biology-vs-ascertainment question
for the central brain and descending neurons, but was explicitly silent
about the optic lobe -- where 372 of 576 matched residual neurons sit
and where every annotation source has near-zero coverage
(docs/annotation_crosscheck_results.md).

The Fly Cell Atlas head dataset is a different measurement modality
(transcript counts, not light-microscopy registration), densely samples
the optic lobe, and -- unlike the Ozel atlas -- annotates cells with
strings that embed connectome cell-type names, so the join can be made
and validated.

READ THE fru RESULT WITH THE ISOFORM CAVEAT. FCA is 10x 3' sequencing;
fru's promoters differ at the 5' end and only P1 gives the sex-specific
FruM that connectome annotations mean. Total fru transcript is a
different quantity. dsx has no such problem. See
docs/scrnaseq_route_assessment.md.

Usage:  python scripts/run_fca_crosscheck.py [--h5ad PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from fruresidual.config import load_config
from fruresidual.data.fca_client import load_fca_fru_dsx
from fruresidual.data.flywire_client import load_flywire_annotations
from fruresidual.features.fru_dsx_join import (
    DIMORPHIC_CATEGORIES,
    has_fru_dsx,
    load_male_cns_annotations,
)

# A type needs enough cells for a percentage to mean anything.
MIN_CELLS = 20


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", default=str(Path.home() / "Downloads" / "v2_fca_biohub_head_10x_raw.h5ad"))
    args = parser.parse_args()

    h5ad = Path(args.h5ad)
    if not h5ad.exists():
        raise SystemExit(
            f"FCA file not found at {h5ad}.\n"
            "Get v2_fca_biohub_head_10x_raw.h5ad from flycellatlas.org "
            "(or pass --h5ad PATH)."
        )

    mcns = load_male_cns_annotations(cfg, traced_only=True)
    fw = load_flywire_annotations(cfg.data_raw / "flywire_annotations.feather")
    connectome_types = set(mcns["type"].dropna().astype(str)) | set(fw["cell_type"].dropna().astype(str))
    print(f"connectome vocabulary: {len(connectome_types):,} distinct cell types")

    print(f"reading {h5ad.name} (backed mode)...")
    expr = load_fca_fru_dsx(h5ad, connectome_types)
    print(f"FCA annotations: {expr.attrs['n_annotations_total']} total, "
          f"{expr.attrs['n_annotations_matched']} matched a connectome type")
    print(f"genes found: {expr.attrs['genes_found']}")

    expr = expr[expr["n_cells"] >= MIN_CELLS]
    types_covered = sorted(expr["cell_type"].unique())
    print(f"types with >= {MIN_CELLS} cells in at least one sex: {len(types_covered)}")

    # Which of these does the connectome call residual?
    dim = mcns[mcns["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dim["has_fru_dsx"] = has_fru_dsx(dim, cfg.fru_dsx_aggregation)
    by_type = dim.groupby(dim["type"].astype(str))["has_fru_dsx"].agg(["size", "sum"])
    by_type["is_residual_type"] = by_type["sum"] == 0

    expr["connectome_status"] = expr["cell_type"].map(
        lambda t: ("residual" if by_type.loc[t, "is_residual_type"] else "explained")
        if t in by_type.index else "not_dimorphic"
    )

    print()
    print("=" * 74)
    print("Coverage: what does FCA actually overlap with?")
    print("=" * 74)
    print(expr.drop_duplicates("cell_type")["connectome_status"].value_counts().to_string())

    print()
    print("=" * 74)
    print("fru / dsx expression by connectome status (both sexes pooled)")
    print("=" * 74)
    for gene in expr["gene"].unique():
        g = expr[expr["gene"] == gene]
        print(f"\n  --- {gene} ---")
        for status, sub in g.groupby("connectome_status"):
            per_type = sub.groupby("cell_type")["pct_expressing"].mean()
            print(f"    {status:14s} n_types={len(per_type):3d}  "
                  f"mean pct expressing={per_type.mean():5.1f}%  "
                  f"median={per_type.median():5.1f}%  max={per_type.max():5.1f}%")

    print()
    print("=" * 74)
    print("Sex-differential expression (FruM/DsxM are the sex-specific forms)")
    print("=" * 74)
    wide = expr.pivot_table(index=["cell_type", "gene", "connectome_status"],
                            columns="sex", values="pct_expressing").reset_index()
    if {"male", "female"}.issubset(wide.columns):
        wide["male_minus_female"] = wide["male"] - wide["female"]
        for gene in wide["gene"].unique():
            g = wide[wide["gene"] == gene].dropna(subset=["male", "female"])
            if not len(g):
                continue
            print(f"\n  --- {gene} --- ({len(g)} types with both sexes)")
            print(f"    mean male-female difference: {g['male_minus_female'].mean():+.1f} pct points")
            top = g.reindex(g["male_minus_female"].abs().sort_values(ascending=False).index).head(8)
            print(top[["cell_type", "connectome_status", "female", "male",
                       "male_minus_female"]].to_string(index=False))

    out_csv = cfg.data_processed / "fca_fru_dsx_by_type.csv"
    expr.to_csv(out_csv, index=False)
    summary = {
        "n_connectome_types": len(connectome_types),
        "n_fca_annotations_total": int(expr.attrs["n_annotations_total"]),
        "n_fca_annotations_matched": int(expr.attrs["n_annotations_matched"]),
        "min_cells": MIN_CELLS,
        "n_types_covered": len(types_covered),
        "types_covered": types_covered,
        "genes_found": list(expr.attrs["genes_found"]),
        "coverage_by_status": expr.drop_duplicates("cell_type")["connectome_status"]
            .value_counts().to_dict(),
        "pct_expressing_by_status": {
            gene: {
                status: float(sub.groupby("cell_type")["pct_expressing"].mean().mean())
                for status, sub in g.groupby("connectome_status")
            }
            for gene, g in expr.groupby("gene")
        },
    }
    with open(cfg.data_processed / "fca_crosscheck_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {out_csv.name} and fca_crosscheck_summary.json")


if __name__ == "__main__":
    main()
