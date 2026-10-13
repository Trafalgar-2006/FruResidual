"""Export the fru/dsx-unexplained residual neuron IDs for this project.

The residual is DEFINED in the FruResidual project, not here. That package
is installed into this conda env, so we import its definition rather than
re-implementing it -- a second copy of `dimorphic AND NOT has_fru_dsx`
would be free to drift from the one that was actually validated, and the
whole ablation experiment rests on these IDs being the same population.

Reads only. Nothing in FruResidual or its data directory is modified.

Usage:  python scripts/export_residual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import (
    DIMORPHIC_CATEGORIES,
    has_fru_dsx,
    load_male_cns_annotations,
)

OUT_DIR = Path("C:/CourtshipData/processed")


def main() -> None:
    cfg = load_config()
    rule = cfg.fru_dsx_aggregation
    print(f"aggregation rule (from FruResidual config): {rule}")

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)

    residual = dimorphic.loc[~dimorphic["has_fru_dsx"]].copy()
    explained = dimorphic.loc[dimorphic["has_fru_dsx"]]

    print(f"traced neurons:        {len(df):,}")
    print(f"dimorphic:             {len(dimorphic):,}")
    print(f"  residual (no fru/dsx): {len(residual):,}")
    print(f"  explained:             {len(explained):,}")

    if not len(residual):
        sys.exit("no residual neurons found -- the join is broken, stop here")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("bodyId", "type", "superclass", "dimorphism") if c in residual.columns]
    out = OUT_DIR / "residual_body_ids.csv"
    residual[cols].to_csv(out, index=False)
    print(f"\nwrote {len(residual):,} residual IDs -> {out}")

    # The control population for the ablation: same dimorphic class, but the
    # switch DOES explain them. Ablating the residual means nothing without
    # this comparison.
    out_ctrl = OUT_DIR / "explained_body_ids.csv"
    explained[cols].to_csv(out_ctrl, index=False)
    print(f"wrote {len(explained):,} explained IDs -> {out_ctrl}")

    # Sanity: these must be disjoint and must jointly cover the dimorphic set.
    assert not set(residual["bodyId"]) & set(explained["bodyId"])
    assert len(residual) + len(explained) == len(dimorphic)
    print("\nchecks passed: residual and explained are disjoint and cover the dimorphic set")

    counts = residual["superclass"].value_counts() if "superclass" in residual else pd.Series(dtype=int)
    if len(counts):
        print("\nresidual by superclass (top 8):")
        print(counts.head(8).to_string())


if __name__ == "__main__":
    main()
