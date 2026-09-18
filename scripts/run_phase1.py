"""Phase 1: the unexplained-dimorphism atlas. See docs/project_plan.md
section (d), PHASE 1.

Usage:  python scripts/run_phase1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruresidual.config import load_config
from fruresidual.data.neuprint_client import fetch_roi_info
from fruresidual.features.atlas import (
    load_neurotransmitters,
    multi_dimension_enrichment,
    parse_roi_info,
    roi_residual_summary,
)
from fruresidual.features.fru_dsx_join import (
    DIMORPHIC_CATEGORIES,
    has_fru_dsx,
    load_male_cns_annotations,
)
from fruresidual.viz.atlas import plot_enrichment_bar, plot_roi_bar, plot_roi_bar_by_pct

DIMENSIONS = [
    "superclass", "class", "subclass", "itoleeHl", "trumanHl",
    "somaNeuromere", "entryNerve", "exitNerve", "somaSide", "consensus_nt",
]


def main() -> None:
    cfg = load_config()
    rule = cfg.fru_dsx_aggregation
    cfg.data_interim.mkdir(parents=True, exist_ok=True)
    cfg.data_processed.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(__file__).resolve().parents[1] / "reports" / "figures" / "phase1"

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)
    residual_mask = ~dimorphic["has_fru_dsx"]
    residual_body_ids = set(dimorphic.loc[residual_mask, "bodyId"])
    print(f"dimorphic/male-specific: {len(dimorphic)}, residual: {len(residual_body_ids)}")

    # --- Neurotransmitter join ---
    nt = load_neurotransmitters(cfg)
    dimorphic = dimorphic.merge(nt, on="bodyId", how="left")
    # merge() resets the index, so recompute the mask against the merged
    # frame rather than reuse the pre-merge one (index no longer aligns).
    residual_mask = dimorphic["bodyId"].isin(residual_body_ids)

    # --- Categorical enrichment sweep ---
    enrichment_results = multi_dimension_enrichment(dimorphic, residual_mask, DIMENSIONS)
    for dim, result in enrichment_results.items():
        result.to_csv(cfg.data_processed / f"phase1_enrichment_{dim}.csv", index=False)
        sig = result[result["significant"]]
        print(f"\n=== {dim}: {len(sig)}/{len(result)} categories significantly enriched ===")
        if len(sig):
            print(sig[["category", "n_in_subset", "pct_of_subset", "odds_ratio", "p_adj"]].to_string(index=False))
        plot_enrichment_bar(
            result, f"Residual enrichment by {dim}", figures_dir / f"enrichment_{dim}.png"
        )

    # --- ROI / neuropil atlas (live neuPrint query, targeted) ---
    print(f"\nFetching ROI info for {len(dimorphic)} neurons from neuPrint...")
    roi_cache = cfg.data_interim / "phase1_roi_info.feather"
    if roi_cache.exists():
        roi_raw = pd.read_feather(roi_cache)
    else:
        roi_raw = fetch_roi_info(cfg, dimorphic["bodyId"].tolist())
        roi_raw.to_feather(roi_cache)
    print(f"  got roiInfo for {roi_raw['bodyId'].nunique()} neurons")

    roi_long = parse_roi_info(roi_raw["roiInfo"], roi_raw["bodyId"])
    roi_summary = roi_residual_summary(roi_long, residual_body_ids)
    roi_summary.to_csv(cfg.data_processed / "phase1_roi_summary.csv")
    plot_roi_bar(roi_summary, figures_dir / "roi_residual_by_involvement.png")
    plot_roi_bar_by_pct(roi_summary, figures_dir / "roi_residual_by_pct.png")
    print("\n=== top ROIs by total dimorphic-neuron involvement ===")
    print(roi_summary.head(15).to_string())

    # --- Summary JSON ---
    summary = {
        "rule": rule,
        "n_dimorphic": len(dimorphic),
        "n_residual": len(residual_body_ids),
        "dimensions_tested": list(enrichment_results.keys()),
        "significant_by_dimension": {
            dim: result.loc[result["significant"], "category"].tolist()
            for dim, result in enrichment_results.items()
        },
        "top_rois_by_pct_residual": roi_summary[roi_summary["total_synweight"] >= 100]
        .sort_values("pct_residual", ascending=False)
        .head(10)[["pct_residual", "total_synweight"]]
        .reset_index()
        .to_dict("records"),
    }
    with open(cfg.data_processed / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved figures to", figures_dir)
    print("Saved summary to", cfg.data_processed / "phase1_summary.json")


if __name__ == "__main__":
    main()
