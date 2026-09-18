"""Phase 4: cross-dataset validation against BANC (independent female
reconstruction). See docs/project_plan.md section (d), PHASE 4, and
docs/phase4_results.md for why this validates dimorphism concordance
rather than re-deriving the Phase 2 wiring-difference metric (BANC's
live connectivity API needs a "view" permission this project's account
doesn't have -- see fruresidual.data.banc_client for the full story).

Usage:  python scripts/run_phase4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruresidual.config import load_config
from fruresidual.data.banc_client import load_banc_meta
from fruresidual.features.cross_validation import compute_concordance, match_banc_to_malecns
from fruresidual.features.fru_dsx_join import (
    DIMORPHIC_CATEGORIES,
    has_fru_dsx,
    load_male_cns_annotations,
)
from fruresidual.viz.phase4 import plot_concordance_bar


def main() -> None:
    cfg = load_config()
    rule = cfg.fru_dsx_aggregation
    cfg.data_processed.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(__file__).resolve().parents[1] / "reports" / "figures" / "phase4"

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic_mask = df["dimorphism"].isin(DIMORPHIC_CATEGORIES)
    dimorphic = df[dimorphic_mask].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)
    residual_body_ids = set(dimorphic.loc[~dimorphic["has_fru_dsx"], "bodyId"])
    print(f"MaleCNS traced population: {len(df)}, dimorphic/male-specific: {len(dimorphic)}, "
          f"residual: {len(residual_body_ids)}")

    banc_df = load_banc_meta(cfg.data_raw / "banc_888_meta.feather")
    print(f"BANC metadata: {len(banc_df)} neurons, {banc_df['malecns_match'].notna().sum()} "
          f"with a raw (possibly ambiguous) malecns_match")

    all_traced_ids = set(df["bodyId"])
    matched = match_banc_to_malecns(banc_df, all_traced_ids)
    print(f"unambiguous 1:1 BANC<->MaleCNS matches within the traced population: {len(matched)}")

    malecns_dimorphism = df.set_index("bodyId")["dimorphism"]
    result = compute_concordance(matched, malecns_dimorphism, residual_body_ids)

    print("\n=== cross-dataset concordance ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Also break down by superclass, for the strata Phase 3 actually
    # found something in -- does ascending_neuron's candidate signature
    # hold up under BANC's independent dimorphism call too?
    by_superclass = {}
    for sc in ("ascending_neuron", "cb_intrinsic", "vnc_intrinsic", "ol_intrinsic"):
        sc_ids = set(dimorphic.loc[dimorphic["superclass"] == sc, "bodyId"])
        sc_matched = matched[matched["bodyId"].isin(sc_ids)]
        if len(sc_matched) < 5:
            by_superclass[sc] = {"n_matched": len(sc_matched), "note": "too few matches to test"}
            continue
        sc_result = compute_concordance(sc_matched, malecns_dimorphism, residual_body_ids)
        by_superclass[sc] = sc_result
        print(f"\n  [{sc}] n_matched={sc_result['n_matched_total']}, "
              f"residual confirmed={sc_result['pct_residual_confirmed_dimorphic_by_banc']}%, "
              f"explained confirmed={sc_result['pct_explained_confirmed_dimorphic_by_banc']}%, "
              f"p={sc_result['residual_vs_explained_p_value']}")

    summary = {
        "n_traced": len(df),
        "n_dimorphic": len(dimorphic),
        "n_residual": len(residual_body_ids),
        "n_unambiguous_banc_matches": len(matched),
        "overall": result,
        "by_superclass": by_superclass,
    }
    with open(cfg.data_processed / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    matched.to_csv(cfg.data_processed / "phase4_banc_matches.csv", index=False)

    plot_concordance_bar(result, by_superclass, figures_dir / "concordance_bar.png")

    print("\nSaved summary to", cfg.data_processed / "phase4_summary.json")
    print("Saved figure to", figures_dir / "concordance_bar.png")


if __name__ == "__main__":
    main()
