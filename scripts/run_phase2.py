"""Phase 2 step 2: features, baselines (with/without fru/dsx ablation),
GNN, and the per-neuron residual scores that feed Phase 3. Requires
scripts/run_phase2_data.py to have been run first (produces
phase2_wiring_targets.feather and the two profile caches).

Usage:  python scripts/run_phase2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fruresidual.config import load_config
from fruresidual.data.flywire_client import load_flywire_annotations
from fruresidual.data.neuprint_client import fetch_population_edges
from fruresidual.features.atlas import load_neurotransmitters
from fruresidual.features.fru_dsx_join import load_male_cns_annotations
from fruresidual.features.ml_features import build_features
from fruresidual.features.rewiring import select_case_control_population
from fruresidual.models.baselines import evaluate_baselines, out_of_fold_predictions
from fruresidual.models.gnn import build_graph, cross_validate_gnn
from fruresidual.viz.phase2 import (
    plot_ablation_comparison,
    plot_predicted_vs_actual,
    plot_residual_distribution,
)

RANDOM_STATE = 0
# The GNN is k-fold cross-validated and repeated over seeds, matching the
# baselines' cv=5 so the numbers are comparable -- see the call site.
GNN_CV_FOLDS = 5
GNN_SEEDS = (0, 1, 2)


def main() -> None:
    cfg = load_config()
    figures_dir = Path(__file__).resolve().parents[1] / "reports" / "figures" / "phase2"

    targets = pd.read_feather(cfg.data_processed / "phase2_wiring_targets.feather")
    male_profiles = pd.read_feather(cfg.data_interim / "phase2_male_partner_profiles.feather")

    mcns = load_male_cns_annotations(cfg, traced_only=True)
    fw = load_flywire_annotations(cfg.data_raw / "flywire_annotations.feather")
    fw_types = set(fw["cell_type"].dropna().unique())
    cases, controls = select_case_control_population(mcns, flywire_cell_types=fw_types, random_state=RANDOM_STATE)
    population = pd.concat([cases, controls], ignore_index=True)
    population = population[population["bodyId"].isin(targets["bodyId"])]

    # consensus_nt is a build_features() input but lives in a separate
    # bulk file (not in load_male_cns_annotations' table) -- same join
    # scripts/run_phase1.py does. `body` is unique in that file (verified
    # 2026-09-17), so this left-merge can't duplicate or reorder rows.
    nt = load_neurotransmitters(cfg)
    population = population.merge(nt, on="bodyId", how="left")

    print(f"population with targets: {len(population)} ({population['is_case'].sum()} cases, "
          f"{(~population['is_case']).sum()} controls)")

    # --- Features: with and without fru/dsx (the ablation) ---
    # build_features() merges internally, which can in principle reorder
    # rows -- rather than trust that a left-merge preserves order (true
    # today, but an implicit assumption worth not depending on), derive y
    # from each X's OWN bodyId column via an explicit indexed lookup, so
    # features and target are aligned by construction, not by luck.
    X_without = build_features(population, male_profiles, include_fru_dsx=False)
    X_with = build_features(population, male_profiles, include_fru_dsx=True)
    assert X_without["bodyId"].equals(X_with["bodyId"]), "ablation feature sets must share row order"

    body_ids = X_without["bodyId"]
    y = targets.set_index("bodyId").loc[body_ids, "wiring_difference"].reset_index(drop=True)
    X_without_fit = X_without.drop(columns=["bodyId"])
    X_with_fit = X_with.drop(columns=["bodyId"])

    # --- Baselines + ablation ---
    results_without = evaluate_baselines(X_without_fit, y, random_state=RANDOM_STATE)
    results_without["ablation"] = "without_fru_dsx"
    results_with = evaluate_baselines(X_with_fit, y, random_state=RANDOM_STATE)
    results_with["ablation"] = "with_fru_dsx"
    ablation_table = pd.concat([results_without, results_with], ignore_index=True)
    ablation_table.to_csv(cfg.data_processed / "phase2_ablation_table.csv", index=False)
    print("\n=== ablation table ===")
    print(ablation_table.to_string(index=False))

    plot_ablation_comparison(ablation_table, figures_dir / "ablation_comparison.png")

    # --- Per-neuron residual scores: the actual Phase 2 deliverable ---
    # From the WITHOUT-fru/dsx model -- the point is what's left unexplained
    # by structure/region/neurotransmitter alone, mirroring Phase 0's logic
    # at the model-prediction level instead of the raw-annotation level.
    oof_pred = out_of_fold_predictions(X_without_fit, y, model_name="gradient_boosting", random_state=RANDOM_STATE)
    is_case_by_body = population.set_index("bodyId")["is_case"]
    residual_scores = pd.DataFrame(
        {
            "bodyId": body_ids.values,
            "actual_wiring_difference": y.values,
            "predicted_wiring_difference": oof_pred,
            "residual": y.values - oof_pred,
            "is_case": is_case_by_body.loc[body_ids].values,
        }
    )
    residual_scores.to_feather(cfg.data_processed / "phase2_residual_scores.feather")
    plot_predicted_vs_actual(residual_scores, figures_dir / "predicted_vs_actual.png")
    plot_residual_distribution(residual_scores, figures_dir / "residual_distribution.png")

    # --- GNN ---
    print("\nfetching population-internal edges for GNN...")
    edges = fetch_population_edges(cfg, population["bodyId"].tolist())
    print(f"  {len(edges)} population-internal edges")

    graph_features = X_without.copy()  # keeps bodyId for build_graph's id_to_idx mapping
    data = build_graph(graph_features, y, edges)

    # K-fold cross-validated, repeated over seeds -- the same protocol the
    # sklearn baselines above use (cv=5), so the two numbers are directly
    # comparable. Reporting a single GNN split against cross-validated
    # baselines is what originally made the GNN look like it won.
    gnn_cv = cross_validate_gnn(data, k=GNN_CV_FOLDS, seeds=GNN_SEEDS, epochs=200)
    best_baseline_r2 = results_without["r2_mean"].max()
    n_folds_beating = int(np.sum(np.array(gnn_cv["r2_per_fold"]) > best_baseline_r2))

    summary = {
        "n_population": len(population),
        "n_cases": int(population["is_case"].sum()),
        "n_controls": int((~population["is_case"]).sum()),
        "ablation_table": ablation_table.to_dict("records"),
        "gnn_cv_k": gnn_cv["k"],
        "gnn_cv_seeds": gnn_cv["seeds"],
        "gnn_cv_n_folds": gnn_cv["n_folds"],
        "gnn_test_r2_per_fold": gnn_cv["r2_per_fold"],
        "gnn_test_r2_mean": gnn_cv["r2_mean"],
        "gnn_test_r2_std": gnn_cv["r2_std"],
        "gnn_test_r2_min": gnn_cv["r2_min"],
        "gnn_test_r2_max": gnn_cv["r2_max"],
        "gnn_test_mae_mean": gnn_cv["mae_mean"],
        "gnn_n_folds_beating_best_baseline": n_folds_beating,
        "gnn_beats_best_baseline_on_mean": bool(gnn_cv["r2_mean"] > best_baseline_r2),
        "n_population_internal_edges": len(edges),
    }
    with open(cfg.data_processed / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Best baseline (without fru/dsx) R^2: {best_baseline_r2:.3f}  (5-fold CV mean)")
    print(f"GNN R^2, {gnn_cv['k']}-fold CV x {len(gnn_cv['seeds'])} seeds "
          f"({gnn_cv['n_folds']} fits): {gnn_cv['r2_mean']:.3f} +/- {gnn_cv['r2_std']:.3f} "
          f"(min {gnn_cv['r2_min']:.3f}, max {gnn_cv['r2_max']:.3f})")
    print(f"GNN beats best baseline on {n_folds_beating}/{gnn_cv['n_folds']} folds; "
          f"on the mean: {'YES' if summary['gnn_beats_best_baseline_on_mean'] else 'NO'}")
    print(f"{'='*60}")
    print("Saved figures to", figures_dir)
    print("Saved summary to", cfg.data_processed / "phase2_summary.json")


if __name__ == "__main__":
    main()
