"""Integration test against the real cell-type-resolution Phase 2 outputs.
Skipped if scripts/run_phase2_celltype.py hasn't been run yet -- mirrors
the pattern in test_phase2_integration.py.

Pins the claims docs/phase2_celltype_results.md and docs/paper.md (§4.6)
actually make, so a rerun that silently changes them fails here instead
of quietly contradicting the writeup.
"""

import json

import pandas as pd
import pytest

from fruresidual.config import load_config

cfg = load_config()
_summary_path = cfg.data_processed / "phase2ct_summary.json"
_scores_path = cfg.data_processed / "phase2ct_residual_scores.feather"

pytestmark = pytest.mark.skipif(
    not _summary_path.exists(),
    reason=f"cell-type summary not found (expected at {_summary_path}); "
    "run scripts/run_phase2_celltype.py first",
)


@pytest.fixture(scope="module")
def summary():
    with open(_summary_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fold_test(summary):
    return {row["model"]: row for row in summary["ablation_fold_test"]}


def test_finer_target_separates_better_than_coarse(summary):
    # The whole point of this phase: +0.659 here against the coarse
    # target's +0.259. Anything near the coarse value means the finer
    # vocabulary silently failed to build.
    assert summary["separation_rank_biserial"] > 0.6
    assert summary["case_mean"] > 2 * summary["control_mean"]


def test_structure_alone_still_predicts_the_finer_target(summary):
    # Gradient boosting without fru/dsx -- the project's central framing.
    without = [
        r for r in summary["ablation_table"]
        if r["ablation"] == "without_fru_dsx" and r["model"] == "gradient_boosting"
    ]
    assert len(without) == 1
    assert without[0]["r2_mean"] > 0.35


def test_ridge_ablation_is_unanimous_across_folds(fold_test):
    # The resolution-dependent finding: at cell-type resolution fru/dsx
    # helps the LINEAR model in every fold. 0.0625 is Wilcoxon's floor at
    # n=5, so unanimity is the strongest signal this fold count allows.
    ridge = fold_test["ridge"]
    assert ridge["n_folds_improving"] == 5
    assert ridge["delta_min"] > 0
    assert ridge["wilcoxon_p"] == pytest.approx(0.0625)


def test_gradient_boosting_ablation_is_not_unanimous(fold_test):
    # The other half of the finding -- the tree model does NOT need it.
    # If this ever goes unanimous too, the "resolution-dependent" claim
    # in both docs becomes a plain "fru/dsx helps", which is a different
    # conclusion and must not pass silently.
    gb = fold_test["gradient_boosting"]
    assert gb["n_folds_improving"] < 5
    assert gb["delta_min"] < 0


def test_fold_test_means_match_the_reported_ablation_table(summary, fold_test):
    # The paired test must be measuring the same fits the ablation table
    # reports, or the table and the fold counts are two different things.
    for row in summary["ablation_table"]:
        key = "r2_with_per_fold" if row["ablation"] == "with_fru_dsx" else "r2_without_per_fold"
        per_fold = fold_test[row["model"]][key]
        assert sum(per_fold) / len(per_fold) == pytest.approx(row["r2_mean"]), (
            row["model"],
            row["ablation"],
        )


def test_gnn_still_loses_to_gradient_boosting(summary):
    # Refit on the finer target and still negative -- stated in both docs.
    assert summary["gnn_beats_best_baseline_on_mean"] is False
    assert summary["gnn_n_folds_beating_best_baseline"] < summary["gnn_cv_n_folds"] / 2


def test_gnn_is_less_erratic_on_the_finer_target(summary):
    # The documented improvement: variance roughly a third of the coarse
    # run's +/-0.665, which is why part of the instability was the target.
    assert summary["gnn_test_r2_std"] < 0.4
    assert summary["gnn_test_r2_mean"] > 0


def test_no_controls_look_rewired_at_this_resolution(summary):
    # Closes the lead docs/phase2_results.md left open (5 at coarse
    # resolution). If this ever comes back non-zero, that thread reopens.
    assert summary["n_control_outliers_above_case_q99"] == 0


def test_residual_scores_are_out_of_fold_and_cover_the_population(summary):
    scores = pd.read_feather(_scores_path)
    assert len(scores) == summary["n_scored"]
    assert scores["bodyId"].duplicated().sum() == 0
    # residual is defined as actual - predicted; check it wasn't recomputed
    # against a different column somewhere along the way.
    recomputed = scores["actual_wiring_difference"] - scores["predicted_wiring_difference"]
    assert (recomputed - scores["residual"]).abs().max() < 1e-9
