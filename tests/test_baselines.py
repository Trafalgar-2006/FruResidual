import numpy as np
import pandas as pd
import pytest

from fruresidual.models.baselines import (
    ablation_fold_test,
    evaluate_baselines,
    out_of_fold_predictions,
)


def _synthetic_regression(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
            "signal": rng.normal(size=n),
        }
    )
    y = pd.Series(3 * X["signal"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_baselines_learn_real_signal():
    X, y = _synthetic_regression()
    results = evaluate_baselines(X, y, cv=5)
    assert (results["r2_mean"] > 0.7).all()  # should easily learn y = 3*signal + small noise


def test_baselines_fail_on_pure_noise():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"noise1": rng.normal(size=300), "noise2": rng.normal(size=300)})
    y = pd.Series(rng.normal(size=300))  # unrelated to X
    results = evaluate_baselines(X, y, cv=5)
    assert (results["r2_mean"] < 0.1).all()  # shouldn't find signal that isn't there


def test_out_of_fold_predictions_shape_and_correlate_with_target():
    X, y = _synthetic_regression()
    preds = out_of_fold_predictions(X, y, model_name="gradient_boosting", cv=5)
    assert len(preds) == len(y)
    assert np.corrcoef(preds, y)[0, 1] > 0.7


def test_ablation_feature_improves_r2_when_informative():
    X, y = _synthetic_regression()
    without_signal = X[["noise1", "noise2"]]
    with_signal = X[["noise1", "noise2", "signal"]]
    r2_without = evaluate_baselines(without_signal, y, cv=5)["r2_mean"].max()
    r2_with = evaluate_baselines(with_signal, y, cv=5)["r2_mean"].max()
    assert r2_with > r2_without + 0.5  # the ablation should show a large, clear gap


def test_ablation_fold_test_is_paired_and_unanimous_on_real_signal():
    X, y = _synthetic_regression()
    fold_test = ablation_fold_test(X[["noise1", "noise2"]], X, y, cv=5)
    for row in fold_test.to_dict("records"):
        # Adding the one informative feature must help in EVERY fold, and
        # the paired test must see it. 0.0625 is Wilcoxon's floor at n=5.
        assert row["n_folds_improving"] == 5, row["model"]
        assert row["delta_min"] > 0, row["model"]
        assert row["wilcoxon_p"] == pytest.approx(0.0625), row["model"]
        # Per-fold arrays must be paired: same length, one per fold.
        assert len(row["r2_without_per_fold"]) == len(row["r2_with_per_fold"]) == 5


def test_ablation_fold_test_reports_no_difference_for_identical_arms():
    X, y = _synthetic_regression()
    fold_test = ablation_fold_test(X, X, y, cv=5)
    # Identical arms means zero differences, which wilcoxon cannot rank --
    # it must come back as "no evidence", not an exception.
    assert (fold_test["delta_mean"] == 0).all()
    assert (fold_test["n_folds_improving"] == 0).all()
    assert (fold_test["wilcoxon_p"] == 1.0).all()


def test_ablation_fold_test_mean_matches_evaluate_baselines():
    X, y = _synthetic_regression()
    without = X[["noise1", "noise2"]]
    fold_test = ablation_fold_test(without, X, y, cv=5, random_state=0)
    baselines = evaluate_baselines(without, y, cv=5, random_state=0).set_index("model")
    # Same splits, same models: the fold test's per-fold R^2 must average to
    # exactly what evaluate_baselines reports, or the two are not comparable
    # and the paired test says nothing about the reported ablation.
    for row in fold_test.to_dict("records"):
        assert np.mean(row["r2_without_per_fold"]) == pytest.approx(
            baselines.loc[row["model"], "r2_mean"]
        ), row["model"]
