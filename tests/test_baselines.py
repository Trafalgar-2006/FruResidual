import numpy as np
import pandas as pd

from fruresidual.models.baselines import evaluate_baselines, out_of_fold_predictions


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
