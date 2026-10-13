"""Phase 2 baseline models: Ridge regression + gradient boosting,
predicting the wiring-difference target (fruresidual.features.rewiring)
from non-leaking features (fruresidual.features.ml_features).

Uses k-fold cross-validation rather than a single train/test split --
with ~1,400 rows, a single 75/25 split gives a noisy estimate; CV is the
more defensible choice at this sample size. Baselines run before any GNN,
per docs/project_plan.md section (d), PHASE 2, step 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _models(random_state: int) -> dict:
    return {
        # Ridge is scale-sensitive twice over: its L2 penalty shrinks
        # coefficients in the units they are measured in, so without
        # standardization a feature in the thousands
        # (total_connectivity_weight) is penalized far less per unit than
        # one in [0, 3] (connectivity_entropy), and the fit is driven by
        # arbitrary units rather than signal. Scaling lives inside the
        # pipeline so it is refit per CV fold -- fitting the scaler on all
        # rows first would leak test-fold statistics into training.
        # LightGBM is scale-invariant and deliberately left unscaled.
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gradient_boosting": LGBMRegressor(random_state=random_state, verbosity=-1, n_estimators=100),
    }


def evaluate_baselines(X: pd.DataFrame, y: pd.Series, cv: int = 5, random_state: int = 0) -> pd.DataFrame:
    """Cross-validated R^2 and MAE for each baseline model."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, model in _models(random_state).items():
        r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
        mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        rows.append(
            {
                "model": name,
                "r2_mean": r2.mean(),
                "r2_std": r2.std(),
                "mae_mean": mae.mean(),
                "mae_std": mae.std(),
            }
        )
    return pd.DataFrame(rows)


def out_of_fold_predictions(
    X: pd.DataFrame, y: pd.Series, model_name: str = "gradient_boosting", cv: int = 5, random_state: int = 0
) -> np.ndarray:
    """Out-of-fold predictions for the given model -- each row predicted
    by a model that never saw it during training, so residuals (y - pred)
    are honest per-neuron error estimates rather than overfit train-set
    residuals. This is the actual Phase 2 deliverable: per-neuron residual
    scores that feed Phase 3.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    model = _models(random_state)[model_name]
    return cross_val_predict(model, X, y, cv=kf)


def ablation_fold_test(
    X_without: pd.DataFrame, X_with: pd.DataFrame, y: pd.Series, cv: int = 5, random_state: int = 0
) -> pd.DataFrame:
    """Per-fold R^2 for both ablation arms on identical splits, plus the
    paired comparison between them.

    Exists because "does adding fru/dsx help" is a PAIRED question, and
    the mean/std that evaluate_baselines reports cannot answer it: that
    r2_std is variability across folds, not the standard error of the
    difference, so eyeballing two overlapping error bars is the wrong
    test. Both arms are scored on the same KFold splits (same cv and
    random_state as evaluate_baselines), so fold i of one arm pairs with
    fold i of the other.

    Wilcoxon's smallest attainable p at cv=5 is 0.0625, so a unanimous
    direction across five folds is the strongest evidence this fold
    count can produce -- not weak evidence.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, model in _models(random_state).items():
        without = cross_val_score(model, X_without, y, cv=kf, scoring="r2")
        with_ = cross_val_score(model, X_with, y, cv=kf, scoring="r2")
        delta = with_ - without
        # wilcoxon drops zero-differences and errors when none are left;
        # identical arms are "no evidence of a difference", i.e. p=1.
        p = 1.0 if np.allclose(delta, 0) else float(wilcoxon(with_, without)[1])
        rows.append(
            {
                "model": name,
                "r2_without_per_fold": [float(x) for x in without],
                "r2_with_per_fold": [float(x) for x in with_],
                "delta_mean": float(delta.mean()),
                "delta_min": float(delta.min()),
                "delta_max": float(delta.max()),
                "n_folds_improving": int((delta > 0).sum()),
                "n_folds": cv,
                "wilcoxon_p": p,
            }
        )
    return pd.DataFrame(rows)
