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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score


def _models(random_state: int) -> dict:
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
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
