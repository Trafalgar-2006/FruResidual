"""Phase 2 GNN: a small GraphSAGE regressor on the population-internal
connectome subgraph, predicting the same wiring-difference target as the
baselines. Per docs/project_plan.md section (d), PHASE 2, step 3 ("if the
GNN doesn't beat gradient boosting, report that") -- this module reports
honestly, it doesn't tune until it wins.

Population size (~1,400 nodes) is small enough to train full-batch on a
4GB GPU -- no neighbor sampling needed, unlike a GNN over the full 165K-
neuron connectome.
"""

from __future__ import annotations

import os

# Must be set before torch is imported. On Windows, PyTorch's bundled
# OpenMP/MKL DLLs conflict with numpy's own (conda-forge numpy + pip
# torch, both linking their own copies) -- without this, a numpy linear-
# algebra call (e.g. np.corrcoef) made after any torch op in the same
# process can hard-crash with "Fatal Python error: Aborted", reproduced
# 2026-09-17. This is the standard, documented workaround, not a fix for
# the root cause (which would mean rebuilding one of the two stacks).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:  # pragma: no cover
    raise ImportError("torch_geometric is required for fruresidual.models.gnn") from e


class WiringDiffGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x).squeeze(-1)


def build_graph(
    features: pd.DataFrame, targets: pd.Series, edges: pd.DataFrame, device: str = "cpu"
) -> Data:
    """features: rows indexed/ordered by bodyId (bodyId column dropped
    before use). targets: same order, one wiring_difference value per
    node. edges: (source, target) bodyId pairs, subset to nodes present
    in `features`.
    """
    body_ids = features["bodyId"].tolist()
    id_to_idx = {b: i for i, b in enumerate(body_ids)}

    # Z-score standardization is required here, not optional: unlike the
    # tree-based baselines (scale-invariant by construction -- this is
    # exactly why the same raw features worked fine for them), a neural
    # net trained on raw features (e.g. total_connectivity_weight in the
    # thousands) against a target in [0, 1] diverges outright -- verified
    # 2026-09-17: unscaled features gave R^2 = -7250, MAE = 2.58 against
    # a target whose max is under 0.7. std has a floor to avoid div-by-
    # zero on any constant feature column.
    x_raw = features.drop(columns=["bodyId"]).values.astype(float)
    x_mean, x_std = x_raw.mean(axis=0), x_raw.std(axis=0)
    x_scaled = (x_raw - x_mean) / np.maximum(x_std, 1e-8)

    x = torch.tensor(x_scaled, dtype=torch.float)
    y = torch.tensor(targets.values, dtype=torch.float)

    valid_edges = edges[edges["source"].isin(id_to_idx) & edges["target"].isin(id_to_idx)]
    src = valid_edges["source"].map(id_to_idx).values
    dst = valid_edges["target"].map(id_to_idx).values
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y).to(device)


def _fit_gnn(
    data: Data,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden_channels: int = 32,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    random_state: int = 0,
    device: str = "cpu",
) -> dict:
    """One transductive fit against explicit node index sets. Shared by
    train_gnn (one random split) and cross_validate_gnn (k folds) so both
    train identically and only the split strategy differs.
    """
    # Seeds BOTH sources of randomness. The caller's numpy rng only
    # controls the split; model weight init draws from torch's global
    # RNG, so without this two runs with the same random_state start from
    # different weights and return different R^2 -- which is exactly what
    # happened: a documented GNN R^2 of 0.735 came back as 0.702 on a
    # later rerun (2026-09-19).
    torch.manual_seed(random_state)

    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    model = WiringDiffGNN(data.num_node_features, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index)
            val_loss = F.mse_loss(val_pred[val_mask], data.y[val_mask]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_pred = model(data.x, data.edge_index).cpu().numpy()

    y_np = data.y.cpu().numpy()
    test_pred, test_y = all_pred[test_idx], y_np[test_idx]
    ss_res = np.sum((test_y - test_pred) ** 2)
    ss_tot = np.sum((test_y - test_y.mean()) ** 2)
    return {
        "model": model,
        "predictions": all_pred,
        "test_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "test_mae": float(np.mean(np.abs(test_y - test_pred))),
    }


def cross_validate_gnn(
    data: Data,
    k: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2),
    hidden_channels: int = 32,
    epochs: int = 200,
    device: str = "cpu",
) -> dict:
    """K-fold cross-validation over the graph's nodes, repeated across
    `seeds` to also capture weight-initialization variance.

    Exists because a single train/val/test split is not comparable to the
    sklearn baselines, which are already k-fold cross-validated means --
    comparing one GNN split against a CV mean is what originally made the
    GNN look like it won. Every node is held out exactly once per seed,
    so `r2_mean` is a like-for-like counterpart to
    `evaluate_baselines(..., cv=5)["r2_mean"]`.

    Transductive: the whole graph (all edges, all features) is visible
    every fold; only the LABELS of held-out nodes are withheld via the
    loss mask. That is the standard protocol for node-level regression on
    a single fixed graph, and it is the same protocol train_gnn uses --
    it is not a leak of the target.
    """
    n = data.num_nodes
    fold_r2, fold_mae = [], []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        folds = np.array_split(perm, k)
        for i in range(k):
            test_idx = folds[i]
            val_idx = folds[(i + 1) % k]  # next fold for early-stopping selection
            train_idx = np.concatenate([folds[j] for j in range(k) if j not in (i, (i + 1) % k)])
            res = _fit_gnn(
                data, train_idx, val_idx, test_idx,
                hidden_channels=hidden_channels, epochs=epochs,
                random_state=seed, device=device,
            )
            fold_r2.append(res["test_r2"])
            fold_mae.append(res["test_mae"])

    fold_r2 = np.asarray(fold_r2, dtype=float)
    return {
        "r2_per_fold": [float(x) for x in fold_r2],
        "r2_mean": float(fold_r2.mean()),
        "r2_std": float(fold_r2.std()),
        "r2_min": float(fold_r2.min()),
        "r2_max": float(fold_r2.max()),
        "mae_mean": float(np.mean(fold_mae)),
        "n_folds": len(fold_r2),
        "k": k,
        "seeds": list(seeds),
    }


def train_gnn(
    data: Data,
    hidden_channels: int = 32,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
) -> dict:
    """Trains WiringDiffGNN transductively (full graph in memory, masked
    loss) with a random train/val/test node split. Returns the trained
    model, final test R^2/MAE, and predictions for every node (so
    residuals can be computed for the full population, matching the
    baselines' out-of-fold predictions).
    """
    # random_state must seed BOTH sources of randomness here. The numpy
    # rng below only controls the train/val/test split; model weight init
    # draws from torch's global RNG, so without this line two runs with
    # the same random_state start from different weights and return
    # different R^2 -- which is exactly what happened: a documented
    # GNN R^2 of 0.735 came back as 0.702 on a later rerun (2026-09-19).
    torch.manual_seed(random_state)

    n = data.num_nodes
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    fit = _fit_gnn(
        data, train_idx, val_idx, test_idx,
        hidden_channels=hidden_channels, epochs=epochs, lr=lr,
        weight_decay=weight_decay, random_state=random_state, device=device,
    )
    model, all_pred = fit["model"], fit["predictions"]
    r2, mae = fit["test_r2"], fit["test_mae"]

    return {
        "model": model,
        "test_r2": r2,
        "test_mae": mae,
        "predictions": all_pred,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
