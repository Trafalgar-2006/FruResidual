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
    n = data.num_nodes
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

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
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(test_y - test_pred)))

    return {
        "model": model,
        "test_r2": r2,
        "test_mae": mae,
        "predictions": all_pred,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
