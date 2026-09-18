import numpy as np
import pandas as pd
import pytest

from fruresidual.models.gnn import build_graph, train_gnn


def _synthetic_population_graph(n=200, seed=0):
    rng = np.random.default_rng(seed)
    body_ids = list(range(1000, 1000 + n))
    features = pd.DataFrame(
        {
            "bodyId": body_ids,
            "signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    targets = pd.Series(3 * features["signal"].values + rng.normal(scale=0.1, size=n))

    # random sparse edges among the population (not meaningful, just
    # exercises the message-passing path)
    n_edges = n * 3
    src = rng.choice(body_ids, size=n_edges)
    dst = rng.choice(body_ids, size=n_edges)
    edges = pd.DataFrame({"source": src, "target": dst, "weight": 1})
    return features, targets, edges


def test_build_graph_shapes():
    features, targets, edges = _synthetic_population_graph()
    data = build_graph(features, targets, edges)
    assert data.num_nodes == len(features)
    assert data.num_node_features == 2  # signal, noise (bodyId dropped)
    assert data.edge_index.shape[0] == 2


def test_gnn_learns_real_signal():
    features, targets, edges = _synthetic_population_graph(n=400)
    data = build_graph(features, targets, edges)
    result = train_gnn(data, epochs=100, hidden_channels=16, random_state=0)
    assert result["test_r2"] > 0.5  # should pick up the linear signal via self-features
    assert len(result["predictions"]) == data.num_nodes


def test_gnn_predictions_correlate_with_target():
    features, targets, edges = _synthetic_population_graph(n=400)
    data = build_graph(features, targets, edges)
    result = train_gnn(data, epochs=100, hidden_channels=16, random_state=0)
    corr = np.corrcoef(result["predictions"], targets.values)[0, 1]
    assert corr > 0.7


def test_gnn_stable_with_unscaled_large_magnitude_features():
    """Regression test for a real failure mode (2026-09-18): raw,
    unstandardized features (e.g. a connectivity-weight sum in the
    thousands) against a target in [0, 1] made the GNN diverge outright
    (R^2 = -7250, MAE = 2.58 on a target maxing under 0.7). build_graph
    must standardize features internally so this can't recur silently.
    """
    rng = np.random.default_rng(0)
    n = 300
    body_ids = list(range(2000, 2000 + n))
    signal = rng.normal(size=n)
    features = pd.DataFrame(
        {
            "bodyId": body_ids,
            "signal": signal,
            "huge_unscaled_feature": rng.normal(loc=5000, scale=2000, size=n),  # thousands-scale
        }
    )
    # target on a tiny, [0, 1]-bounded scale, like the real wiring-difference metric
    targets = pd.Series(np.clip(0.05 + 0.02 * signal + rng.normal(scale=0.01, size=n), 0, 1))
    edges = pd.DataFrame({"source": [], "target": [], "weight": []})

    data = build_graph(features, targets, edges)
    result = train_gnn(data, epochs=100, hidden_channels=16, random_state=0)

    assert result["test_r2"] > -1.0  # sanity floor -- catastrophic divergence would be far below this
    assert result["test_mae"] < 1.0  # target itself never exceeds ~0.1 here
