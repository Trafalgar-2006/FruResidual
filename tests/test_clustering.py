import numpy as np
import pandas as pd

from fruresidual.features.clustering import (
    build_knn_graph,
    build_profile_matrix,
    cluster_enrichment,
    leiden_cluster,
)


def _two_group_profile(n_per_group=60, seed=0):
    """Two well-separated groups in connectivity-profile space: group A
    connects mostly to partner "X" (with some "Y"), group B mostly to "Y"
    (with some "X"). Leiden should recover this as two clusters.

    Uses continuous Gaussian-distributed weights (clipped at 0), not
    near-identical integers -- tried integers-with-tiny-noise first and
    it made every within-group point nearly equidistant from every
    other, causing pathological near-tied kNN neighbor selection that
    fragmented each real group into several disconnected sub-cliques
    instead of one connected community. Smooth continuous variance
    avoids that failure mode and is also more representative of real
    connectivity data anyway.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_per_group):
        rows.append({"bodyId": i, "partner_superclass": "X", "weight": max(0.0, rng.normal(100, 20))})
        rows.append({"bodyId": i, "partner_superclass": "Y", "weight": max(0.0, rng.normal(10, 5))})
    for i in range(n_per_group, 2 * n_per_group):
        rows.append({"bodyId": i, "partner_superclass": "X", "weight": max(0.0, rng.normal(10, 5))})
        rows.append({"bodyId": i, "partner_superclass": "Y", "weight": max(0.0, rng.normal(100, 20))})
    return pd.DataFrame(rows)


def test_build_profile_matrix_rows_are_unit_norm():
    profile = _two_group_profile()
    matrix = build_profile_matrix(profile, "partner_superclass")
    norms = np.linalg.norm(matrix.values, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_leiden_recovers_two_well_separated_groups():
    profile = _two_group_profile(n_per_group=60)
    matrix = build_profile_matrix(profile, "partner_superclass")
    graph = build_knn_graph(matrix, k=10)

    # Verified directly (2026-09-18): the two groups form exactly 2 fully
    # disconnected connected-components in this graph, so this is a
    # regression test that leiden_cluster's default resolution recovers
    # that cleanly -- NOT a claim that resolution=0.3 is universally
    # correct (see the docstring in clustering.py: resolution=1.0
    # over-fragmented this same graph into 6 pieces).
    assert graph.connected_components().sizes() == [60, 60]

    labels = leiden_cluster(graph, random_state=0)  # uses the module default (0.3)

    assert labels.nunique() >= 2
    group_a = labels.loc[list(range(60))]
    group_b = labels.loc[list(range(60, 120))]
    # each group should be dominated by a single cluster label, and the
    # two groups' dominant labels should differ
    assert group_a.mode()[0] != group_b.mode()[0]
    assert (group_a == group_a.mode()[0]).mean() > 0.8
    assert (group_b == group_b.mode()[0]).mean() > 0.8


def test_cluster_enrichment_detects_concentrated_residual():
    profile = _two_group_profile(n_per_group=60)
    matrix = build_profile_matrix(profile, "partner_superclass")
    graph = build_knn_graph(matrix, k=10)
    labels = leiden_cluster(graph, resolution=1.0, random_state=0)

    # residual = all of group A (bodyIds 0-59) -- should show up as
    # significantly enriched in whichever cluster group A landed in
    residual_ids = set(range(60))
    result = cluster_enrichment(labels, residual_ids)
    assert result["significant"].any()


def test_permutation_null_rejects_a_structureless_graph():
    """If the residual label has nothing to do with the clustering, the
    observed enrichment should sit inside the null distribution. This is
    the failure mode the parametric BH p-value cannot see: it assumes
    cluster assignment is independent of the labels, when the clusters
    came from the same data and the resolution was chosen by a sweep.
    """
    import numpy as np

    from fruresidual.features.clustering import build_knn_graph, label_permutation_null

    rng = np.random.default_rng(0)
    # 120 neurons, random profiles -> no real community structure
    matrix = pd.DataFrame(
        rng.normal(size=(120, 8)),
        index=range(1000, 1120),
        columns=[f"c{i}" for i in range(8)],
    )
    graph = build_knn_graph(matrix, k=10)
    residual = set(rng.choice(matrix.index, size=60, replace=False).tolist())

    out = label_permutation_null(graph, residual, resolution=0.1,
                                 n_permutations=100, random_state=0)
    assert out["n_permutations"] == 100
    assert out["n_residual"] == 60
    # an empirical p-value can never be exactly 0
    assert 0 < out["empirical_p_value"] <= 1.0
    # random labels on a structureless graph must not look significant
    assert out["empirical_p_value"] > 0.05


def test_permutation_null_detects_a_planted_cluster_signal():
    """Guards over-correction: a real, planted association must still
    come out significant, or the null would reject everything."""
    import numpy as np

    from fruresidual.features.clustering import build_knn_graph, label_permutation_null

    rng = np.random.default_rng(0)
    # two well-separated groups; residual label given to one of them
    a = rng.normal(loc=0.0, scale=0.1, size=(60, 8))
    b = rng.normal(loc=8.0, scale=0.1, size=(60, 8))
    matrix = pd.DataFrame(
        np.vstack([a, b]), index=range(2000, 2120),
        columns=[f"c{i}" for i in range(8)],
    )
    graph = build_knn_graph(matrix, k=10)
    residual = set(range(2000, 2060))  # exactly the first group

    out = label_permutation_null(graph, residual, resolution=0.1,
                                 n_permutations=100, random_state=0)
    assert out["empirical_p_value"] <= 0.05
    assert out["observed_max_odds_ratio"] > out["null_p95"]
