"""Phase 3: cluster the dimorphic population on connectivity embeddings
(partner-superclass profiles) and test whether the residual set is
enriched in any cluster beyond chance. See docs/project_plan.md section
(d), PHASE 3, step 3.

Leiden (via igraph/leidenalg, both already project dependencies) on a
cosine-similarity kNN graph -- the plan's own suggested method.
"""

from __future__ import annotations

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from fruresidual.eval.enrichment import category_enrichment


def build_profile_matrix(profile_long: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Pivots long-format (bodyId, category_col, weight) into a dense
    bodyId x category matrix, L2-normalized per row (so clustering is
    driven by connectivity *shape*, not raw connection volume -- a
    heavily-connected and a sparsely-connected neuron with the same
    partner-superclass mix should cluster together).
    """
    pivot = profile_long.pivot_table(
        index="bodyId", columns=category_col, values="weight", fill_value=0
    )
    norms = np.linalg.norm(pivot.values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pd.DataFrame(pivot.values / norms, index=pivot.index, columns=pivot.columns)


def build_knn_graph(matrix: pd.DataFrame, k: int = 15) -> ig.Graph:
    """Cosine-similarity kNN graph (rows already L2-normalized, so cosine
    similarity is just the dot product) over the neurons in `matrix`."""
    k_eff = min(k + 1, len(matrix))  # +1: NearestNeighbors includes self
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(matrix.values)
    distances, indices = nn.kneighbors(matrix.values)

    edges, weights = [], []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices, strict=True)):
        for dist, j in zip(dist_row, idx_row, strict=True):
            if i == j:
                continue
            edges.append((i, j))
            weights.append(1 - dist)  # cosine distance -> similarity

    g = ig.Graph(n=len(matrix), edges=edges, edge_attrs={"weight": weights})
    g = g.simplify(combine_edges="max")  # kNN is not symmetric; dedupe A->B / B->A
    g.vs["bodyId"] = matrix.index.tolist()
    return g


def leiden_cluster(graph: ig.Graph, resolution: float = 0.05, random_state: int = 0) -> pd.Series:
    """Returns a pd.Series of cluster labels indexed by bodyId.

    default resolution=0.05, not leidenalg's own default of 1.0: on a
    synthetic two-well-separated-group test (2 real, fully disconnected
    60-node components), resolution=1.0 fragmented them into 6 pieces
    and even resolution=0.3 still split each component in two. A sweep
    down to 0.001 confirmed 0.001-0.05 all recover the true 2-component
    structure cleanly; 0.05 is the least aggressive-smoothing choice in
    that verified-clean range. This is a real property of
    RBConfigurationVertexPartition on dense cosine-similarity kNN graphs,
    not specific to this one dataset -- run a resolution sensitivity
    sweep (see scripts/run_phase3.py) rather than trust one value
    blindly on real data.
    """
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight",
        seed=random_state,
    )
    labels = pd.Series(partition.membership, index=graph.vs["bodyId"], name="cluster")
    return labels.rename_axis("bodyId")


def cluster_enrichment(
    cluster_labels: pd.Series, residual_body_ids: set[int], alpha: float = 0.05, min_odds_ratio: float = 2.0
) -> pd.DataFrame:
    """Tests whether residual neurons are over-represented in any Leiden
    cluster relative to the full (residual + explained) population."""
    is_residual = cluster_labels.index.isin(residual_body_ids)
    return category_enrichment(
        cluster_labels[is_residual], cluster_labels, alpha=alpha, min_odds_ratio=min_odds_ratio
    )


def label_permutation_null(
    graph: ig.Graph,
    residual_body_ids: set[int],
    resolution: float,
    n_permutations: int = 1000,
    random_state: int = 0,
    min_odds_ratio: float = 2.0,
) -> dict:
    """Empirical null for the cluster-enrichment result.

    `cluster_enrichment` reports a Fisher/BH p-value, which assumes the
    cluster assignment is independent of the residual labels. It is not:
    the clusters were derived from the same connectivity data, and the
    resolution was chosen by a sweep. That makes the parametric p-value
    optimistic by an unknown amount. docs/project_plan.md section (e)
    calls for a degree-preserving null precisely for this reason and it
    was never built.

    Method: hold the graph and its clustering FIXED, and permute which
    neurons carry the residual label, `n_permutations` times. The
    observed statistic is the maximum enrichment odds ratio across
    clusters; the null distribution is that same statistic under random
    labelling. The empirical p-value is the fraction of permutations
    reaching the observed value.

    Permuting labels rather than rewiring edges is the right null here.
    Rewiring would test "is this graph unusual", which is not the claim.
    The claim is "residual neurons sit in one cluster more than chance",
    so chance means the same graph and the same clusters with the
    residual label scattered at random -- and it preserves every
    degree, every community, and the residual set's size exactly.
    """
    labels = leiden_cluster(graph, resolution=resolution, random_state=random_state)
    body_ids = labels.index.to_numpy()
    n_residual = int(pd.Series(body_ids).isin(residual_body_ids).sum())

    def _max_odds(residual_mask: np.ndarray) -> float:
        result = category_enrichment(
            labels[residual_mask], labels, min_odds_ratio=min_odds_ratio
        )
        return float(result["odds_ratio"].max()) if len(result) else float("nan")

    observed_mask = pd.Series(body_ids).isin(residual_body_ids).to_numpy()
    observed = _max_odds(observed_mask)

    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = np.zeros(len(body_ids), dtype=bool)
        shuffled[rng.choice(len(body_ids), size=n_residual, replace=False)] = True
        null[i] = _max_odds(shuffled)

    # +1 in numerator and denominator: an empirical p-value must never be
    # exactly 0, since n_permutations only bounds it from above.
    n_ge = int(np.sum(null >= observed))
    return {
        "observed_max_odds_ratio": observed,
        "n_permutations": n_permutations,
        "null_mean": float(np.nanmean(null)),
        "null_std": float(np.nanstd(null)),
        "null_p95": float(np.nanpercentile(null, 95)),
        "null_max": float(np.nanmax(null)),
        "n_null_ge_observed": n_ge,
        "empirical_p_value": float((n_ge + 1) / (n_permutations + 1)),
        "resolution": resolution,
        "n_residual": n_residual,
        "n_total": len(body_ids),
    }
