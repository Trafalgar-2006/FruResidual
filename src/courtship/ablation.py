"""Degree-matched ablation nulls.

Silencing 89 neurons that each carry thousands of synapses removes far
more drive than silencing 89 quiet ones, so "the song got weaker" is not
evidence about WHICH neurons were removed unless the comparison set
carries the same traffic. This is the same degree-confound correction
FruResidual's Phase 3 needed, in a different guise.

Matching is on total outgoing synapses, binned by quantile, sampling the
same number of controls from each bin as the target set has.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def out_synapses(adj: sp.csr_matrix) -> np.ndarray:
    """Total outgoing synaptic weight per neuron (sign ignored)."""
    return abs(adj).sum(axis=1).A1


def degree_matched_sample(
    target_idx: np.ndarray,
    out_syn: np.ndarray,
    rng: np.random.Generator,
    n_bins: int = 10,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a control set the same size as `target_idx`, matched on the
    distribution of outgoing synapses.

    Falls back to sampling from the nearest populated bin when a bin has
    too few candidates, rather than silently returning a short set -- a
    control set of the wrong size would make the comparison meaningless.
    """
    n = len(out_syn)
    forbidden = np.zeros(n, dtype=bool)
    forbidden[target_idx] = True
    if exclude is not None and len(exclude):
        forbidden[exclude] = True

    # Quantile bins over the whole population.
    edges = np.quantile(out_syn, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    bin_of = np.clip(np.searchsorted(edges, out_syn, side="right") - 1, 0, n_bins - 1)

    picked: list[int] = []
    for b in range(n_bins):
        need = int((bin_of[target_idx] == b).sum())
        if not need:
            continue
        pool = np.where((bin_of == b) & ~forbidden)[0]
        if len(pool) < need:
            # Bin exhausted: fall back to the nearest neurons by out-synapse
            # count across the whole population. The candidate window must
            # stay TIGHT -- widening it to need*20 made the "matched" null
            # effectively uniform whenever a bin ran short, which is exactly
            # the confound this function exists to remove.
            pool = np.where(~forbidden)[0]
            want = out_syn[target_idx][bin_of[target_idx] == b].mean()
            window = min(len(pool), max(2 * need, need + 5))
            pool = pool[np.argsort(np.abs(out_syn[pool] - want))[:window]]
        take = rng.choice(pool, size=need, replace=False)
        picked.extend(take.tolist())
        forbidden[take] = True

    out = np.array(picked, dtype=np.int64)
    assert len(out) == len(target_idx), f"control set {len(out)} != target {len(target_idx)}"
    return out


def empirical_p(observed_drop: float, null_drops: np.ndarray) -> float:
    """One-sided p: how often a matched random ablation drops the readout
    at least as much. (k+1)/(n+1) so p is never exactly 0.
    """
    k = int((null_drops >= observed_drop).sum())
    return (k + 1) / (len(null_drops) + 1)
