"""Tests for degree-matched ablation nulls."""

import numpy as np
import pytest
import scipy.sparse as sp

from courtship.ablation import degree_matched_sample, empirical_p, out_synapses


def random_adj(n=500, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n, 4000)
    cols = rng.integers(0, n, 4000)
    vals = rng.integers(1, 50, 4000).astype(np.float32)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def test_out_synapses_ignores_sign():
    adj = sp.coo_matrix((np.array([5.0, -7.0], np.float32), ([0, 0], [1, 2])),
                        shape=(3, 3)).tocsr()
    assert out_synapses(adj)[0] == pytest.approx(12.0)


def test_control_set_is_the_same_size_and_disjoint():
    adj = random_adj()
    out = out_synapses(adj)
    rng = np.random.default_rng(0)
    target = rng.choice(len(out), 40, replace=False)
    ctrl = degree_matched_sample(target, out, rng)
    assert len(ctrl) == len(target)
    assert not set(ctrl) & set(target), "control overlaps the target"


def test_control_set_matches_the_degree_distribution():
    # The whole point: a control drawn uniformly would have a very different
    # mean out-degree from a deliberately high-degree target.
    adj = random_adj()
    out = out_synapses(adj)
    rng = np.random.default_rng(1)
    target = np.argsort(out)[-40:]  # the 40 busiest neurons
    matched = degree_matched_sample(target, out, rng)
    uniform = rng.choice(len(out), 40, replace=False)
    err_matched = abs(out[matched].mean() - out[target].mean())
    err_uniform = abs(out[uniform].mean() - out[target].mean())
    assert err_matched < err_uniform, "matching is no better than uniform sampling"


def test_excluded_neurons_are_never_drawn():
    adj = random_adj()
    out = out_synapses(adj)
    rng = np.random.default_rng(2)
    target = rng.choice(len(out), 20, replace=False)
    banned = rng.choice(np.setdiff1d(np.arange(len(out)), target), 300, replace=False)
    ctrl = degree_matched_sample(target, out, rng, exclude=banned)
    assert not set(ctrl) & set(banned)


def test_empirical_p_is_never_zero():
    assert empirical_p(10.0, np.zeros(99)) == pytest.approx(1 / 100)


def test_empirical_p_is_one_when_nulls_all_exceed():
    assert empirical_p(0.0, np.ones(9)) == pytest.approx(1.0)
