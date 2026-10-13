"""Tests for the LIF engine on tiny networks where the answer is known."""

import numpy as np
import pytest
import scipy.sparse as sp

from courtship.lif import LIFParams, group_rate, simulate


def chain(n=5, weight=600.0):
    """0 -> 1 -> 2 -> ... a feed-forward chain, strongly excitatory.

    weight 600 x the default 0.02 mV/synapse = 12 mV per presynaptic spike,
    comfortably over the 7 mV from rest to threshold, so one spike drives
    one spike. At weight 200 it is 4 mV, which decays before the next spike
    arrives and never propagates -- correct behaviour, useless test.
    """
    rows = np.arange(n - 1)
    cols = rows + 1
    return sp.coo_matrix((np.full(n - 1, weight, np.float32), (rows, cols)),
                         shape=(n, n), dtype=np.float32).tocsr()


def test_no_stimulus_means_no_spikes():
    # The model has no spontaneous term. Any activity without input is a bug.
    r = simulate(chain(), np.array([], dtype=int), duration_ms=50.0, record_every=0)
    assert r["spike_counts"].sum() == 0


def test_stimulated_neuron_fires_at_about_the_requested_rate():
    p = LIFParams(stim_rate_hz=100.0)
    r = simulate(chain(), np.array([0]), duration_ms=1000.0, params=p, record_every=0)
    assert r["rates_hz"][0] == pytest.approx(100.0, rel=0.25)


def test_activity_propagates_along_the_chain():
    r = simulate(chain(), np.array([0]), duration_ms=500.0, record_every=0)
    counts = r["spike_counts"]
    assert counts[0] > 0, "stimulated neuron silent"
    assert counts[-1] > 0, "signal never reached the end of the chain"


def test_silencing_a_link_blocks_everything_downstream():
    # Silencing node 2 must stop 3 and 4 while leaving 0 and 1 alone. This is
    # the mechanism the whole ablation experiment rests on.
    r = simulate(chain(), np.array([0]), duration_ms=500.0,
                 silenced_idx=np.array([2]), record_every=0)
    counts = r["spike_counts"]
    assert counts[0] > 0 and counts[1] > 0
    assert counts[2] == 0, "silenced neuron fired"
    assert counts[3] == 0 and counts[4] == 0, "signal passed a silenced neuron"


def test_silencing_does_not_mutate_the_matrix():
    adj = chain()
    before = adj.copy()
    simulate(adj, np.array([0]), duration_ms=50.0,
             silenced_idx=np.array([1, 2]), record_every=0)
    assert (adj != before).nnz == 0, "simulate() modified the connectome"


def test_inhibition_suppresses_a_target():
    # 0 excites 2; 1 inhibits 2 harder. Driving both must leave 2 quieter
    # than driving 0 alone -- i.e. the sign is actually used.
    n = 3
    adj = sp.coo_matrix(
        (np.array([200.0, -400.0], np.float32), ([0, 1], [2, 2])),
        shape=(n, n), dtype=np.float32).tocsr()
    exc = simulate(adj, np.array([0]), duration_ms=500.0, record_every=0)
    both = simulate(adj, np.array([0, 1]), duration_ms=500.0, record_every=0)
    assert exc["spike_counts"][2] > both["spike_counts"][2]


def test_group_traces_have_the_expected_shape():
    gi = {"first": np.array([0]), "last": np.array([4])}
    r = simulate(chain(), np.array([0]), duration_ms=100.0, record_every=0,
                 track_groups=gi, bin_ms=5.0)
    assert set(r["group_traces_hz"]) == {"first", "last"}
    assert len(r["group_traces_hz"]["first"]) == 20  # 100ms / 5ms bins


def test_group_rate_handles_empty_group():
    assert np.isnan(group_rate(np.array([1.0, 2.0]), np.array([], dtype=int)))


def test_simulation_is_deterministic():
    a = simulate(chain(), np.array([0]), duration_ms=200.0, record_every=0)
    b = simulate(chain(), np.array([0]), duration_ms=200.0, record_every=0)
    assert np.array_equal(a["spike_counts"], b["spike_counts"])
