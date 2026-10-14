"""Leaky integrate-and-fire simulation over a signed connectome.

The scheme follows Shiu et al. (Nature 2024), who ran this on the FlyWire
connectome and validated it by predicting proboscis extension and leg
grooming, then confirming both in live flies: every neuron is a LIF unit,
every synapse injects a fixed quantum of current scaled by its synapse
count, and the sign comes from the presynaptic transmitter.

The constants below are a standard LIF parameterisation, not a claim to
reproduce that paper's exact values. What makes the model trustworthy
here is behavioural validation, not constant-matching: stimulating a
known input must activate known downstream partners, in the documented
order, with plausible latency -- and must NOT activate the pathway when
run backwards. `scripts/run_simulation.py` checks exactly that.

Runs on GPU if available. The matrix is ~25.5M non-zeros, so one
timestep is one sparse mat-vec; nothing here needs more than ~1GB.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class LIFParams:
    """Membrane voltages in mV, times in ms."""

    v_rest: float = -52.0
    v_reset: float = -52.0
    v_threshold: float = -45.0
    tau_m: float = 20.0          # membrane time constant
    refractory_ms: float = 2.2
    # Current delivered by ONE synapse when its presynaptic neuron fires.
    # Edge weights are raw synapse COUNTS (mean 4.7, max 2,591), so this
    # multiplies them -- it is the single most important free parameter
    # and decides whether activity dies out or runs away.
    #
    # Calibrated 2026-10-13 by sweeping it against a specificity test
    # (scripts/calibrate_gain.py), not chosen a priori:
    #   0.0050  pathway dead, nothing reaches wing motor
    #   0.0200  wing motor 13.8 Hz to pC1 vs 0.0-0.1 Hz to three
    #           size-matched control populations  <-- chosen
    #   0.0500  wing motor 41.3 Hz to pC1 but 28.3 Hz to random
    #           cb_intrinsic: the network is so excitable that the readout
    #           no longer distinguishes the courtship command from noise
    #   0.2750  fully saturated, population mean 52 Hz
    # The criterion is specificity, not realism: the experiment needs a
    # regime where removing neurons can actually change the output.
    synapse_mv: float = 0.02
    dt: float = 0.5
    stim_rate_hz: float = 150.0  # firing rate forced on stimulated neurons


def to_torch_transposed(adj: sp.csr_matrix, device: str) -> torch.Tensor:
    """Returns adj.T as a sparse CSR tensor.

    adj[i, j] is i -> j, so neuron j's incoming drive is
    sum_i spikes[i] * adj[i, j] = (adj.T @ spikes)[j]. The transpose is
    taken ONCE here rather than per timestep: transposing a sparse tensor
    inside the loop is both slow and, for CSR, not reliably supported.
    """
    coo = adj.T.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    t = torch.sparse_coo_tensor(idx, val, adj.shape, device=device).coalesce()
    return t.to_sparse_csr()


def simulate(
    adj: sp.csr_matrix,
    stim_idx: np.ndarray,
    duration_ms: float = 1000.0,
    params: LIFParams | None = None,
    silenced_idx: np.ndarray | None = None,
    device: str | None = None,
    record_every: int = 1,
    track_groups: dict[str, np.ndarray] | None = None,
    bin_ms: float = 5.0,
    stim_rate_series: np.ndarray | None = None,
) -> dict:
    """Run the network with `stim_idx` driven at params.stim_rate_hz.

    `silenced_idx` are ablated: they never fire, which is how a lesion is
    applied. Nothing is deleted -- the matrix is untouched and the mask is
    per-run, so the same adjacency serves intact and lesioned conditions.

    `stim_rate_series` replaces the constant rate with a per-timestep one,
    which is how a sound becomes an input: the envelope of what the fly
    hears sets its ear neurons' firing rate moment to moment. Its length
    defines the run, overriding `duration_ms`.

    Returns per-neuron spike counts and the population firing-rate trace.
    """
    p = params or LIFParams()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = adj.shape[0]
    steps = round(duration_ms / p.dt)

    rate_series = None
    if stim_rate_series is not None:
        rate_series = torch.from_numpy(
            np.asarray(stim_rate_series, dtype=np.float32)).to(device)
        steps = len(rate_series)
        duration_ms = steps * p.dt

    w_t = to_torch_transposed(adj, device)
    v = torch.full((n,), p.v_rest, dtype=torch.float32, device=device)
    refrac = torch.zeros(n, dtype=torch.float32, device=device)
    counts = torch.zeros(n, dtype=torch.int32, device=device)
    prev_spikes = torch.zeros(n, dtype=torch.float32, device=device)

    stim = torch.zeros(n, dtype=torch.bool, device=device)
    if len(stim_idx):
        stim[torch.from_numpy(np.asarray(stim_idx, dtype=np.int64)).to(device)] = True

    silent = torch.zeros(n, dtype=torch.bool, device=device)
    if silenced_idx is not None and len(silenced_idx):
        silent[torch.from_numpy(np.asarray(silenced_idx, dtype=np.int64)).to(device)] = True

    decay = p.dt / p.tau_m
    # Probability a stimulated neuron fires on any given step.
    p_stim = min(1.0, p.stim_rate_hz * p.dt / 1000.0)
    refrac_steps = p.refractory_ms

    rate_trace: list[float] = []
    gen = torch.Generator(device=device).manual_seed(0)

    # Per-group spike counts in time bins, so the cascade can be watched
    # propagating rather than only summarised at the end.
    tracks: dict[str, list[float]] = {}
    bin_steps = max(1, round(bin_ms / p.dt))
    if track_groups:
        g_idx = {k: torch.from_numpy(np.asarray(v, dtype=np.int64)).to(device)
                 for k, v in track_groups.items() if len(v)}
        g_acc = {k: 0 for k in g_idx}
        tracks = {k: [] for k in g_idx}

    for t in range(steps):
        # Leak toward rest.
        v += decay * (p.v_rest - v)

        # Drive from last step's spikes.
        v += p.synapse_mv * torch.mv(w_t, prev_spikes)

        v = torch.where(refrac > 0, torch.full_like(v, p.v_reset), v)
        refrac = torch.clamp(refrac - p.dt, min=0.0)

        spikes = (v >= p.v_threshold) & (refrac <= 0)
        # Stimulated neurons fire as a Poisson-like process regardless of input.
        # With a rate series, the probability follows the sound's envelope.
        step_p = p_stim if rate_series is None else float(rate_series[t]) * p.dt / 1000.0
        if step_p > 0:
            draw = torch.rand(n, generator=gen, device=device) < step_p
            spikes = spikes | (stim & draw)
        spikes = spikes & ~silent

        v = torch.where(spikes, torch.full_like(v, p.v_reset), v)
        refrac = torch.where(spikes, torch.full_like(refrac, refrac_steps), refrac)

        counts += spikes.int()
        prev_spikes = spikes.float()
        if record_every and t % record_every == 0:
            rate_trace.append(float(spikes.sum()) / n * 1000.0 / p.dt)

        if track_groups:
            for k, gidx in g_idx.items():
                g_acc[k] += int(spikes[gidx].sum())
            if (t + 1) % bin_steps == 0:
                window_s = bin_steps * p.dt / 1000.0
                for k, gidx in g_idx.items():
                    tracks[k].append(g_acc[k] / len(gidx) / window_s)
                    g_acc[k] = 0

    return {
        "spike_counts": counts.cpu().numpy(),
        "rates_hz": counts.cpu().numpy() / (duration_ms / 1000.0),
        "rate_trace_hz": np.array(rate_trace, dtype=np.float32),
        "group_traces_hz": {k: np.array(v, dtype=np.float32) for k, v in tracks.items()},
        "bin_ms": bin_ms,
        "duration_ms": duration_ms,
        "device": device,
        "params": p,
    }


def group_rate(rates: np.ndarray, idx: np.ndarray) -> float:
    """Mean firing rate (Hz) over a neuron group."""
    return float(rates[idx].mean()) if len(idx) else float("nan")


def calibrate_to_population_rate(
    adj: sp.csr_matrix,
    stim_idx: np.ndarray,
    target_hz: float = 0.6,
    lo: float = 1e-4,
    hi: float = 0.5,
    duration_ms: float = 100.0,
    iters: int = 14,
    device: str | None = None,
    silenced_idx: np.ndarray | None = None,
) -> tuple[float, float]:
    """Bisect `synapse_mv` until the whole-network mean firing rate hits
    `target_hz` under the given stimulus.

    Needed for any cross-dataset comparison. The gain multiplies raw
    synapse COUNTS, and two connectomes reconstructed by different groups
    with different synapse detection do not share a weight distribution --
    MaleCNS and FlyWire certainly do not. Running both at one fixed gain
    would make every male/female difference partly an artefact of that
    mismatch. Calibrating each network to the same operating point
    removes it: afterwards the networks are equally excitable overall, so
    a difference in a SPECIFIC pathway is about wiring, not drive.

    Returns (synapse_mv, achieved_population_hz). Monotone in gain, so
    bisection is safe.
    """
    best = (hi, float("nan"))
    for _ in range(iters):
        mid = (lo + hi) / 2
        # The silence mask must match the condition the gain will be USED
        # under. Calibrating a whole-CNS network and then running it with
        # the nerve cord silenced sets the gain for a network that is not
        # the one being measured.
        r = simulate(adj, stim_idx, duration_ms=duration_ms,
                     params=LIFParams(synapse_mv=mid), record_every=0, device=device,
                     silenced_idx=silenced_idx)
        rate = float(r["rates_hz"].mean())
        best = (mid, rate)
        if rate < target_hz:
            lo = mid
        else:
            hi = mid
    return best
