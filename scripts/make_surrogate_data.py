"""Generate training data for the surrogate: (ear drive -> readout response).

What the surrogate approximates is a single-input, single-output, causal,
time-invariant nonlinear filter. The connectome never changes between
runs, every JO neuron receives the same drive, and the readout is one
population density -- so the whole system is one fixed filter, which is
about the most learnable object available.

Two consequences worth stating, because they shape this script:

TARGETS DO NOT NEED TO BE CLEAN. The simulation is stochastic, but MSE
regression converges on the conditional mean, so a network trained on
single-trial targets learns the TRIAL-AVERAGED response by itself. That
removes the 8x cost of averaging during data generation, and it means the
surrogate outputs the denoised signal in one pass at inference.

DIVERSITY MATTERS MORE THAN LENGTH. The filter has roughly 100 ms of
memory, so what it needs is many short, varied excerpts rather than a few
long ones. Stimuli are synthesised across tones, chords, sweeps,
amplitude modulation, pulse trains, noise bursts and silence, so the
training distribution covers the shapes real audio decomposes into.

Usage:  python scripts/make_surrogate_data.py [--clips 600] [--batch 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtship.audio import audio_to_firing_rate
from courtship.circuit import normalize_annotations, resolve_groups, to_indices
from courtship.lif import LIFParams, simulate

P = Path("C:/CourtshipData/processed")
OUT = Path("C:/CourtshipData/surrogate")
FLYWIRE_ANN = Path("C:/FruResidualData/raw/flywire_annotations.feather")
SR = 22050


def random_clip(rng: np.random.Generator, seconds: float, sr: int = SR) -> np.ndarray:
    """One random stimulus. The kinds are chosen to span the shapes audio
    decomposes into, not to sound like anything in particular."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    kind = rng.integers(0, 7)

    if kind == 0:                                    # pure tone
        x = np.sin(2 * np.pi * rng.uniform(80, 1000) * t)
    elif kind == 1:                                  # two-tone chord
        x = (np.sin(2 * np.pi * rng.uniform(80, 600) * t) +
             np.sin(2 * np.pi * rng.uniform(200, 1000) * t))
    elif kind == 2:                                  # sweep, either direction
        f0, f1 = rng.uniform(80, 900), rng.uniform(80, 900)
        x = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / max(t[-1], 1e-9)) * t)
    elif kind == 3:                                  # amplitude modulation
        x = (np.sin(2 * np.pi * rng.uniform(100, 800) * t) *
             (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 40) * t)))
    elif kind == 4:                                  # pulse train (song-like)
        x = np.zeros(n)
        tp = np.arange(int(0.004 * sr)) / sr
        pulse = np.sin(2 * np.pi * rng.uniform(180, 320) * tp) * np.exp(-tp / 0.0013)
        step = int(rng.uniform(0.015, 0.08) * sr)
        for k in range(0, n - len(pulse), max(step, 1)):
            x[k:k + len(pulse)] += pulse
    elif kind == 5:                                  # band-limited noise burst
        x = rng.normal(size=n)
        gate = np.zeros(n)
        for _ in range(rng.integers(1, 4)):
            a = rng.integers(0, max(n - 1, 1))
            b = min(n, a + int(rng.uniform(0.05, 0.4) * sr))
            gate[a:b] = 1.0
        x = x * gate
    else:                                            # tone with silent gaps
        x = np.sin(2 * np.pi * rng.uniform(100, 700) * t)
        for _ in range(rng.integers(1, 4)):
            a = rng.integers(0, max(n - 1, 1))
            b = min(n, a + int(rng.uniform(0.05, 0.3) * sr))
            x[a:b] = 0.0

    x = np.asarray(x, dtype=np.float32)
    peak = np.abs(x).max()
    if peak:
        x = x / peak
    # Random overall level, so the surrogate sees quiet and loud passages.
    return (x * rng.uniform(0.3, 1.0)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=600)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--condition", choices=["courting", "hearing"], default="courting")
    ap.add_argument("--sex", choices=["male", "female"], default="male")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.sex == "male":
        adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
        ids = np.load(P / "malecns_node_ids.npy")
        ann = load_male_cns_annotations(load_config(), traced_only=True)
        gain = LIFParams().synapse_mv
    else:
        adj = sp.load_npz(P / "flywire_adj_minw1.npz").tocsr()
        ids = np.load(P / "flywire_node_ids.npy")
        ann = pd.read_feather(FLYWIRE_ANN)
        import json
        gain = json.loads((P / "sex_comparison.json").read_text())["female_gain"]
    gi = to_indices(resolve_groups(normalize_annotations(ann)), ids)

    if args.condition == "courting" and "wing_motor" in gi:
        readout, stim = "wing_motor", np.unique(np.concatenate([gi["JO"], gi["pC1"]]))
    else:
        readout, stim = "descending", gi["JO"]
    idx = gi[readout]
    print(f"{args.sex}/{args.condition}: ears {len(gi['JO'])} -> {readout} {len(idx)}, "
          f"gain {gain:.5f}")

    rng = np.random.default_rng(args.seed)
    params = LIFParams(dt=args.dt, synapse_mv=gain)
    window_s = args.dt / 1000.0
    OUT.mkdir(parents=True, exist_ok=True)

    drives, responses = [], []
    done, t0 = 0, time.time()
    while done < args.clips:
        b = min(args.batch, args.clips - done)
        clips = [random_clip(rng, args.seconds) for _ in range(b)]
        cols = [audio_to_firing_rate(c, SR, args.dt, max_rate_hz=600.0, waveform=True)
                for c in clips]
        series = np.stack(cols, axis=1)
        r = simulate(adj, stim, params=params, stim_rate_series=series,
                     record_every=0, track_groups={readout: idx},
                     bin_ms=args.dt, n_batch=b)
        tr = r["group_traces_hz"][readout]                 # (b, steps)
        drives.append(series.T.astype(np.float32))          # (b, steps)
        responses.append((tr * len(idx) * window_s).astype(np.float32))
        done += b
        el = time.time() - t0
        audio_s = done * args.seconds
        print(f"  {done}/{args.clips} clips  ({audio_s:.0f}s audio, {el:.0f}s elapsed, "
              f"{audio_s / max(el, 1e-9):.2f}x realtime)")

    X = np.concatenate(drives, axis=0)
    Y = np.concatenate(responses, axis=0)
    out = OUT / f"{args.sex}_{args.condition}_dt{args.dt}.npz"
    np.savez_compressed(out, drive=X, response=Y, dt_ms=args.dt,
                        n_readout=len(idx), seconds=args.seconds, gain=gain)
    print(f"\n{X.shape[0]} clips x {X.shape[1]} steps -> {out} "
          f"({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
