"""Precompute the site's input library in one batched pass.

Every (stimulus x condition) pair is an independent simulation over the
same matrix, so they all run as columns of a single batch instead of one
at a time. This is what batching was for.

The result is a set of WAV files and a JSON index. Serving them needs no
GPU and no server -- the site is static.

Stimuli are synthesised from physical parameters, not sampled from
recordings, so the library carries no third-party audio.

Usage:  python scripts/make_library.py [--seconds 2] [--trials 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtship.audio import (
    audio_to_firing_rate,
    spikes_to_song,
    spikes_to_waveform,
    write_wav,
)
from courtship.circuit import normalize_annotations, resolve_groups, to_indices
from courtship.lif import LIFParams, simulate

P = Path("C:/CourtshipData/processed")
OUT = Path(__file__).resolve().parents[1] / "reports" / "library"
FLYWIRE_ANN = Path("C:/FruResidualData/raw/flywire_annotations.feather")
SR = 22050


def courtship_song(seconds: float, sr: int = SR) -> np.ndarray:
    """A male Drosophila courtship song, synthesised from its published
    physical parameters rather than sampled from a recording.

    Two modes, which is what the real animal produces: PULSE song, a train
    of brief ~250 Hz damped oscillations about 35 ms apart, and SINE song,
    a continuous hum near 150 Hz. Males alternate between them in bouts.
    This is the stimulus the circuit evolved to process.
    """
    n = int(seconds * sr)
    out = np.zeros(n, dtype=np.float32)
    t_pulse = np.arange(int(0.004 * sr)) / sr
    pulse = (np.sin(2 * np.pi * 250 * t_pulse) *
             np.exp(-t_pulse / 0.0013)).astype(np.float32)

    pos, bout = 0, 0
    while pos < n:
        if bout % 2 == 0:                       # pulse bout
            end = min(n, pos + int(0.35 * sr))
            k = pos
            while k < end - len(pulse):
                out[k:k + len(pulse)] += pulse
                k += int(0.035 * sr)            # inter-pulse interval
            pos = end
        else:                                   # sine bout
            end = min(n, pos + int(0.25 * sr))
            tt = np.arange(end - pos) / sr
            out[pos:end] += 0.6 * np.sin(2 * np.pi * 150 * tt).astype(np.float32)
            pos = end
        bout += 1
    peak = np.abs(out).max()
    return out / peak if peak else out


def stimuli(seconds: float, sr: int = SR) -> dict[str, np.ndarray]:
    """Every stimulus is returned at EXACTLY the same length.

    They are batched as columns of one matrix, which requires identical
    step counts, and the natural lengths differ -- the tone sequence rounds
    per note, the song is built in whole bouts.
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr

    def fit(x):
        """Trim or zero-pad to exactly n samples, then normalise."""
        x = np.asarray(x, dtype=np.float32)
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        x = x[:n]
        p = np.abs(x).max()
        return (x / p).astype(np.float32) if p else x

    norm = fit

    notes = [250, 300, 375, 500, 375, 300, 250, 300]
    seq, per = [], int(seconds * sr / len(notes))
    for f in notes:
        tt = np.arange(per) / sr
        env = np.minimum(1.0, np.minimum(tt, per / sr - tt) * 40)
        seq.append(np.sin(2 * np.pi * f * tt) * env)

    return {
        "courtship_song": courtship_song(seconds, sr),
        "tone_sequence": norm(np.concatenate(seq)),
        "steady_tone": norm(np.sin(2 * np.pi * 300 * t)),
        "rising_sweep": norm(np.sin(2 * np.pi * (120 + 500 * t / t[-1]) * t)),
        "beat": norm(np.sin(2 * np.pi * 250 * t) * (np.sin(2 * np.pi * 5 * t) > 0)),
    }


def load_side(which: str):
    if which == "male":
        adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
        ids = np.load(P / "malecns_node_ids.npy")
        ann = load_male_cns_annotations(load_config(), traced_only=True)
    else:
        f = P / "flywire_adj_minw1.npz"
        if not f.exists():
            return None
        adj = sp.load_npz(f).tocsr()
        ids = np.load(P / "flywire_node_ids.npy")
        ann = pd.read_feather(FLYWIRE_ANN)
    return adj, to_indices(resolve_groups(normalize_annotations(ann)), ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.2)
    args = ap.parse_args()

    gains = {"male": LIFParams().synapse_mv, "female": LIFParams().synapse_mv}
    if (P / "calibration.json").exists():
        gains["male"] = json.loads((P / "calibration.json").read_text())["chosen_synapse_mv"]
    if (P / "sex_comparison.json").exists():
        c = json.loads((P / "sex_comparison.json").read_text())
        if c.get("female_gain"):
            gains["female"] = c["female_gain"]

    OUT.mkdir(parents=True, exist_ok=True)
    stims = stimuli(args.seconds)
    index = {"seconds": args.seconds, "trials": args.trials, "dt_ms": args.dt,
             "gains": gains, "sample_rate": SR, "stimuli": {}, "entries": []}

    for name, wave in stims.items():
        write_wav(OUT / f"input_{name}.wav", wave, SR)
        index["stimuli"][name] = {"wav": f"input_{name}.wav"}

    window_s = args.dt / 1000.0

    for which in ("male", "female"):
        side = load_side(which)
        if side is None:
            print(f"{which}: connectome not built, skipped")
            continue
        adj, gi = side
        if "JO" not in gi:
            continue
        conditions = [("hearing", "descending", False)]
        if which == "male" and "wing_motor" in gi:
            conditions += [("courting", "wing_motor", True),
                           ("hearing_only", "wing_motor", False)]

        for label, readout, with_pc1 in conditions:
            idx = gi[readout]
            stim = gi["JO"] if not with_pc1 else np.unique(
                np.concatenate([gi["JO"], gi["pC1"]]))
            # One batch per (condition): columns are stimulus x trial.
            cols, meta = [], []
            for sname, wave in stims.items():
                drive = audio_to_firing_rate(wave, SR, args.dt, max_rate_hz=600.0,
                                             waveform=True)
                for _ in range(args.trials):
                    cols.append(drive)
                    meta.append(sname)
            series = np.stack(cols, axis=1)
            print(f"{which}/{label}: {series.shape[1]} columns "
                  f"({len(stims)} stimuli x {args.trials} trials), "
                  f"{series.shape[0]:,} steps")
            r = simulate(adj, stim, params=LIFParams(dt=args.dt, synapse_mv=gains[which]),
                         stim_rate_series=series, record_every=0,
                         track_groups={readout: idx}, bin_ms=args.dt,
                         n_batch=series.shape[1])
            tr = r["group_traces_hz"][readout]            # (columns, bins)

            for sname in stims:
                sel = [i for i, m in enumerate(meta) if m == sname]
                counts = tr[sel].mean(axis=0) * len(idx) * window_s
                half = len(sel) // 2
                a = tr[sel[0::2]].mean(axis=0)
                b = tr[sel[1::2]].mean(axis=0)
                rel = float(np.corrcoef(a, b)[0, 1]) if half and a.std() and b.std() else 0.0
                stem = f"{which}_{label}_{sname}"
                write_wav(OUT / f"{stem}_density.wav",
                          spikes_to_waveform(counts, args.dt, sr=SR), SR)
                write_wav(OUT / f"{stem}_pulses.wav",
                          spikes_to_song(counts, args.dt, sr=SR, n_neurons=len(idx)), SR)
                mean_hz = counts.sum() / len(idx) / (len(counts) * window_s)
                index["entries"].append({
                    "sex": which, "condition": label, "stimulus": sname,
                    "readout": readout, "n_readout": len(idx),
                    "mean_hz": float(mean_hz), "reliability": rel,
                    "density_wav": f"{stem}_density.wav",
                    "pulses_wav": f"{stem}_pulses.wav",
                    "envelope": [round(float(x), 3) for x in np.interp(
                        np.linspace(0, len(counts) - 1, 200),
                        np.arange(len(counts)), counts)],
                })
                print(f"   {sname:<16} {mean_hz:6.2f} Hz  reliability {rel:.2f}")

    (OUT / "library.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index['entries'])} entries -> {OUT / 'library.json'}")


if __name__ == "__main__":
    main()
