"""Play an audio file into the fly's ears and record what it does.

Pipeline, both ends of which are real anatomy:

    audio -> band-pass to what Johnston's organ can hear -> envelope
          -> JO neuron firing rate  (the antenna transducing sound)
          -> the connectome
          -> readout spikes -> pulses -> waveform

The fly does NOT reproduce the input. Its wing is not a speaker and the
circuit is not a transmission line: what comes out is the fly's own
response, following the input's rhythm and loudness, not its pitch.

Measured first, before choosing readouts (300 ms, JO driven at 150 Hz):

    JO           147.4 Hz    driven
    cb_sensory     4.18 Hz   sound propagates into the brain
    descending     0.68 Hz   reaches the brain's output channel
    pC1/pIP10      0.00 Hz   song circuit untouched
    wing motor     0.00 Hz   silent

So sound alone does not make a male sing, which is correct biology:
courtship song is released by pheromone and context through P1, not by
hearing. Hence two conditions.

  HEARING    ears driven by the audio, nothing else. Readout is the
             DESCENDING population -- the brain's output channel toward
             the body, present in both sexes and the only honest shared
             readout, since FlyWire is brain-only and has no motor
             neurons at all.

  COURTING   (male only) the same audio, with pC1 also held at its
             command rate: a male already in courtship who is hearing
             something. Readout is wing motor -- actual song, modulated
             by what he hears.

Usage:
    python scripts/run_audio.py clip.mp3 [--seconds 4] [--sex both]
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
    load_audio,
    spikes_to_song,
    spikes_to_waveform,
    write_wav,
)
from courtship.circuit import normalize_annotations, resolve_groups, to_indices
from courtship.lif import LIFParams, simulate

P = Path("C:/CourtshipData/processed")
OUT = Path(__file__).resolve().parents[1] / "reports" / "audio"
FLYWIRE_ANN = Path("C:/FruResidualData/raw/flywire_annotations.feather")
SR = 22050


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
    return adj, ids, to_indices(resolve_groups(normalize_annotations(ann)), ids)


def envelope(x: np.ndarray, n: int = 240) -> list[float]:
    return [round(float(v), 3) for v in
            np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="how much of the clip to use (simulation runs ~20x slower than realtime)")
    ap.add_argument("--sex", choices=["male", "female", "both"], default="both")
    ap.add_argument("--max-rate", type=float, default=600.0,
                    help="peak JO firing rate; 600 measurably carries pitch, 180 does not")
    ap.add_argument("--dt", type=float, default=0.1,
                    help="ms per step. 0.1 gives a 5 kHz ceiling; 0.5 caps output at 1 kHz")
    ap.add_argument("--mode", choices=["waveform", "envelope"], default="waveform",
                    help="waveform rate-codes the signal itself (carries pitch); "
                         "envelope only its loudness")
    args = ap.parse_args()

    params = LIFParams(dt=args.dt)
    samples, sr = load_audio(args.audio, target_sr=SR)
    samples = samples[:int(min(len(samples), args.seconds * sr))]
    print(f"input: {Path(args.audio).name}  {len(samples)/sr:.2f}s @ {sr} Hz")

    if args.mode == "waveform":
        # Rate-code the band-passed signal itself. Measured 2026-10-14: the
        # wing motor spike density then peaks at the input frequency (8/8
        # notes of a tone sequence tracked to within 1%). Rate-coding only
        # the envelope throws that away before the circuit ever sees it.
        drive = audio_to_firing_rate(samples, sr, params.dt,
                                     max_rate_hz=args.max_rate, waveform=True)
    else:
        drive = audio_to_firing_rate(samples, sr, params.dt, max_rate_hz=args.max_rate)
    print(f"JO drive: {len(drive):,} steps, {drive.min():.0f}-{drive.max():.0f} Hz "
          f"(mean {drive.mean():.0f})")

    OUT.mkdir(parents=True, exist_ok=True)
    write_wav(OUT / "input_as_heard.wav", samples, sr)
    window_s = params.dt / 1000.0
    summary = {"input": Path(args.audio).name, "seconds": len(samples) / sr,
               "max_jo_rate_hz": args.max_rate, "dt_ms": params.dt,
               "input_envelope": envelope(drive), "results": []}

    def run(adj, stim, gi, readout, label, which, also_stim=None):
        idx = gi[readout]
        stim_all = stim if also_stim is None else np.unique(np.concatenate([stim, also_stim]))
        # pC1 is held at its own command rate, so it cannot ride the audio
        # envelope; only the ears do. Co-stimulating both from one series
        # would make the "courting" male's song a direct echo of the input.
        r = simulate(adj, stim_all, params=params, stim_rate_series=drive,
                     record_every=0, track_groups={readout: idx}, bin_ms=params.dt)
        counts = np.array(r["group_traces_hz"][readout]) * len(idx) * window_s
        # Two renderings, because they answer different questions.
        # _pulses: what a wing physically radiates -- one fixed-carrier pulse
        #   per spike. True to the animal, but the carrier overwrites pitch.
        # _density: the motor signal itself, which does carry the input's
        #   pitch. This is the one that reproduces a tune.
        name = f"{which}_{label}_density.wav"
        write_wav(OUT / name, spikes_to_waveform(counts, params.dt, sr=SR), SR)
        write_wav(OUT / f"{which}_{label}_pulses.wav",
                  spikes_to_song(counts, params.dt, sr=SR, n_neurons=len(idx)), SR)
        mean_hz = counts.sum() / len(idx) / (len(counts) * window_s)
        print(f"  {label:<10} readout {readout:<12} "
              f"{(counts > 0).mean()*100:>5.1f}% of steps active, mean {mean_hz:6.2f} Hz "
              f"-> {name}")
        summary["results"].append({
            "sex": which, "condition": label, "readout": readout,
            "n_readout": len(idx), "mean_hz": float(mean_hz),
            "pct_steps_active": float((counts > 0).mean() * 100),
            "wav": name, "envelope": envelope(counts),
        })

    for which in (["male", "female"] if args.sex == "both" else [args.sex]):
        side = load_side(which)
        if side is None:
            print(f"\n{which}: FlyWire connectome not built yet -- skipped")
            continue
        adj, _ids, gi = side
        if "JO" not in gi:
            print(f"\n{which}: no Johnston's organ neurons found -- skipped")
            continue
        print(f"\n{which}: {len(gi['JO']):,} ear neurons")

        run(adj, gi["JO"], gi, "descending", "hearing", which)
        if which == "male" and "wing_motor" in gi and "pC1" in gi:
            run(adj, gi["JO"], gi, "wing_motor", "courting", which, also_stim=gi["pC1"])
            run(adj, gi["JO"], gi, "wing_motor", "hearing_only", which)

    (OUT / "audio_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'audio_results.json'}")


if __name__ == "__main__":
    main()
