"""Does the surrogate actually imitate the simulation, or just look like it?

Correlation alone is not enough to clear this. Within a condition the mean
firing rate barely moves between stimuli -- 24-27 Hz across five very
different inputs -- so the amplitude envelope carries almost no
information and a network that learned only the envelope would score a
respectable correlation while being useless. The structure lives in the
TIMING. So the headline test is pitch tracking: play tones the surrogate
never saw and check the frequency its output peaks at.

Three tests, in increasing severity:

  IN-DISTRIBUTION   held-out clips of the kinds it trained on.
  PITCH             pure tones, note by note, against the real simulation.
  OUT-OF-DISTRIBUTION  stimuli drawn from NO training category -- the real
                    question, since a website accepts arbitrary uploads.
                    A surrogate can be excellent in-distribution and
                    confidently wrong outside it, and nothing about the
                    training curve reveals that.

Usage:  python scripts/validate_surrogate.py
"""

from __future__ import annotations

import os

# conda-forge numpy and pip torch ship separate OpenMP runtimes on Windows;
# without this the process aborts the moment both are loaded. Must be set
# BEFORE torch is imported, which is why it sits above the import block.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_surrogate import Surrogate

from courtship.audio import audio_to_firing_rate
from courtship.circuit import resolve_groups, to_indices
from courtship.lif import LIFParams, simulate

P = Path("C:/CourtshipData/processed")
DATA = Path("C:/CourtshipData/surrogate")
MODELS = Path(__file__).resolve().parents[1] / "models"
SR = 22050


TRIALS = 8


def trial_mean(adj, stim, params, idx, series, dt, trials: int = TRIALS):
    """Average `trials` independent runs to estimate the EXPECTED response.

    The surrogate is trained with MSE, so it predicts the conditional mean.
    Comparing it against a single stochastic run measures the model against
    noise it was never asked to reproduce -- the library put single-trial
    split-half reliability for this condition at 0.32-0.75, so one trial
    barely correlates with itself. Averaging gives the quantity the
    surrogate is actually approximating.
    """
    window_s = dt / 1000.0
    acc = None
    for s_ in range(trials):
        r = simulate(adj, stim, params=params, stim_rate_series=series,
                     record_every=0, track_groups={"w": idx}, bin_ms=dt,
                     n_batch=series.shape[1], seed=s_)
        c = r["group_traces_hz"]["w"] * len(idx) * window_s
        acc = c if acc is None else acc + c
    return acc / trials


def peak_hz(sig: np.ndarray, step_sr: float, lo=80.0, hi=1200.0) -> float:
    x = np.asarray(sig, float)
    x = x - x.mean()
    if x.std() == 0:
        return float("nan")
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), 1 / step_sr)
    band = (fr >= lo) & (fr <= hi)
    return float(fr[band][np.argmax(spec[band])])


def ood_stimuli(seconds: float, sr: int = SR) -> dict[str, np.ndarray]:
    """Shapes deliberately absent from the training categories."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1234)

    def norm(x):
        x = np.asarray(x, np.float32)
        p = np.abs(x).max()
        return (x / p).astype(np.float32) if p else x

    # square and sawtooth: harmonically rich, never trained on
    square = np.sign(np.sin(2 * np.pi * 220 * t))
    saw = 2 * ((220 * t) % 1.0) - 1
    # vibrato: frequency modulation, distinct from the amplitude modulation seen
    vib = np.sin(2 * np.pi * 300 * t + 3.0 * np.sin(2 * np.pi * 6 * t))
    # a dense inharmonic cluster
    cluster = sum(np.sin(2 * np.pi * f * t) for f in (137, 263, 491, 757))
    # slow click train far outside the trained inter-pulse range
    clicks = np.zeros(n)
    clicks[:: int(0.25 * sr)] = 1.0
    return {"square_wave": norm(square), "sawtooth": norm(saw),
            "vibrato_fm": norm(vib), "inharmonic_cluster": norm(cluster),
            "slow_clicks": norm(clicks),
            "pink_noise": norm(np.cumsum(rng.normal(size=n)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "male_courting_dt0.2.npz"))
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()

    stem = Path(args.data).stem
    ck = torch.load(MODELS / f"{stem}.pt", map_location="cpu")
    model = Surrogate(ck["channels"], ck["blocks"]).eval()
    model.load_state_dict(ck["state"])
    dt, xs, ys = ck["dt_ms"], ck["x_scale"], ck["y_scale"]
    step_sr = 1000.0 / dt
    print(f"surrogate: {sum(p.numel() for p in model.parameters()):,} parameters, dt={dt} ms")

    def predict(drive: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy((drive / xs).astype(np.float32))[None, None, :]
            return model(x).numpy()[0, 0] * ys

    # --- 1. held-out, in-distribution ---
    d = np.load(args.data)
    X, Y = d["drive"], d["response"]
    rng = np.random.default_rng(0)
    val = rng.permutation(len(X))[: max(1, int(len(X) * 0.15))]
    cors = []
    for i in val:
        p = predict(X[i])
        if p.std() > 0 and Y[i].std() > 0:
            cors.append(np.corrcoef(p, Y[i])[0, 1])
    print(f"\nIN-DISTRIBUTION  held-out clips: n={len(cors)}, "
          f"r = {np.mean(cors):.3f} +/- {np.std(cors):.3f}")

    # --- 2. pitch tracking against the real simulation ---
    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    gi = to_indices(resolve_groups(load_male_cns_annotations(load_config(),
                                                             traced_only=True)), ids)
    stim = np.unique(np.concatenate([gi["JO"], gi["pC1"]]))
    idx = gi["wing_motor"]
    params = LIFParams(dt=dt, synapse_mv=float(d["gain"]))
    t = np.arange(int(args.seconds * SR)) / SR

    print(f"\nPITCH  tones, surrogate vs the real simulation ({args.seconds}s each)")
    print(f"{'played':>8}{'simulated':>12}{'surrogate':>12}{'agree':>8}")
    tones = [180, 250, 330, 440, 600, 800]
    drives = [audio_to_firing_rate(np.sin(2 * np.pi * f * t).astype(np.float32),
                                   SR, dt, max_rate_hz=600.0, waveform=True)
              for f in tones]
    sim_counts = trial_mean(adj, stim, params, idx, np.stack(drives, axis=1), dt)
    hits = 0
    pitch_rows = []
    for k, f in enumerate(tones):
        sim_pk = peak_hz(sim_counts[k], step_sr)
        sur_pk = peak_hz(predict(drives[k]), step_sr)
        ok = abs(sur_pk - sim_pk) < 0.1 * max(sim_pk, 1)
        hits += ok
        pitch_rows.append({"played": f, "simulated_hz": sim_pk, "surrogate_hz": sur_pk,
                           "agree": bool(ok)})
        print(f"{f:>8}{sim_pk:>12.0f}{sur_pk:>12.0f}{'  yes' if ok else '   NO':>8}")
    print(f"  -> {hits}/{len(tones)} tones agree with the simulation")

    # --- 3. out of distribution ---
    print("\nOUT-OF-DISTRIBUTION  shapes absent from every training category")
    oods = ood_stimuli(args.seconds)
    od = [audio_to_firing_rate(w, SR, dt, max_rate_hz=600.0, waveform=True)
          for w in oods.values()]
    sim2 = trial_mean(adj, stim, params, idx, np.stack(od, axis=1), dt)
    print(f"{'stimulus':<22}{'r':>8}{'sim peak':>11}{'surr peak':>11}")
    ood_rows, ood_cors = [], []
    for k, name in enumerate(oods):
        pred = predict(od[k])
        c = float(np.corrcoef(pred, sim2[k])[0, 1]) if pred.std() and sim2[k].std() else 0.0
        sp_, su_ = peak_hz(sim2[k], step_sr), peak_hz(pred, step_sr)
        ood_cors.append(c)
        ood_rows.append({"stimulus": name, "r": c, "sim_hz": sp_, "surrogate_hz": su_})
        print(f"{name:<22}{c:>8.3f}{sp_:>11.0f}{su_:>11.0f}")
    # Deliberately NOT compared against the in-distribution number above.
    # That one is measured against single noisy trials from the stored
    # dataset; these are measured against an 8-trial average. Different
    # ground truths, so the two correlations are not on the same scale and
    # putting them side by side invites exactly the wrong conclusion.
    print(f"  -> mean r = {np.mean(ood_cors):.3f} "
          f"(vs an 8-trial average; the in-distribution figure above uses "
          f"single trials and is not comparable)")

    verdict = ("usable for arbitrary uploads"
               if np.mean(ood_cors) > 0.6 and hits >= len(tones) - 1
               else "NOT usable for arbitrary uploads -- keep the site on the library")
    print(f"\nVERDICT: {verdict}")

    (MODELS / f"{stem}_validation.json").write_text(json.dumps({
        "in_distribution_r_mean": float(np.mean(cors)),
        "in_distribution_r_std": float(np.std(cors)),
        "pitch": pitch_rows, "pitch_agree": f"{hits}/{len(tones)}",
        "out_of_distribution": ood_rows,
        "ood_r_mean": float(np.mean(ood_cors)),
        "verdict": verdict}, indent=2))
    print(f"wrote {MODELS / f'{stem}_validation.json'}")


if __name__ == "__main__":
    main()
