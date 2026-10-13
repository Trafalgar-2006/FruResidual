"""Calibrate the synapse gain, and prove the song readout is specific.

Two things have to hold before any result from this model means anything:

1. Stimulating the courtship command (pC1) must drive wing motor neurons.
2. Stimulating size-matched control populations must NOT. Otherwise the
   readout measures "the network got poked", and every downstream
   comparison is measuring excitability rather than courtship.

The gain is picked on criterion 2, which is why it is swept here rather
than assumed. Re-run this after any change to the connectome build.

Usage:  python scripts/calibrate_gain.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtship.circuit import resolve_groups, to_indices  # noqa: E402
from courtship.lif import LIFParams, group_rate, simulate  # noqa: E402

P = Path("C:/CourtshipData/processed")
GAINS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.275)
DURATION_MS = 200.0
# Minimum ratio of (wing rate to pC1) / (wing rate to best control).
MIN_SPECIFICITY = 5.0


def main() -> None:
    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    ann = load_male_cns_annotations(load_config(), traced_only=True)
    gi = to_indices(resolve_groups(ann), ids)
    sc = ann.set_index("bodyId").reindex(ids)["superclass"].to_numpy()

    rng = np.random.default_rng(0)
    n = len(gi["pC1"])
    controls = {
        f"random {c}": rng.choice(np.where(sc == c)[0], n, replace=False)
        for c in ("cb_intrinsic", "ol_intrinsic", "vnc_sensory")
    }

    # No input must mean no activity: this model has no spontaneous term,
    # so anything above zero here is a bug, not biology.
    quiet = simulate(adj, np.array([], dtype=int), duration_ms=DURATION_MS, record_every=0)
    n_quiet = int(quiet["spike_counts"].sum())
    print(f"unstimulated network total spikes: {n_quiet} (must be 0)")
    if n_quiet:
        sys.exit("FAIL: network fires with no input")

    rows = []
    print(f"\n{'gain':>8}{'pop Hz':>9}{'pIP10':>9}{'TN1':>9}{'WING(pC1)':>11}"
          f"{'WING(ctrl max)':>16}{'specificity':>13}")
    for g in GAINS:
        p = LIFParams(synapse_mv=g)
        r = simulate(adj, gi["pC1"], duration_ms=DURATION_MS, params=p, record_every=0)
        rt = r["rates_hz"]
        wing_sig = group_rate(rt, gi["wing_motor"])
        ctrl_wing = []
        for idx in controls.values():
            rc = simulate(adj, idx, duration_ms=DURATION_MS, params=p, record_every=0)
            ctrl_wing.append(group_rate(rc["rates_hz"], gi["wing_motor"]))
        worst = max(ctrl_wing)
        spec = wing_sig / worst if worst > 0 else float("inf")
        rows.append({"gain": g, "pop_hz": float(rt.mean()),
                     "wing_pc1_hz": wing_sig, "wing_control_max_hz": worst,
                     "specificity": None if spec == float("inf") else spec})
        shown = "inf" if spec == float("inf") else f"{spec:.1f}"
        print(f"{g:>8.4f}{rt.mean():>9.2f}{group_rate(rt, gi['pIP10']):>9.1f}"
              f"{group_rate(rt, gi['TN1']):>9.1f}{wing_sig:>11.1f}{worst:>16.1f}{shown:>13}")

    chosen = LIFParams().synapse_mv
    row = next(r for r in rows if abs(r["gain"] - chosen) < 1e-9)
    spec = row["specificity"]
    print(f"\nconfigured default synapse_mv = {chosen}")
    print(f"  wing to pC1: {row['wing_pc1_hz']:.1f} Hz, "
          f"best control: {row['wing_control_max_hz']:.1f} Hz")
    ok = row["wing_pc1_hz"] > 1.0 and (spec is None or spec >= MIN_SPECIFICITY)
    print("  VERDICT:", "specific enough to run experiments on" if ok else "NOT SPECIFIC -- stop")

    (P / "calibration.json").write_text(json.dumps(
        {"chosen_synapse_mv": chosen, "duration_ms": DURATION_MS, "sweep": rows}, indent=2))
    print(f"\nwrote {P / 'calibration.json'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
