"""Male vs female: does the courtship command reach the descending output?

The honest version of "one brain sings and the other doesn't".

FlyWire (female) is brain-only, so pIP10, TN1, dPR1 and the wing motor
neurons -- everything downstream of the brain in the song pathway -- do
not exist there. Comparing "song" directly is impossible. The one stage
both datasets contain is the DESCENDING NEURON population, the brain's
output channel, and it is almost perfectly size-matched already:
1,314 male against 1,303 female.

Three confounds are controlled, because any of them alone could produce a
"sex difference" that is really a dataset difference:

1. SCOPE. MaleCNS is whole-CNS, FlyWire is brain-only. The male network
   is restricted to its brain for the matched comparison.
2. EXCITABILITY. The gain multiplies raw synapse counts, and the two
   datasets were reconstructed by different groups with different synapse
   detection. Each network is calibrated to the same population firing
   rate first, so neither is simply louder than the other.
3. STIMULUS SIZE. Male pC1 is 156 neurons, female pC1 is 10. Driving 156
   against 10 is not a comparison. The male side is subsampled to 10,
   repeated over seeds, and the full-pC1 result reported separately as
   the biological (uncontrolled) number.

Specificity is checked on BOTH sides: a size-matched random central
population must not drive the descending output the way pC1 does.

Usage:  python scripts/run_sex_comparison.py [--n-sub 20]
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

from courtship.circuit import normalize_annotations, resolve_groups, to_indices
from courtship.lif import LIFParams, calibrate_to_population_rate, group_rate, simulate

P = Path("C:/CourtshipData/processed")
FLYWIRE_ANN = Path("C:/FruResidualData/raw/flywire_annotations.feather")
DURATION_MS = 200.0
TARGET_POP_HZ = 0.6          # common operating point for both networks
MALE_VNC_CLASSES = ("vnc_intrinsic", "vnc_motor", "vnc_sensory",
                    "vnc_efferent", "vnc_endocrine", "ascending_neuron")


def load_side(which: str):
    if which == "male":
        adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
        ids = np.load(P / "malecns_node_ids.npy")
        ann = load_male_cns_annotations(load_config(), traced_only=True)
    else:
        adj = sp.load_npz(P / "flywire_adj_minw1.npz").tocsr()
        ids = np.load(P / "flywire_node_ids.npy")
        ann = pd.read_feather(FLYWIRE_ANN)
    ann = normalize_annotations(ann)
    return adj, ids, ann, to_indices(resolve_groups(ann), ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sub", type=int, default=20,
                    help="male pC1 subsamples for the stimulus-matched test")
    args = ap.parse_args()

    out: dict = {"target_pop_hz": TARGET_POP_HZ, "duration_ms": DURATION_MS}
    rng = np.random.default_rng(0)
    sides = {}

    for which in ("male", "female"):
        adj, ids, ann, gi = load_side(which)
        sc = ann.set_index("bodyId").reindex(ids)["superclass"].to_numpy()
        silence = None
        if which == "male":
            # Restrict to the brain so the scope matches FlyWire.
            silence = np.where(np.isin(sc, MALE_VNC_CLASSES))[0]
            print(f"male: silencing {len(silence):,} nerve-cord neurons for scope match")
        sides[which] = {"adj": adj, "ids": ids, "ann": ann, "gi": gi, "sc": sc, "silence": silence}
        print(f"{which}: {adj.shape[0]:,} neurons, {adj.nnz:,} edges, "
              f"pC1={len(gi['pC1'])}, descending={len(gi['descending'])}")

    # --- 2. calibrate each network to the same population rate ---
    for which, s in sides.items():
        gain, achieved = calibrate_to_population_rate(
            s["adj"], s["gi"]["pC1"], target_hz=TARGET_POP_HZ)
        s["gain"] = gain
        print(f"{which}: calibrated synapse_mv = {gain:.5f} "
              f"(population {achieved:.3f} Hz, target {TARGET_POP_HZ})")
        out[f"{which}_gain"] = gain
        out[f"{which}_calibrated_pop_hz"] = achieved

    def dn_rate(s, stim, silenced=None):
        sil = s["silence"]
        if silenced is not None and len(silenced):
            sil = silenced if sil is None else np.unique(np.concatenate([sil, silenced]))
        r = simulate(s["adj"], stim, duration_ms=DURATION_MS,
                     params=LIFParams(synapse_mv=s["gain"]),
                     silenced_idx=sil, record_every=0)
        return group_rate(r["rates_hz"], s["gi"]["descending"])

    rows = []

    # --- biological (uncontrolled) : all pC1 on each side ---
    for which, s in sides.items():
        rows.append({"comparison": "full pC1 (biological)", "sex": which,
                     "n_stim": len(s["gi"]["pC1"]),
                     "descending_hz": dn_rate(s, s["gi"]["pC1"])})

    # --- stimulus-matched: 10 pC1 each side ---
    n_match = min(len(sides["male"]["gi"]["pC1"]), len(sides["female"]["gi"]["pC1"]))
    female_val = dn_rate(sides["female"], sides["female"]["gi"]["pC1"])
    male_sub = [dn_rate(sides["male"], rng.choice(sides["male"]["gi"]["pC1"], n_match, replace=False))
                for _ in range(args.n_sub)]
    rows.append({"comparison": f"pC1 matched to n={n_match}", "sex": "female",
                 "n_stim": n_match, "descending_hz": female_val})
    rows.append({"comparison": f"pC1 matched to n={n_match}", "sex": "male",
                 "n_stim": n_match, "descending_hz": float(np.mean(male_sub)),
                 "male_sub_std": float(np.std(male_sub)),
                 "male_sub_min": float(np.min(male_sub)),
                 "male_sub_max": float(np.max(male_sub))})
    # Where does the female sit in the male subsample distribution?
    male_sub = np.array(male_sub)
    p_two = (int((np.abs(male_sub - np.mean(male_sub)) >=
                  abs(female_val - np.mean(male_sub))).sum()) + 1) / (len(male_sub) + 1)
    out["male_subsample_descending_hz"] = male_sub.tolist()
    out["female_descending_hz"] = female_val
    out["female_vs_male_subsample_p"] = p_two

    # --- specificity on both sides ---
    for which, s in sides.items():
        central = "cb_intrinsic" if which == "male" else "central"
        pool = np.where(s["sc"] == central)[0]
        ctrl = rng.choice(pool, n_match, replace=False)
        rows.append({"comparison": f"random {central} (specificity control)",
                     "sex": which, "n_stim": n_match,
                     "descending_hz": dn_rate(s, ctrl)})

    df = pd.DataFrame(rows)
    print("\n=== descending-neuron output (the shared readout) ===")
    print(df.to_string(index=False))
    print(f"\nfemale {female_val:.3f} Hz vs male subsampled "
          f"{male_sub.mean():.3f} +/- {male_sub.std():.3f} Hz   two-sided p = {p_two:.4f}")

    out["rows"] = rows
    (P / "sex_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {P / 'sex_comparison.json'}")


if __name__ == "__main__":
    main()
