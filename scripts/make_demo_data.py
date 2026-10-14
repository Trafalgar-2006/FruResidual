"""Export a compact JSON bundle for the interactive demo.

Re-runs the cascade under each headline condition and collects the
already-computed summaries, so the demo page is driven by real simulation
output rather than numbers retyped into a webpage.

Usage:  python scripts/make_demo_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtship.circuit import resolve_groups, to_indices
from courtship.lif import LIFParams, group_rate, simulate

P = Path("C:/CourtshipData/processed")
OUT = Path(__file__).resolve().parents[1] / "reports" / "demo_data.json"
DURATION_MS = 200.0
BIN_MS = 10.0
STAGES = ["pC1", "pIP10", "dPR1", "TN1", "wing_motor"]


def main() -> None:
    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    pos = {b: i for i, b in enumerate(ids)}
    ann = load_male_cns_annotations(load_config(), traced_only=True)
    gi = to_indices(resolve_groups(ann), ids)
    params = LIFParams()
    stim_set = set(gi["pC1"].tolist())

    res = pd.read_csv(P / "residual_body_ids.csv")
    exp = pd.read_csv(P / "explained_body_ids.csv")

    def idx(df, classes):
        return np.array([pos[b] for b in df[df["superclass"].isin(classes)]["bodyId"]
                         if b in pos and pos[b] not in stim_set], dtype=np.int64)

    pip10 = set(gi["pIP10"].tolist())
    conditions = {
        "intact": None,
        "pIP10 silenced (2)": gi["pIP10"],
        "TN1 silenced (35)": gi["TN1"],
        "residual descending silenced (89)": idx(res, ["descending_neuron"]),
        "residual ascending silenced (84)": idx(res, ["ascending_neuron"]),
        "residual optic lobe silenced (490)": idx(res, ["ol_intrinsic"]),
        "explained descending minus pIP10 (14)": np.array(
            [i for i in idx(exp, ["descending_neuron"]) if i not in pip10], dtype=np.int64),
    }

    traces = {}
    for label, sil in conditions.items():
        r = simulate(adj, gi["pC1"], duration_ms=DURATION_MS, params=params,
                     silenced_idx=sil, record_every=0,
                     track_groups={k: gi[k] for k in STAGES if k in gi}, bin_ms=BIN_MS)
        traces[label] = {
            "groups": {k: [round(float(x), 2) for x in v]
                       for k, v in r["group_traces_hz"].items()},
            "wing_hz": round(group_rate(r["rates_hz"], gi["wing_motor"]), 3),
            "n_silenced": 0 if sil is None else len(sil),
        }
        print(f"{label:<42} wing {traces[label]['wing_hz']:>7.2f} Hz")

    bundle = {
        "meta": {
            "neurons": int(adj.shape[0]), "edges": int(adj.nnz),
            "synapses": int(abs(adj).sum()),
            "duration_ms": DURATION_MS, "bin_ms": BIN_MS,
            "synapse_mv": params.synapse_mv, "stim_rate_hz": params.stim_rate_hz,
            "group_sizes": {k: len(v) for k, v in gi.items()},
        },
        "traces": traces,
        "stages": [s for s in STAGES if s in gi],
    }
    for name, key in (("ablation_results.json", "ablation"),
                      ("residual_breakdown.json", "breakdown"),
                      ("sex_comparison.json", "sex"),
                      ("calibration.json", "calibration")):
        p = P / name
        if p.exists():
            bundle[key] = json.loads(p.read_text())
        else:
            print(f"  (skipping {name}: not generated yet)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle))
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
