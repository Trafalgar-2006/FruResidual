"""Where inside the residual does the effect live?

The headline ablation showed the residual as a whole matters (15.4% drop)
while its optic-lobe half does nothing (0.0%). Those two facts together
mean the effect is concentrated somewhere, and 'somewhere' is worth
naming: a 7% drop spread over 915 neurons is a different claim from a 7%
drop carried by 33.

Each superclass is ablated on its own, against degree-matched nulls of
the same size, exactly as in run_ablation.py.

Usage:  python scripts/run_residual_breakdown.py [--n-null 50]
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

from courtship.ablation import degree_matched_sample, empirical_p, out_synapses  # noqa: E402
from courtship.circuit import resolve_groups, to_indices  # noqa: E402
from courtship.lif import LIFParams, group_rate, simulate  # noqa: E402

P = Path("C:/CourtshipData/processed")
DURATION_MS = 200.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=50)
    args = ap.parse_args()

    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    pos = {b: i for i, b in enumerate(ids)}
    ann = load_male_cns_annotations(load_config(), traced_only=True)
    gi = to_indices(resolve_groups(ann), ids)
    params = LIFParams()
    out_syn = out_synapses(adj)
    stim_set = set(gi["pC1"].tolist())

    def wing_rate(silenced=None) -> float:
        r = simulate(adj, gi["pC1"], duration_ms=DURATION_MS, params=params,
                     silenced_idx=silenced, record_every=0)
        return group_rate(r["rates_hz"], gi["wing_motor"])

    baseline = wing_rate()
    print(f"baseline wing motor: {baseline:.2f} Hz\n")

    res = pd.read_csv(P / "residual_body_ids.csv")
    rng = np.random.default_rng(0)
    rows = []

    for sc, grp in res.groupby("superclass"):
        target = np.array([pos[b] for b in grp["bodyId"]
                           if b in pos and pos[b] not in stim_set], dtype=np.int64)
        if len(target) < 2:
            continue
        obs = wing_rate(target)
        drop = baseline - obs
        nulls = np.array([baseline - wing_rate(degree_matched_sample(target, out_syn, rng))
                          for _ in range(args.n_null)])
        p = empirical_p(drop, nulls)
        rows.append({"superclass": sc, "n": int(len(target)),
                     "wing_hz": obs, "pct_drop": 100 * drop / baseline,
                     "pct_drop_per_neuron": 100 * drop / baseline / len(target),
                     "null_pct_drop_mean": 100 * float(nulls.mean()) / baseline,
                     "p_empirical": p})
        print(f"{sc:<22} n={len(target):>4}  drop {100*drop/baseline:>6.2f}%  "
              f"null {100*nulls.mean()/baseline:>6.2f}%  p={p:.4f}")

    df = pd.DataFrame(rows).sort_values("pct_drop", ascending=False)
    print("\n=== residual, by superclass (sorted) ===")
    print(df.to_string(index=False))
    (P / "residual_breakdown.json").write_text(json.dumps(
        {"baseline_wing_hz": baseline, "n_null": args.n_null,
         "rows": df.to_dict("records")}, indent=2))
    print(f"\nwrote {P / 'residual_breakdown.json'}")


if __name__ == "__main__":
    main()
