"""The experiment: does silencing the fru/dsx-unexplained residual break
the courtship song?

Protocol, mirroring the real optogenetic experiment (P1 activation drives
song): stimulate pC1 at 150 Hz, read out mean firing rate of T2 wing
motor neurons. Then silence a set of neurons and re-measure.

Silencing is a per-run mask. The connectome is never modified.

Every ablation is compared against degree-matched random ablations of the
SAME SIZE, because removing high-traffic neurons weakens the song whoever
they are. Positive controls (pIP10, TN1) must break the song; a random
set of the same size must not. If those fail, nothing else is readable.

Usage:  python scripts/run_ablation.py [--n-null 50]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtship.ablation import (
    degree_matched_sample,
    empirical_p,
    out_synapses,
)
from courtship.circuit import resolve_groups, to_indices
from courtship.lif import LIFParams, group_rate, simulate

P = Path("C:/CourtshipData/processed")
DURATION_MS = 200.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=50,
                    help="degree-matched random ablations per condition")
    args = ap.parse_args()

    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    pos = {b: i for i, b in enumerate(ids)}
    ann = load_male_cns_annotations(load_config(), traced_only=True)
    gi = to_indices(resolve_groups(ann), ids)
    params = LIFParams()
    out_syn = out_synapses(adj)

    def wing_rate(silenced=None) -> float:
        r = simulate(adj, gi["pC1"], duration_ms=DURATION_MS, params=params,
                     silenced_idx=silenced, record_every=0)
        return group_rate(r["rates_hz"], gi["wing_motor"])

    t0 = time.time()
    baseline = wing_rate()
    per_run = time.time() - t0
    print(f"intact male, pC1 stimulated -> wing motor {baseline:.2f} Hz "
          f"({per_run:.1f}s per run)")
    if baseline < 1.0:
        sys.exit("baseline song too weak to measure a lesion against")

    # --- the ablation targets ---
    res = pd.read_csv(P / "residual_body_ids.csv")
    exp = pd.read_csv(P / "explained_body_ids.csv")

    def idx(df, superclasses=None):
        d = df if superclasses is None else df[df["superclass"].isin(superclasses)]
        return np.array([pos[b] for b in d["bodyId"] if b in pos], dtype=np.int64)

    # An ablation set that contains the STIMULATED neurons silences the
    # stimulus itself, which abolishes the song for trivial reasons and has
    # nothing to do with the neurons being studied. pC1 is both the command
    # population and fru/dsx-positive, so it lands in the "explained" set
    # and would hand that arm a free 100% drop. Excluded from every target.
    stim_set = set(gi["pC1"].tolist())

    def drop_stim(a: np.ndarray, label: str) -> np.ndarray:
        keep = np.array([i for i in a if i not in stim_set], dtype=np.int64)
        if len(keep) != len(a):
            print(f"  [{label}] removed {len(a) - len(keep)} stimulated pC1 neurons "
                  f"from the ablation set")
        return keep

    CENTRAL = ["cb_intrinsic", "descending_neuron", "ascending_neuron"]
    pip10 = set(gi["pIP10"].tolist())
    conditions = {
        # positive controls -- these MUST break the song
        "pIP10 (positive control)": gi["pIP10"],
        "TN1 (positive control)": gi["TN1"],
        # the actual question
        "residual: descending only": idx(res, ["descending_neuron"]),
        "residual: central (cb+desc+asc)": idx(res, CENTRAL),
        "residual: all": idx(res),
        "residual: optic lobe only": idx(res, ["ol_intrinsic"]),
        # biological control: dimorphic neurons the switch DOES explain
        "explained: descending only": idx(exp, ["descending_neuron"]),
        # pIP10 is fru/dsx-positive AND the song pathway's bottleneck, so it
        # sits inside "explained" and would otherwise drive that arm's whole
        # effect on its own. Reported both ways rather than silently either.
        "explained: descending minus pIP10": np.array(
            [i for i in idx(exp, ["descending_neuron"]) if i not in pip10], dtype=np.int64),
        "explained: central (cb+desc+asc)": idx(exp, CENTRAL),
    }
    conditions = {k: drop_stim(v, k) for k, v in conditions.items()}

    rng = np.random.default_rng(0)
    results = []
    for label, target in conditions.items():
        if len(target) == 0:
            print(f"\n{label}: empty, skipped")
            continue
        obs = wing_rate(target)
        drop = baseline - obs
        pct = 100 * drop / baseline

        nulls = []
        for _ in range(args.n_null):
            ctrl = degree_matched_sample(target, out_syn, rng)
            nulls.append(baseline - wing_rate(ctrl))
        nulls = np.array(nulls)
        p = empirical_p(drop, nulls)

        results.append({
            "condition": label, "n_silenced": len(target),
            "wing_hz": obs, "drop_hz": drop, "pct_drop": pct,
            "null_drop_mean": float(nulls.mean()), "null_drop_std": float(nulls.std()),
            "p_empirical": p, "n_null": args.n_null,
        })
        print(f"\n{label}")
        print(f"  silenced {len(target):,} neurons -> wing {obs:.2f} Hz "
              f"(drop {drop:.2f} Hz, {pct:.1f}%)")
        print(f"  degree-matched null drop: {nulls.mean():.2f} +/- {nulls.std():.2f} Hz")
        print(f"  empirical p = {p:.4f}")

    out = {"baseline_wing_hz": baseline, "duration_ms": DURATION_MS,
           "synapse_mv": params.synapse_mv, "stim_rate_hz": params.stim_rate_hz,
           "n_null": args.n_null, "conditions": results}
    (P / "ablation_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {P / 'ablation_results.json'}")

    print("\n=== summary ===")
    df = pd.DataFrame(results)[["condition", "n_silenced", "pct_drop",
                                "null_drop_mean", "p_empirical"]]
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
