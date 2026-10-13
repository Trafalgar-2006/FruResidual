"""Does amputating the nerve cord change the brain's own dynamics?

Every connectome-based model published so far runs on a brain-only
reconstruction (FlyWire/FAFB) and wires brain output straight to a
simulated body. FlyGM (arXiv 2602.17997) lists synapse detection,
cell-type annotation and compute as its limitations; the missing nerve
cord is not among them, and the paper never states the connectome is
brain-only or female.

MaleCNS contains the whole CNS, so the assumption can simply be tested:
simulate the identical stimulus on the full network and on a brain-only
version of itself, and compare the BRAIN's activity. Everything else is
held constant -- same neurons, same synapses, same parameters -- so any
difference is caused by the missing nerve cord and nothing else.

The mechanism at stake is the 1,846 ascending neurons, which carry VNC
-> brain feedback. If removing them changes the command circuit, then
brain-only models are missing a real effect rather than a tidy
abstraction boundary.

Usage:  python scripts/run_brain_only.py
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

from courtship.circuit import resolve_groups, to_indices  # noqa: E402
from courtship.lif import LIFParams, group_rate, simulate  # noqa: E402

P = Path("C:/CourtshipData/processed")
DURATION_MS = 200.0
# Classes that live in the ventral nerve cord. Descending neurons are kept:
# their somas are in the brain and FlyWire contains them (truncated at the
# neck), which is exactly the boundary a brain-only reconstruction draws.
VNC_CLASSES = ("vnc_intrinsic", "vnc_motor", "vnc_sensory",
               "vnc_efferent", "vnc_endocrine", "ascending_neuron")


def main() -> None:
    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    ann = load_male_cns_annotations(load_config(), traced_only=True)
    gi = to_indices(resolve_groups(ann), ids)
    sc = ann.set_index("bodyId").reindex(ids)["superclass"].to_numpy()

    is_vnc = np.isin(sc, VNC_CLASSES)
    vnc_idx = np.where(is_vnc)[0]
    ascending = np.where(sc == "ascending_neuron")[0]
    print(f"whole CNS: {len(ids):,} neurons")
    print(f"  nerve-cord side silenced for the brain-only condition: {len(vnc_idx):,}")
    print(f"  of which ascending (VNC -> brain feedback): {len(ascending):,}")

    params = LIFParams()
    track = {k: gi[k] for k in ("pC1", "pIP10", "dPR1", "TN1", "wing_motor") if k in gi}

    def run(silenced, label):
        r = simulate(adj, gi["pC1"], duration_ms=DURATION_MS, params=params,
                     silenced_idx=silenced, record_every=0,
                     track_groups=track, bin_ms=10.0)
        rt = r["rates_hz"]
        row = {"condition": label,
               "pIP10_hz": group_rate(rt, gi["pIP10"]),
               "dPR1_hz": group_rate(rt, gi["dPR1"]),
               "TN1_hz": group_rate(rt, gi["TN1"]),
               "wing_hz": group_rate(rt, gi["wing_motor"]),
               "brain_mean_hz": float(rt[~is_vnc].mean())}
        return row, r

    rows = []
    row_full, r_full = run(None, "whole CNS")
    rows.append(row_full)

    # Brain-only: the entire nerve cord silenced, as a brain-only
    # reconstruction would have it.
    row_brain, r_brain = run(vnc_idx, "brain only (no VNC)")
    rows.append(row_brain)

    # Isolate the feedback: silence ONLY the ascending neurons, leaving the
    # rest of the VNC intact. If this reproduces the brain-only effect, the
    # feedback loop is the mechanism rather than bulk neuron removal.
    row_asc, _ = run(ascending, "ascending neurons only")
    rows.append(row_asc)

    df = pd.DataFrame(rows)
    print("\n=== brain activity, with and without the nerve cord ===")
    print(df.to_string(index=False))

    d_full = row_full["pIP10_hz"]
    d_brain = row_brain["pIP10_hz"]
    print(f"\npIP10 (the song command's output channel):")
    print(f"  whole CNS   {d_full:.1f} Hz")
    print(f"  brain only  {d_brain:.1f} Hz   "
          f"({100*(d_brain-d_full)/d_full:+.1f}% vs whole CNS)" if d_full else "")

    out = {"duration_ms": DURATION_MS, "synapse_mv": params.synapse_mv,
           "n_vnc_silenced": int(len(vnc_idx)), "n_ascending": int(len(ascending)),
           "conditions": rows,
           "pIP10_trace_whole_cns": r_full["group_traces_hz"]["pIP10"].tolist(),
           "pIP10_trace_brain_only": r_brain["group_traces_hz"]["pIP10"].tolist(),
           "bin_ms": r_full["bin_ms"]}
    (P / "brain_only_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {P / 'brain_only_results.json'}")


if __name__ == "__main__":
    main()
