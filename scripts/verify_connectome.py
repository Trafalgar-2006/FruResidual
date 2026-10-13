"""Checks the built connectome against known biology.

The failure this exists to catch is transposition. If A[i,j] meant j->i
instead of i->j, every summary statistic -- node count, edge count,
density, degree distribution, E/I ratio -- would be IDENTICAL, and every
downstream result would be silently backwards. The only thing that
notices is biology: sensory cells must send more than they receive, and
motor cells must receive far more than they send.

Exits nonzero on failure, so it works as a gate before any simulation.

Usage:  python scripts/verify_connectome.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

P = Path("C:/CourtshipData/processed")
SENSORY = ("sensory_ascending", "cb_sensory", "vnc_sensory", "ol_sensory")
MOTOR = ("vnc_motor", "vnc_efferent", "cb_motor")


def main() -> None:
    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    assert adj.shape[0] == adj.shape[1] == len(ids), "matrix/node-list mismatch"

    cfg = load_config()
    ann = load_male_cns_annotations(cfg, traced_only=True).set_index("bodyId").reindex(ids)
    superclass = ann["superclass"].to_numpy()

    mag = abs(adj)
    out_syn = mag.sum(axis=1).A1
    in_syn = mag.tocsc().sum(axis=0).A1

    failures = []

    def ratio(name: str) -> float:
        m = superclass == name
        if m.sum() == 0:
            failures.append(f"superclass {name!r} absent from annotations")
            return float("nan")
        o, i = out_syn[m].mean(), in_syn[m].mean()
        return o / i if i else float("inf")

    print(f"{'class':<22}{'out/in':>9}   verdict")
    for name in SENSORY:
        r = ratio(name)
        ok = r > 1.5
        print(f"{name:<22}{r:>9.2f}   {'OK' if ok else 'FAIL (should send >> receive)'}")
        if not ok:
            failures.append(f"{name} out/in={r:.2f}, expected > 1.5")
    for name in MOTOR:
        r = ratio(name)
        ok = r < 0.5
        print(f"{name:<22}{r:>9.2f}   {'OK' if ok else 'FAIL (should receive >> send)'}")
        if not ok:
            failures.append(f"{name} out/in={r:.2f}, expected < 0.5")

    # Signs must be present and both polarities represented.
    n_exc = int((adj.data > 0).sum())
    n_inh = int((adj.data < 0).sum())
    print(f"\nexcitatory entries {n_exc:,}   inhibitory entries {n_inh:,}")
    if not (n_exc > 0 and n_inh > 0):
        failures.append("missing one polarity -- sign mapping is broken")
    if not 0.3 < n_inh / (n_exc + n_inh) < 0.6:
        failures.append(f"inhibitory fraction {n_inh/(n_exc+n_inh):.2f} outside plausible range")

    # The VNC motor output FlyWire does not have -- the point of using MaleCNS.
    n_motor = int(np.isin(superclass, MOTOR).sum())
    print(f"motor/efferent neurons present: {n_motor:,}")
    if n_motor < 500:
        failures.append(f"only {n_motor} motor neurons -- VNC may be missing")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
