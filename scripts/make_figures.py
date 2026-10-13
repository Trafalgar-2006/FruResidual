"""Figures: the song cascade propagating, and what silencing breaks it.

Usage:  python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fruresidual.config import load_config  # noqa: E402
from fruresidual.features.fru_dsx_join import load_male_cns_annotations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtship.circuit import resolve_groups, to_indices  # noqa: E402
from courtship.lif import LIFParams, simulate  # noqa: E402

P = Path("C:/CourtshipData/processed")
FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
STAGES = ["pC1", "pIP10", "dPR1", "TN1", "wing_motor"]
LABELS = {"pC1": "pC1 (command)", "pIP10": "pIP10 (descending)",
          "dPR1": "dPR1 (VNC)", "TN1": "TN1 (VNC premotor)",
          "wing_motor": "wing motor (song)"}


def cascade_panel(ax, adj, gi, silenced, title):
    r = simulate(adj, gi["pC1"], duration_ms=200.0, params=LIFParams(),
                 silenced_idx=silenced, record_every=0,
                 # 10 ms bins: pIP10 and dPR1 are 2 neurons each, so at 5 ms
                 # their rate can only be 0, 200 or 400 Hz and the trace is
                 # mostly quantisation artefact.
                 track_groups={k: gi[k] for k in STAGES if k in gi}, bin_ms=10.0)
    tr = r["group_traces_hz"]
    t = np.arange(len(next(iter(tr.values())))) * r["bin_ms"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(STAGES)))
    for c, stage in zip(colors, STAGES, strict=False):
        if stage in tr:
            ax.plot(t, tr[stage], label=LABELS[stage], color=c, lw=2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_ylim(bottom=0)
    return r


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    adj = sp.load_npz(P / "malecns_adj_minw1.npz").tocsr()
    ids = np.load(P / "malecns_node_ids.npy")
    ann = load_male_cns_annotations(load_config(), traced_only=True)
    gi = to_indices(resolve_groups(ann), ids)

    # --- Figure 1: the cascade, intact vs pIP10 silenced ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    cascade_panel(axes[0], adj, gi, None, "Intact male: pC1 stimulated")
    cascade_panel(axes[1], adj, gi, gi["pIP10"],
                  "pIP10 silenced (2 neurons)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("The courtship song cascade, brain to wing muscle",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "cascade.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'cascade.png'}")

    # --- Figure 2: ablation results ---
    res_path = P / "ablation_results.json"
    if not res_path.exists():
        print("ablation_results.json not found -- run scripts/run_ablation.py first")
        return
    data = json.loads(res_path.read_text())
    conds = data["conditions"]
    labels = [c["condition"] for c in conds]
    drops = [c["pct_drop"] for c in conds]
    nulls = [100 * c["null_drop_mean"] / data["baseline_wing_hz"] for c in conds]
    null_sd = [100 * c["null_drop_std"] / data["baseline_wing_hz"] for c in conds]

    def color(lbl):
        if "positive control" in lbl:
            return "#555555"
        if lbl.startswith("residual"):
            return "#c2410c"
        return "#1d4ed8"

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(labels) + 2.2))
    ax.barh(y, drops, color=[color(x) for x in labels], height=0.6,
            label="observed", zorder=3)
    ax.errorbar(nulls, y, xerr=null_sd, fmt="o", color="k", ms=4,
                lw=1.2, capsize=3, zorder=4, label="degree-matched null")
    for i, c in enumerate(conds):
        ax.text(max(drops) * 1.02, i, f"n={c['n_silenced']:,}  p={c['p_empirical']:.3f}",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("drop in song output (% of intact wing-motor rate)")
    ax.set_title("What silencing breaks the song\n"
                 f"(baseline {data['baseline_wing_hz']:.1f} Hz; "
                 f"{data['n_null']} degree-matched random ablations per condition)",
                 fontsize=12)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False)
    ax.set_xlim(right=max(drops) * 1.35)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'ablation.png'}")


if __name__ == "__main__":
    main()
