"""Phase 5 (package): composes the four already-produced per-phase headline
figures into one 2x2 "at a glance" panel for the paper-shaped writeup and
README. Does not recompute anything -- purely a presentation step, run
after scripts/run_phase1.py..run_phase4.py.

Usage:  python scripts/make_summary_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

PANELS = [
    (FIG / "phase1" / "roi_residual_by_involvement.png",
     "A. Phase 1 -- where the residual concentrates (by neuropil)"),
    (FIG / "phase2" / "ablation_comparison.png",
     "B. Phase 2 -- fru/dsx adds ~nothing once structure is modeled"),
    (FIG / "phase3" / "cluster_scatter_ascending_neuron.png",
     "C. Phase 3 -- ascending_neuron: residual concentrates in one cluster"),
    (FIG / "phase4" / "concordance_bar.png",
     "D. Phase 4 -- BANC confirms residual and explained equally well"),
]


def main() -> None:
    missing = [p for p, _ in PANELS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing source figure(s), run the corresponding run_phaseN.py first: {missing}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (path, caption) in zip(axes.flat, PANELS, strict=True):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(caption, fontsize=11, loc="left")

    fig.suptitle(
        "The fru/dsx-unexplained residual: where it concentrates, what predicts it,\n"
        "the strongest circuit-level candidate, and independent cross-dataset confirmation",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = FIG / "summary_headline.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
