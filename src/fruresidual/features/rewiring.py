"""Phase 2: define the per-neuron rewiring target (docs/project_plan.md
section d, PHASE 2) -- how different is a MaleCNS neuron's wiring from
its matched FlyWire (female) type's average wiring?

Scope decision, made explicit rather than silently assumed: FlyWire
(FAFB) is brain-only, no VNC (see docs/project_plan.md section b.2). So
"case" neurons here are restricted to brain-like superclasses (excludes
vnc_*, ascending_neuron, descending_neuron -- these have no fair
FlyWire-side comparison since their defining connectivity leaves the
brain). Within that scope, both sides' partner profiles are further
restricted to the 10 superclass buckets FlyWire can express -- computed
via SHARED_CATEGORIES below -- so a synapse to a VNC partner (which
MaleCNS can see and FlyWire structurally cannot) never masquerades as a
"wiring difference" driven by biology when it's actually a dataset
coverage gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Superclasses excluded from the case population: their defining
# connectivity is VNC-based, which FlyWire (brain-only) cannot represent
# fairly. See module docstring.
VNC_ONLY_SUPERCLASSES = frozenset(
    {
        "vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent", "vnc_endocrine",
        "vnc_sensory_tbc", "vnc_tbc", "ascending_neuron", "descending_neuron",
        "sensory_descending", "efferent_ascending", "efferent_descending", "ENS",
    }
)

# FlyWire's super_class vocabulary IS the shared coordinate system (it's
# the coarser, brain-only version of MaleCNS's superclass -- verified by
# inspection 2026-09-17). Only these 10 categories are used when scoring
# wiring difference; anything else (VNC partners) is dropped from both
# sides before comparing, not silently zero-filled.
SHARED_CATEGORIES = [
    "optic", "central", "sensory", "visual_projection", "ascending",
    "descending", "sensory_ascending", "visual_centrifugal", "motor", "endocrine",
]

# MaleCNS superclass -> shared category. Values not in this map (VNC-only
# ones) are dropped when building profiles.
MALECNS_TO_SHARED = {
    "ol_intrinsic": "optic",
    "ol_sensory": "optic",
    "cb_intrinsic": "central",
    "cb_sensory": "sensory",
    "cb_motor": "motor",
    "cb_endocrine": "endocrine",
    "cb_efferent": "motor",
    "visual_projection": "visual_projection",
    "visual_projection_tbc": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "sensory_ascending": "sensory_ascending",
    "sensory_ascending_tbc": "sensory_ascending",
}


def select_case_control_population(
    df: pd.DataFrame, flywire_cell_types: set[str], control_ratio: float = 1.0, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cases: sexually-dimorphic (matched-pair) neurons with a clean
    single-value flywireType, restricted to brain-like superclasses.
    Controls: an equal-superclass-distribution random sample of isomorphic
    neurons from the same eligible pool, at control_ratio x len(cases).

    male-specific/potentially-male-specific are excluded from cases by
    construction (they have no FlyWire counterpart at all -- see
    docs/phase2_results.md for the verified 0% flywireType coverage).

    `flywire_cell_types` (from flywire_annotations["cell_type"].unique())
    matters: MaleCNS's flywireType field is the MaleCNS team's *label* for
    the corresponding FlyWire type, but that string doesn't always exist
    verbatim in FlyWire's own cell_type vocabulary -- verified on
    2026-09-17, ~91-93% real match rate, not 100%. Filtering on this
    (rather than just flywireType.notna()) avoids silently including
    neurons with no real FlyWire counterpart to compare against.
    """
    brain_like = ~df["superclass"].isin(VNC_ONLY_SUPERCLASSES)
    clean_type = (
        df["flywireType"].notna()
        & ~df["flywireType"].str.contains(",", na=False)
        & df["flywireType"].isin(flywire_cell_types)
    )

    cases = df[
        df["dimorphism"].isin({"sexually dimorphic", "potentially sexually dimorphic"})
        & clean_type
        & brain_like
    ].copy()
    cases["is_case"] = True

    pool = df[df["dimorphism"].isna() & clean_type & brain_like].copy()
    case_superclass_dist = cases["superclass"].value_counts(normalize=True)
    n_controls = int(len(cases) * control_ratio)

    rng = np.random.default_rng(random_state)
    picks = []
    for sc, frac in case_superclass_dist.items():
        sc_pool = pool[pool["superclass"] == sc]
        n_pick = min(len(sc_pool), round(n_controls * frac))
        if n_pick > 0:
            idx = rng.choice(sc_pool.index, size=n_pick, replace=False)
            picks.append(sc_pool.loc[idx])
    controls = pd.concat(picks) if picks else pool.iloc[0:0]
    controls = controls.copy()
    controls["is_case"] = False

    return cases, controls


def malecns_superclass_to_shared(superclass_series: pd.Series) -> pd.Series:
    return superclass_series.map(MALECNS_TO_SHARED)


def profile_vector(category_weights: dict[str, float]) -> np.ndarray:
    """Turns a {shared_category: weight} dict into a fixed-order vector
    over SHARED_CATEGORIES (0 for missing categories)."""
    return np.array([category_weights.get(c, 0.0) for c in SHARED_CATEGORIES], dtype=float)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. Returns 1.0 (max distance) if either vector
    is all-zero (no shared-category connectivity to compare)."""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1 - np.dot(a, b) / (norm_a * norm_b))
