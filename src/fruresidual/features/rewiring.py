"""Phase 2: define the per-neuron rewiring target (docs/project_plan.md
section d, PHASE 2) -- how different is a MaleCNS neuron's wiring from
its matched FlyWire (female) type's average wiring?

Scope decision, made explicit rather than silently assumed: FlyWire
(FAFB) is brain-only, no VNC (see docs/project_plan.md section b.2). So
"case" neurons here are restricted to brain-like superclasses (excludes
vnc_*, ascending_neuron, descending_neuron -- these have no fair
FlyWire-side comparison since their defining connectivity leaves the
brain). Within that scope, both sides' partner profiles are restricted
to the 10 buckets SHARED_CATEGORIES defines, so a partner only one
dataset can see never masquerades as a "wiring difference" driven by
biology when it is actually a dataset coverage gap.

That restriction has to hold in BOTH directions, which is the part an
earlier version of this module got wrong: it dropped MaleCNS-only
partner categories but left two FlyWire-only ones (ascending/descending)
in the comparison vector. See the MALECNS_TO_SHARED comment below; the
module-level assert now enforces symmetry at import time.
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

# MaleCNS superclass -> shared category. Values not in this map are
# dropped when building profiles.
#
# ascending_neuron/descending_neuron map to FlyWire's ascending/descending
# and must NOT be dropped. They were, until 2026-09-19, and that was a
# real bug: FlyWire reports ascending/descending partners (104K weight
# across 293 of 371 types) while the male side was structurally unable to
# express those two dimensions at all, so every bit of that female-side
# weight inflated the cosine distance one-sidedly -- exactly the
# coverage-gap-as-wiring-difference artifact this module's docstring
# claims to prevent. It accounted for ~76% of the measured signal.
#
# Being excluded from the CASE population (VNC_ONLY_SUPERCLASSES, since a
# neuron whose defining connectivity leaves the brain can't be fairly
# scored against brain-only FlyWire) is a separate question from being a
# valid PARTNER category -- an ascending neuron is a perfectly real
# partner of a brain neuron, and both datasets can see that partner.
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
    "ascending_neuron": "ascending",
    "descending_neuron": "descending",
}

# Categories reachable from each side. Every SHARED_CATEGORY must be
# reachable from the MaleCNS side too, or the comparison is asymmetric by
# construction -- asserted at import so the bug above cannot silently
# return if either vocabulary changes.
_MALE_REACHABLE = set(MALECNS_TO_SHARED.values())
_UNREACHABLE = [c for c in SHARED_CATEGORIES if c not in _MALE_REACHABLE]
assert not _UNREACHABLE, (
    f"SHARED_CATEGORIES contains categories no MaleCNS superclass maps to: {_UNREACHABLE}. "
    "The male profile vector would always be 0 there while FlyWire's is not, inflating "
    "cosine distance one-sidedly. Either map a MaleCNS superclass to them or remove them."
)


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
    """1 - cosine similarity.

    Returns NaN, not 1.0, when either vector is all-zero. An all-zero
    profile means "this neuron has no connectivity in the shared
    categories, so there is nothing to compare" -- that is missing data,
    and scoring it as 1.0 would assert maximal rewiring on the basis of
    no evidence, putting it above every genuinely-measured neuron. NaN
    makes callers drop it explicitly instead (scripts/run_phase2_data.py
    does). No neuron in the current population hits this path, but the
    1.0 default was a silent trap if coverage ever changed.
    """
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(1 - np.dot(a, b) / (norm_a * norm_b))
