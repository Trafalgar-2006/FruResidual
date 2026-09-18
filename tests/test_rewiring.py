import numpy as np
import pandas as pd
import pytest

from fruresidual.features.rewiring import (
    SHARED_CATEGORIES,
    cosine_distance,
    malecns_superclass_to_shared,
    profile_vector,
    select_case_control_population,
)


def test_malecns_superclass_maps_to_shared_vocabulary():
    s = pd.Series(["ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "cb_motor"])
    mapped = malecns_superclass_to_shared(s)
    assert mapped.iloc[0] == "optic"
    assert mapped.iloc[1] == "central"
    assert pd.isna(mapped.iloc[2])  # vnc_intrinsic has no shared-vocabulary mapping
    assert mapped.iloc[3] == "motor"


def test_profile_vector_fixed_order_and_missing_as_zero():
    v = profile_vector({"optic": 10, "central": 5})
    assert len(v) == len(SHARED_CATEGORIES)
    assert v[SHARED_CATEGORIES.index("optic")] == 10
    assert v[SHARED_CATEGORIES.index("motor")] == 0


def test_cosine_distance_identical_vectors_is_zero():
    v = profile_vector({"optic": 10, "central": 5})
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal_vectors_is_one():
    a = profile_vector({"optic": 10})
    b = profile_vector({"central": 10})
    assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)


def test_cosine_distance_handles_all_zero_vector():
    a = np.zeros(len(SHARED_CATEGORIES))
    b = profile_vector({"optic": 10})
    assert cosine_distance(a, b) == 1.0


def _toy_population():
    n_case_super = ["ol_intrinsic"] * 6 + ["cb_intrinsic"] * 4
    n_control_pool = ["ol_intrinsic"] * 50 + ["cb_intrinsic"] * 50 + ["vnc_intrinsic"] * 50
    df = pd.DataFrame(
        {
            "dimorphism": ["sexually dimorphic"] * 10 + [None] * 150,
            "superclass": n_case_super + n_control_pool,
            "flywireType": ["T1"] * 6 + ["T2"] * 4 + ["T1"] * 50 + ["T2"] * 50 + ["T3"] * 50,
        }
    )
    return df


def test_select_case_control_population_excludes_vnc_and_unmatched_types():
    df = _toy_population()
    cases, controls = select_case_control_population(df, flywire_cell_types={"T1", "T2"}, control_ratio=1.0)
    assert set(cases["superclass"]) <= {"ol_intrinsic", "cb_intrinsic"}
    assert "vnc_intrinsic" not in set(controls["superclass"])  # excluded: no FlyWire VNC
    assert set(cases["flywireType"]) <= {"T1", "T2"}
    assert len(cases) == 10  # all 10 toy cases are eligible


def test_select_case_control_population_matches_case_superclass_distribution():
    df = _toy_population()
    cases, controls = select_case_control_population(df, flywire_cell_types={"T1", "T2"}, control_ratio=1.0)
    case_frac = cases["superclass"].value_counts(normalize=True)
    control_frac = controls["superclass"].value_counts(normalize=True)
    for sc in case_frac.index:
        assert control_frac.get(sc, 0) == pytest.approx(case_frac[sc], abs=0.15)
