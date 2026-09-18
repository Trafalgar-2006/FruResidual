"""Phase 2 feature engineering. Deliberately excludes anything that could
let a model "cheat" at predicting wiring difference: no dimorphism label,
no raw cell-type/flywireType identity (with ~8,199 distinct types across
~1,400 rows, a model could otherwise memorize per-type answers instead of
learning generalizable structure), and no fru/dsx UNLESS explicitly
requested via include_fru_dsx -- that flag exists only for the ablation
comparison in scripts/run_phase2.py, never for the "real" feature set.
See docs/project_plan.md section (d), PHASE 2, step 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NON_LEAKING_CATEGORICALS = ["superclass", "subclass", "somaSide", "consensus_nt"]

LEAKAGE_COLUMNS = frozenset(
    {"dimorphism", "matchingNotes", "flywireType", "type", "instance", "hemibrainType",
     "mancType", "synonyms", "vfbId", "bodyId", "group", "fruDsx"}
)


def build_connectivity_summary_features(male_profiles_long: pd.DataFrame) -> pd.DataFrame:
    """Per-bodyId structural summary stats from the MaleCNS partner-
    superclass profile: total degree (connectivity weight), and Shannon
    entropy of the partner-superclass distribution (how "specialized" vs.
    "generalist" a neuron's connectivity is) -- legitimate connectivity
    context that doesn't encode cell-type identity or the fru/dsx answer.
    """
    def entropy(weights: pd.Series) -> float:
        p = weights / weights.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    grouped = male_profiles_long.groupby("bodyId")["weight"]
    summary = pd.DataFrame(
        {
            "total_connectivity_weight": grouped.sum(),
            "n_partner_superclasses": male_profiles_long.groupby("bodyId")["partner_superclass"].nunique(),
            "connectivity_entropy": grouped.apply(entropy),
        }
    )
    return summary.reset_index()


def build_features(
    population: pd.DataFrame,
    male_profiles_long: pd.DataFrame,
    include_fru_dsx: bool = False,
) -> pd.DataFrame:
    """Builds the feature matrix for `population` (from
    fruresidual.features.rewiring.select_case_control_population).

    include_fru_dsx=True adds a single boolean feature (fru/dsx positive,
    any_positive rule) -- exists solely for the with/without ablation.
    """
    conn_features = build_connectivity_summary_features(male_profiles_long)
    df = population.merge(conn_features, on="bodyId", how="left")

    X = df[["bodyId"]].copy()
    # One-hot, not .cat.codes. These are nominal categories with no
    # ordering, and integer codes assert a false one: a linear model reads
    # "superclass 7" as seven times "superclass 1". Tree baselines are
    # immune (they split on arbitrary thresholds), which is exactly why
    # the bug was invisible in the gradient-boosting numbers while
    # quietly handicapping ridge. Cardinality here is low (~35 columns
    # total), so one-hot costs nothing.
    for col in NON_LEAKING_CATEGORICALS:
        dummies = pd.get_dummies(df[col].astype("category"), prefix=col, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    X["total_connectivity_weight"] = df["total_connectivity_weight"].fillna(0)
    X["n_partner_superclasses"] = df["n_partner_superclasses"].fillna(0)
    X["connectivity_entropy"] = df["connectivity_entropy"].fillna(0)

    if include_fru_dsx:
        from fruresidual.features.fru_dsx_join import has_fru_dsx

        # Named differently from the raw "fruDsx" column so it can never
        # collide with the leakage check below.
        X["fru_dsx_positive"] = has_fru_dsx(df, "any_positive").astype(int)

    # bodyId is kept (for joining to targets downstream, dropped before
    # fitting any model) -- everything else in X must NOT be a leakage
    # column, including "fruDsx" itself when include_fru_dsx=False.
    leaked = (LEAKAGE_COLUMNS - {"bodyId"}) & set(X.columns)
    assert not leaked, f"leakage column(s) slipped into features: {leaked}"
    return X
