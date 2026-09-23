"""The FCA loader's risky step is parsing a cell type out of a free-text
annotation string. These pin the validation that stops it inventing one.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("anndata")

from fruresidual.data.fca_client import load_fca_fru_dsx


def _write_h5ad(tmp_path, annotations, sexes, fru, dsx):
    import anndata as ad

    obs = pd.DataFrame({
        "annotation": pd.Categorical(annotations),
        "sex": pd.Categorical(sexes),
    }, index=[f"c{i}" for i in range(len(annotations))])
    x = np.column_stack([np.asarray(fru, float), np.asarray(dsx, float)])
    a = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=["fru", "dsx"]))
    p = tmp_path / "fca.h5ad"
    a.write_h5ad(p)
    return p


def test_only_annotations_matching_the_connectome_vocabulary_are_kept(tmp_path):
    """The type is the last token of a descriptive string, which could
    easily produce garbage ("cell", "neuron", "unknown"). Validating
    against the connectome's own vocabulary makes an invented type
    impossible -- at the cost of dropping annotations with no connectome
    counterpart, which is the right trade."""
    p = _write_h5ad(
        tmp_path,
        annotations=["medullary intrinsic neuron Mi1", "medullary intrinsic neuron Mi1",
                     "Kenyon cell", "unknown", "transmedullary neuron Tm9"],
        sexes=["male", "female", "male", "female", "male"],
        fru=[1, 0, 5, 5, 2], dsx=[0, 0, 5, 5, 1],
    )
    out = load_fca_fru_dsx(p, connectome_types={"Mi1", "Tm9"})
    assert set(out["cell_type"]) == {"Mi1", "Tm9"}
    assert "cell" not in set(out["cell_type"])
    assert "unknown" not in set(out["cell_type"])


def test_pct_expressing_is_fraction_of_nonzero_cells(tmp_path):
    # Raw counts are unnormalised, so "fraction of cells with any
    # transcript" is the robust statistic -- no library-size correction
    # needed, and directly comparable to Ozel's pct.1/pct.2.
    p = _write_h5ad(
        tmp_path,
        annotations=["medullary intrinsic neuron Mi1"] * 4,
        sexes=["male"] * 4,
        fru=[0, 3, 7, 0], dsx=[0, 0, 0, 0],
    )
    out = load_fca_fru_dsx(p, connectome_types={"Mi1"})
    fru = out[(out["cell_type"] == "Mi1") & (out["gene"] == "fru")].iloc[0]
    assert fru["n_cells"] == 4
    assert fru["n_expressing"] == 2
    assert fru["pct_expressing"] == pytest.approx(50.0)
    dsx = out[(out["cell_type"] == "Mi1") & (out["gene"] == "dsx")].iloc[0]
    assert dsx["pct_expressing"] == pytest.approx(0.0)


def test_sexes_are_reported_separately_and_mixed_is_dropped(tmp_path):
    # FCA has a 'mix' sex category; it cannot inform a male/female
    # comparison and must not be pooled into either.
    p = _write_h5ad(
        tmp_path,
        annotations=["medullary intrinsic neuron Mi1"] * 3,
        sexes=["male", "female", "mix"],
        fru=[1, 0, 1], dsx=[0, 0, 0],
    )
    out = load_fca_fru_dsx(p, connectome_types={"Mi1"})
    assert set(out["sex"]) == {"male", "female"}
    assert int(out[out["gene"] == "fru"]["n_cells"].sum()) == 2


def test_missing_genes_fail_loudly(tmp_path):
    p = _write_h5ad(tmp_path, ["medullary intrinsic neuron Mi1"], ["male"], [1], [0])
    with pytest.raises(ValueError) as exc:
        load_fca_fru_dsx(p, connectome_types={"Mi1"}, genes=("notagene",))
    assert "notagene" in str(exc.value)
