"""The Codex adapter is the ingestion path for the one annotation source
that can actually settle biology-vs-ascertainment. These tests exist so
that when the export finally arrives, a format surprise fails loudly and
immediately rather than halfway through an analysis.
"""

import pandas as pd
import pytest

from fruresidual.data.flywire_client import load_codex_annotations


def _write(tmp_path, rows, columns, name="codex.csv"):
    p = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(p, index=False)
    return p


def test_normalizes_gene_field_into_fru_dsx(tmp_path):
    p = _write(tmp_path, [
        ("T1", "central", "Fruitless"),
        ("T2", "central", "Doublesex"),
        ("T3", "optic", "Acetylcholine"),
        ("T4", "optic", None),
    ], ["cell_type", "super_class", "gene"])

    out = load_codex_annotations(p)
    assert list(out.columns) == ["cell_type", "super_class", "fru_dsx"]
    pos = set(out.loc[out["fru_dsx"].notna(), "cell_type"])
    assert pos == {"T1", "T2"}, "only fru/dsx markers count as positive"


def test_gene_matching_is_case_insensitive_and_handles_multi_marker_fields(tmp_path):
    # Codex lists several markers per neuron; positive if ANY is fru/dsx.
    p = _write(tmp_path, [
        ("T1", "central", "GABA;fruitless"),
        ("T2", "central", "DSX, acetylcholine"),
        ("T3", "central", "gaba,glutamate"),
    ], ["cell_type", "super_class", "gene"])

    out = load_codex_annotations(p)
    pos = set(out.loc[out["fru_dsx"].notna(), "cell_type"])
    assert pos == {"T1", "T2"}
    assert "T3" not in pos


def test_alternative_column_names_are_auto_detected(tmp_path):
    # Codex has renamed export headers between releases, so the loader
    # matches against candidates rather than assuming one spelling.
    p = _write(tmp_path, [
        ("T1", "central", "fru"),
    ], ["type", "superclass", "genes"])

    out = load_codex_annotations(p)
    assert out.loc[0, "cell_type"] == "T1"
    assert out.loc[0, "super_class"] == "central"
    assert pd.notna(out.loc[0, "fru_dsx"])


def test_missing_column_fails_loudly_and_names_what_is_present(tmp_path):
    p = _write(tmp_path, [("T1", "central")], ["cell_type", "super_class"])
    with pytest.raises(ValueError) as exc:
        load_codex_annotations(p)
    msg = str(exc.value)
    assert "gene" in msg
    assert "Columns present" in msg, "error must show the real columns, not just fail"


def test_reads_tsv_and_rows_without_a_cell_type_are_dropped(tmp_path):
    p = tmp_path / "codex.tsv"
    pd.DataFrame(
        [("T1", "central", "fru"), (None, "central", "fru")],
        columns=["cell_type", "super_class", "gene"],
    ).to_csv(p, sep="\t", index=False)

    out = load_codex_annotations(p)
    # a fru/dsx call with no cell_type can't be joined to MaleCNS
    assert len(out) == 1
    assert out.loc[0, "cell_type"] == "T1"


def test_output_feeds_the_crosscheck_unchanged(tmp_path):
    from fruresidual.features.annotation_crosscheck import (
        build_flywire_type_lookup,
        flywire_coverage_by_super_class,
    )

    p = _write(tmp_path, [
        ("T1", "central", "fruitless"),
        ("T1", "central", None),
        ("T2", "optic", None),
    ], ["cell_type", "super_class", "gene"])
    out = load_codex_annotations(p)

    lookup = build_flywire_type_lookup(out)
    assert bool(lookup.loc["T1", "fw_fru_dsx_any"]) is True
    assert bool(lookup.loc["T2", "fw_fru_dsx_any"]) is False
    cov = flywire_coverage_by_super_class(out)
    assert cov.loc["central", "pct_positive"] == 50.0


def _curated():
    """Stand-in for the cached curated FlyWire table: supplies the full
    population plus structural fields (super_class, cell_type). Those are
    safe to reuse because they are not fru/dsx calls, so they cannot leak
    the variable under test."""
    return pd.DataFrame({
        "root_id": ["720575940600000001", "720575940600000002",
                    "720575940600000003", "720575940600000004"],
        "cell_type": ["T1", "T2", "T3", "T4"],
        "super_class": ["central", "central", "optic", "optic"],
        "fru_dsx": [None, None, None, None],
    })


def test_search_export_matches_on_root_id(tmp_path):
    from fruresidual.data.flywire_client import load_codex_gene_matches

    # a Codex "gene == Fruitless" result list: just the positives
    p = _write(tmp_path, [("720575940600000001",), ("720575940600000003",)], ["root_id"])
    out = load_codex_gene_matches(p, _curated())

    assert out.attrs["matched_on"] == "root_id"
    pos = set(out.loc[out["fru_dsx"].notna(), "cell_type"])
    assert pos == {"T1", "T3"}
    # the full population is retained, not just the positives -- the
    # coverage denominator depends on it
    assert len(out) == 4


def test_search_export_falls_back_to_cell_type(tmp_path):
    from fruresidual.data.flywire_client import load_codex_gene_matches

    p = _write(tmp_path, [("T2",), ("T4",)], ["cell_type"])
    out = load_codex_gene_matches(p, _curated())
    assert out.attrs["matched_on"] == "cell_type"
    assert set(out.loc[out["fru_dsx"].notna(), "cell_type"]) == {"T2", "T4"}


def test_root_ids_are_not_corrupted_by_float_precision(tmp_path):
    """FlyWire root_ids are 64-bit. Read as float64 they lose their last
    digits and silently match nothing, so the whole export would look
    empty rather than erroring."""
    from fruresidual.data.flywire_client import load_codex_gene_matches

    p = _write(tmp_path, [("720575940600000001",)], ["root_id"])
    out = load_codex_gene_matches(p, _curated())
    assert out.attrs["n_matched"] == 1


def test_export_from_the_wrong_dataset_fails_with_a_useful_message(tmp_path):
    """Codex hosts MCNS, BANC and MANC alongside FAFB. Downloading the
    wrong one is an easy mistake and would otherwise show up as a
    mysterious 0% rather than as an error."""
    from fruresidual.data.flywire_client import load_codex_gene_matches

    p = _write(tmp_path, [("999999999999999999",), ("888888888888888888",)], ["root_id"])
    with pytest.raises(ValueError) as exc:
        load_codex_gene_matches(p, _curated())
    msg = str(exc.value)
    assert "different dataset" in msg
    assert "FAFB v783" in msg
