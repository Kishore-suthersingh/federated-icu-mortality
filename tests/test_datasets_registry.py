"""Registry-level invariants that must hold for every clinical dataset.

These are cheap to state and expensive to discover late: an identifier left in
the feature matrix, a target column that survives into X, or a positive class
that silently inverts all produce plausible-looking results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm.config import DataConfig  # noqa: E402
from fedlgbm.datasets import REGISTRY, get_spec  # noqa: E402

CLINICAL = ["wids", "diabetes"]


def test_registry_contains_expected_domains() -> None:
    assert {"ciciot", *CLINICAL} <= set(REGISTRY)


def test_unknown_domain_raises_with_a_helpful_message() -> None:
    with pytest.raises(KeyError, match="known:"):
        get_spec("does_not_exist")


def test_ciciot_spec_routes_to_the_legacy_pipeline() -> None:
    """The manuscript under review depends on the original code path."""
    assert get_spec("ciciot").uses_legacy_pipeline is True


@pytest.mark.parametrize("name", CLINICAL)
def test_clinical_specs_use_the_generic_pipeline(name: str) -> None:
    assert get_spec(name).uses_legacy_pipeline is False


@pytest.mark.parametrize("name", CLINICAL)
def test_identifiers_and_target_are_excluded(name: str) -> None:
    """Ids and the label must never reach X, for any spec."""
    spec = get_spec(name)
    cfg = DataConfig()
    cfg.domain = name

    columns = {"junk_feature": [0.0]}
    for col in filter(None, [spec.group_column, spec.site_column]):
        columns[col] = [1]
    for col in spec.baseline_score_columns:
        columns[col] = [0.5]
    target = "hospital_death" if name == "wids" else "readmitted"
    columns[target] = [0 if name == "wids" else "NO"]
    df = pd.DataFrame(columns)

    excluded = spec.excluded_columns(df, cfg)
    assert target in excluded
    if spec.group_column in df.columns:
        assert spec.group_column in excluded
    for col in spec.baseline_score_columns:
        assert col in excluded, f"{col} is a model prediction of the target"
    assert "junk_feature" not in excluded


@pytest.mark.parametrize("name", CLINICAL)
def test_binary_specs_put_the_adverse_outcome_at_index_one(name: str) -> None:
    """Index 1 must be the adverse class.

    Several consumers assume it — isolation_forest fits on the negative class,
    and every probability metric scores column 1. Deriving the order from
    sorted(unique) would put "died" before "survived" and silently invert
    every reported number.
    """
    spec = get_spec(name)
    cfg = DataConfig()
    cfg.domain = name
    if name == "wids":
        raw = pd.Series([0, 1, 0, 1])
        expected_positive = 2
    else:
        raw = pd.Series(["NO", "<30", ">30", "<30"])
        expected_positive = 2

    y, class_names = spec.derive_labels(raw, cfg)
    assert len(class_names) == 2
    assert int(np.sum(y == 1)) == expected_positive
    assert any(k in class_names[1] for k in ("died", "readmitted_within"))


def test_wids_rejects_a_non_binary_target() -> None:
    spec = get_spec("wids")
    cfg = DataConfig()
    cfg.domain = "wids"
    with pytest.raises(ValueError, match="0/1 target"):
        spec.derive_labels(pd.Series([0, 1, 2]), cfg)


def test_wids_rejects_a_target_with_missing_values() -> None:
    spec = get_spec("wids")
    cfg = DataConfig()
    cfg.domain = "wids"
    with pytest.raises(ValueError, match="missing values"):
        spec.derive_labels(pd.Series([0.0, 1.0, np.nan]), cfg)


@pytest.mark.parametrize("name", CLINICAL)
def test_clinical_specs_do_not_cap_per_class(name: str) -> None:
    """Capping the majority class changes the base rate.

    AUPRC and calibration are both defined against prevalence, so a cap would
    silently make them describe a population that does not exist.
    """
    assert get_spec(name).default_max_per_class is None


def test_diabetes_defaults_to_group_splitting() -> None:
    """Patients genuinely recur here: 71,518 patients over 101,766 encounters,
    and a random split places 9,414 of them in more than one partition."""
    spec = get_spec("diabetes")
    assert spec.default_split_mode == "group"
    assert spec.group_column == "patient_nbr"


def test_wids_declares_a_site_column_and_diabetes_does_not() -> None:
    """WiDS federates on real hospitals; Diabetes-130 ships no site id."""
    assert get_spec("wids").site_column == "hospital_id"
    assert get_spec("diabetes").site_column is None
