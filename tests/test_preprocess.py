"""Leakage invariants for the clinical preprocessing path.

Every test here targets a way test-set information could reach training. These
are the properties a clinical reviewer checks first, and none of them were
covered before the port: the CIC-IoT path cleans before splitting, so it cannot
satisfy them and is deliberately not exercised here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm.preprocess import MISSING_LEVEL, RARE_LEVEL, Preprocessor  # noqa: E402


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(60, 15, n),
        "lactate": rng.normal(2, 1, n),
        "sparse_lab": rng.normal(5, 2, n),
        "constant": np.ones(n),
        "gender": rng.choice(["M", "F"], n),
        "icu_type": rng.choice(["MICU", "SICU", "CCU"], n),
    })
    # A realistically sparse lab column, and a few scattered gaps elsewhere.
    df.loc[df.sample(frac=0.7, random_state=seed).index, "sparse_lab"] = np.nan
    df.loc[df.sample(frac=0.1, random_state=seed + 1).index, "lactate"] = np.nan
    return df


def test_medians_come_from_train_only() -> None:
    """Wild test values must not move the training transform at all."""
    train = _frame(300, seed=1)
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    baseline = pre.transform(train)

    poisoned = _frame(200, seed=2)
    poisoned["lactate"] = 999_999.0
    poisoned["age"] = -999_999.0
    pre.transform(poisoned)

    assert np.array_equal(baseline, pre.transform(train)), (
        "transforming other data changed the training transform"
    )


def test_scaler_statistics_are_train_only() -> None:
    train = _frame(300, seed=3)
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    out = pre.transform(train)
    numeric = out[:, : len(pre.numeric_columns_)]
    assert np.allclose(numeric.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(numeric.std(axis=0), 1.0, atol=1e-3)


def test_unseen_categorical_level_maps_to_other() -> None:
    """A level that appears only at test time must not create a column."""
    train = _frame(300, seed=4)
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    width = pre.transform(train).shape[1]

    test = _frame(100, seed=5)
    test["icu_type"] = "NEURO_ICU"  # never seen in training
    out = pre.transform(test)

    assert out.shape[1] == width
    other_col = pre.feature_names_.index(f"icu_type={RARE_LEVEL}")
    assert out[:, other_col].sum() == len(test)


def test_missing_categorical_becomes_an_explicit_level() -> None:
    train = _frame(300, seed=6)
    train.loc[train.sample(frac=0.2, random_state=1).index, "gender"] = None
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    assert f"gender={MISSING_LEVEL}" in pre.feature_names_


def test_missing_indicators_precede_imputation() -> None:
    """The indicator must record the gap that imputation then fills."""
    train = _frame(400, seed=7)
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    assert "sparse_lab__isna" in pre.feature_names_

    out = pre.transform(train)
    col = pre.feature_names_.index("sparse_lab__isna")
    assert out[:, col].sum() == train["sparse_lab"].isna().sum()


def test_high_missingness_column_survives_default_threshold() -> None:
    """70% missing must be kept: at 0.5 the clinical lab panel disappears."""
    train = _frame(300, seed=8)
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    assert "sparse_lab" in pre.numeric_columns_
    assert "sparse_lab" not in pre.dropped_columns_


def test_constant_column_is_dropped() -> None:
    train = _frame(300, seed=9)
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    assert "constant" in pre.constant_columns_
    assert "constant" not in pre.feature_names_


def test_transform_is_stable_across_row_order() -> None:
    """Row order must not change any learned parameter."""
    train = _frame(300, seed=10)
    a = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    shuffled = train.sample(frac=1.0, random_state=99)
    b = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(shuffled)
    assert a.feature_names_ == b.feature_names_
    for col, median in a.medians_.items():
        assert median == pytest.approx(b.medians_[col])


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        Preprocessor().transform(_frame(10))


def test_column_missing_at_transform_time_is_tolerated() -> None:
    """A site that never records a lab must not crash the pipeline."""
    train = _frame(300, seed=11)
    pre = Preprocessor(categorical_columns=["gender", "icu_type"]).fit(train)
    partial = _frame(50, seed=12).drop(columns=["sparse_lab"])
    out = pre.transform(partial)
    assert out.shape == (50, len(pre.feature_names_))
