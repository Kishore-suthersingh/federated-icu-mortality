"""Fit-on-train preprocessing for tabular clinical data.

The CIC-IoT path cleans the full frame before splitting (`data.clean`), which is
harmless there — few missing values, a huge sample, and imputation that barely
moves. It is not harmless on ICU data, where 74 of 186 columns are more than
half missing and are median-imputed. Computing those medians over train+val+test
lets test statistics reach training, which is the first thing a clinical
reviewer checks for.

So this module exists rather than a patch to `clean()`: every decision here is
made from training rows only and then applied unchanged to validation and test.
`Preprocessor` is deliberately sklearn-shaped (`fit`/`transform`) so the
train-only property is structural rather than a convention someone has to
remember.

Design notes specific to this data:

* Missing-value threshold defaults to 0.95, not 0.5. The APACHE lab panel is
  20-90% missing and is where the predictive signal lives; dropping at 0.5 would
  discard 40% of the dataset.
* Missingness is itself informative. A lab that was never ordered says something
  about the clinician's judgement, so we add explicit indicator columns rather
  than imputing the trace away.
* Categoricals get an explicit ``__MISSING__`` level for the same reason: "not
  recorded" is a category, not a value to be guessed at.
* One-hot columns are left unscaled. Standardising them makes them dense and
  non-zero, which costs the MLP the sparsity it would otherwise exploit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

MISSING_LEVEL = "__MISSING__"
RARE_LEVEL = "__OTHER__"


class Preprocessor:
    """Learns imputation, encoding and scaling from training rows only.

    Parameters mirror the fields of :class:`~fedlgbm.config.DataConfig` that
    control preprocessing, passed explicitly so the class stays usable outside
    the experiment harness (notably in tests).
    """

    def __init__(
        self,
        *,
        categorical_columns: list[str] | None = None,
        missing_drop_threshold: float = 0.95,
        add_missing_indicators: bool = True,
        indicator_min_rate: float = 0.005,
        indicator_max_rate: float = 0.99,
        rare_level_min_count: int = 50,
        max_onehot_cardinality: int = 20,
        scaler: str = "standard",
    ) -> None:
        self.categorical_columns = list(categorical_columns or [])
        self.missing_drop_threshold = missing_drop_threshold
        self.add_missing_indicators = add_missing_indicators
        self.indicator_min_rate = indicator_min_rate
        self.indicator_max_rate = indicator_max_rate
        self.rare_level_min_count = rare_level_min_count
        self.max_onehot_cardinality = max_onehot_cardinality
        self.scaler_kind = scaler

        # Learned state. Everything below is derived from training rows only.
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.levels_: dict[str, list[str]] = {}
        self.ordinal_columns_: list[str] = []
        self.indicator_columns_: list[str] = []
        self.dropped_columns_: list[str] = []
        self.constant_columns_: list[str] = []
        self.feature_names_: list[str] = []
        self.categorical_indices_: list[int] = []
        self._scaler = None
        self._n_numeric = 0
        self.stats_: dict = {}

    # -- fit ---------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Learn every transform parameter from `df`, which must be train only."""
        frame = df.copy()
        n_rows = len(frame)

        declared = [c for c in self.categorical_columns if c in frame.columns]
        inferred = [
            c for c in frame.columns
            if c not in declared and frame[c].dtype == object
        ]
        categorical = declared + inferred
        numeric = [c for c in frame.columns if c not in categorical]

        # Infinities are missing values that happen to survive dtype checks.
        for col in numeric:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)

        missing_rate = frame.isna().mean()
        self.dropped_columns_ = sorted(
            c for c in frame.columns if missing_rate[c] > self.missing_drop_threshold
        )
        numeric = [c for c in numeric if c not in self.dropped_columns_]
        categorical = [c for c in categorical if c not in self.dropped_columns_]

        # A column that is constant in training carries no signal and would give
        # a zero-variance scaler column.
        self.constant_columns_ = sorted(
            c for c in numeric if frame[c].nunique(dropna=True) <= 1
        )
        numeric = [c for c in numeric if c not in self.constant_columns_]

        self.medians_ = {c: float(frame[c].median()) for c in numeric}
        # median() is NaN when a column is entirely missing in train; such a
        # column cannot inform anything, so impute zero and let it be constant.
        self.medians_ = {
            c: (0.0 if not np.isfinite(v) else v) for c, v in self.medians_.items()
        }

        if self.add_missing_indicators:
            self.indicator_columns_ = [
                c for c in numeric
                if self.indicator_min_rate < missing_rate[c] < self.indicator_max_rate
            ]
        else:
            self.indicator_columns_ = []

        self.levels_ = {}
        self.ordinal_columns_ = []
        for col in categorical:
            counts = frame[col].astype("string").fillna(MISSING_LEVEL).value_counts()
            kept = [str(level) for level, n in counts.items()
                    if n >= self.rare_level_min_count]
            if MISSING_LEVEL not in kept and frame[col].isna().any():
                kept.append(MISSING_LEVEL)
            if RARE_LEVEL not in kept:
                kept.append(RARE_LEVEL)
            self.levels_[col] = sorted(kept)
            if len(self.levels_[col]) > self.max_onehot_cardinality:
                self.ordinal_columns_.append(col)

        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical

        # Build the output schema once so transform() is a pure application.
        names = list(numeric)
        names += [f"{c}__isna" for c in self.indicator_columns_]
        self._n_numeric = len(names)
        cat_names: list[str] = []
        cat_indices: list[int] = []
        for col in categorical:
            if col in self.ordinal_columns_:
                cat_indices.append(self._n_numeric + len(cat_names))
                cat_names.append(f"{col}__code")
            else:
                cat_names += [f"{col}={lvl}" for lvl in self.levels_[col]]
        self.feature_names_ = names + cat_names
        self.categorical_indices_ = cat_indices

        imputed, _ = self._numeric_block(frame)
        if self.scaler_kind != "none":
            self._scaler = (
                StandardScaler() if self.scaler_kind == "standard" else MinMaxScaler()
            )
            self._scaler.fit(imputed)

        self.stats_ = {
            "rows_fit": int(n_rows),
            "n_features_out": len(self.feature_names_),
            "numeric_columns": len(numeric),
            "categorical_columns": len(categorical),
            "ordinal_columns": list(self.ordinal_columns_),
            "missing_indicator_columns": len(self.indicator_columns_),
            "columns_dropped_mostly_missing": self.dropped_columns_,
            "constant_columns_dropped": self.constant_columns_,
            "missing_drop_threshold": self.missing_drop_threshold,
            "imputation_strategy": "train-set column median",
            "top_missing_rates": {
                c: round(float(missing_rate[c]), 4)
                for c in missing_rate.sort_values(ascending=False).head(10).index
            },
        }
        logger.info(
            "Preprocessor fit on %d rows: %d numeric + %d categorical -> %d features "
            "(%d dropped >%.0f%% missing, %d constant)",
            n_rows, len(numeric), len(categorical), len(self.feature_names_),
            len(self.dropped_columns_), self.missing_drop_threshold * 100,
            len(self.constant_columns_),
        )
        return self

    # -- transform ---------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the learned transform. Never inspects `df` to make decisions."""
        if not self.feature_names_:
            raise RuntimeError("Preprocessor.transform called before fit")

        frame = df.copy()
        for col in self.numeric_columns_:
            if col not in frame.columns:
                frame[col] = np.nan
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[self.numeric_columns_] = frame[self.numeric_columns_].replace(
            [np.inf, -np.inf], np.nan
        )

        imputed, indicators = self._numeric_block(frame)
        if self._scaler is not None:
            imputed = self._scaler.transform(imputed)
        # Indicators are appended after scaling: they are already 0/1, and
        # standardising them would destroy that and make them dense.
        block = np.hstack([imputed, indicators]) if indicators.size else imputed

        cat_blocks: list[np.ndarray] = []
        for col in self.categorical_columns_:
            values = (
                frame[col].astype("string").fillna(MISSING_LEVEL)
                if col in frame.columns
                else pd.Series([MISSING_LEVEL] * len(frame), index=frame.index)
            )
            levels = self.levels_[col]
            known = set(levels)
            # Any level absent from training — including one that appears only in
            # test — collapses to __OTHER__ rather than creating a new column.
            values = values.map(lambda v: v if v in known else RARE_LEVEL)
            if col in self.ordinal_columns_:
                lookup = {lvl: i for i, lvl in enumerate(levels)}
                codes = values.map(lookup).to_numpy(dtype=np.float32)
                cat_blocks.append(codes.reshape(-1, 1))
            else:
                onehot = np.zeros((len(frame), len(levels)), dtype=np.float32)
                lookup = {lvl: i for i, lvl in enumerate(levels)}
                idx = values.map(lookup).to_numpy()
                onehot[np.arange(len(frame)), idx.astype(int)] = 1.0
                cat_blocks.append(onehot)

        parts = [block.astype(np.float32)]
        if cat_blocks:
            parts.append(np.hstack(cat_blocks))
        out = np.hstack(parts).astype(np.float32)

        if out.shape[1] != len(self.feature_names_):
            raise AssertionError(
                f"transform produced {out.shape[1]} columns but the schema "
                f"learned at fit has {len(self.feature_names_)}"
            )
        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # -- internals ---------------------------------------------------------

    def _numeric_block(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (imputed numeric columns, missingness indicators) separately.

        They are kept apart so the scaler can be fit and applied to the imputed
        values alone. Indicators are computed before imputation, since
        imputation is exactly what erases the information they carry.
        """
        cols = self.numeric_columns_
        n = len(frame)
        if not cols:
            return (np.zeros((n, 0), dtype=np.float32),
                    np.zeros((n, 0), dtype=np.float32))

        sub = frame[cols]
        if self.indicator_columns_:
            indicators = np.hstack([
                sub[c].isna().to_numpy(dtype=np.float32).reshape(-1, 1)
                for c in self.indicator_columns_
            ])
        else:
            indicators = np.zeros((n, 0), dtype=np.float32)
        filled = sub.fillna(value=self.medians_).to_numpy(dtype=np.float32)
        return filled, indicators
