"""CIC-IoT-2023 loading, label harmonisation, cleaning and splitting.

The published dataset ships as ~170 CSV part files sharing one schema: 46
numeric flow features plus a `label` column carrying 34 fine-grained class
names. Column naming varies slightly between mirrors (spaces, capitalisation,
`Label` vs `label`), so everything here is deliberately schema-tolerant rather
than hard-coding positions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import DataConfig

logger = logging.getLogger(__name__)

BENIGN = "Benign"

# Fine label -> coarse attack family. Matched by prefix against the normalised
# label, longest prefix first, so "DDoS-" wins over "DoS-" style collisions.
CATEGORY_PREFIXES: list[tuple[str, str]] = [
    ("ddos", "DDoS"),
    ("dos", "DoS"),
    ("mirai", "Mirai"),
    ("recon", "Recon"),
    ("vulnerabilityscan", "Recon"),
    ("dns_spoofing", "Spoofing"),
    ("mitm", "Spoofing"),
    ("browserhijacking", "Web"),
    ("backdoor", "Web"),
    ("xss", "Web"),
    ("uploading", "Web"),
    ("sqlinjection", "Web"),
    ("commandinjection", "Web"),
    ("dictionarybruteforce", "BruteForce"),
    ("bruteforce", "BruteForce"),
    ("benign", BENIGN),
    ("benigntraffic", BENIGN),
]


def normalise_label(raw: str) -> str:
    """Strip whitespace/case noise so mirrors with different formatting agree."""
    return re.sub(r"[\s]+", "", str(raw)).strip().lower()


def to_category(raw: str) -> str:
    """Map a fine-grained CIC-IoT-2023 label to one of 8 coarse families."""
    norm = normalise_label(raw)
    # Longest prefix first so "ddos" is tested before "dos".
    for prefix, category in sorted(CATEGORY_PREFIXES, key=lambda p: -len(p[0])):
        if norm.startswith(prefix):
            return category
    logger.warning("Unrecognised label %r — bucketing as Unknown", raw)
    return "Unknown"


def to_binary(raw: str) -> int:
    """1 = malicious, 0 = benign."""
    return 0 if to_category(raw) == BENIGN else 1


def find_label_column(df: pd.DataFrame) -> str:
    """Locate the label column regardless of casing/spacing across mirrors."""
    for col in df.columns:
        if str(col).strip().lower() in {"label", "labels", "class", "attack"}:
            return col
    raise ValueError(
        f"No label column found. Available columns: {list(df.columns)[:50]}"
    )


def load_raw(cfg: DataConfig) -> pd.DataFrame:
    """Read CSV parts into one dataframe, respecting the row/file caps.

    Reads file-by-file and stops as soon as `max_rows` is satisfied, so local
    iteration never has to touch the full ~13 GB.
    """
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Locally: run `python scripts/make_subsample.py` first.\n"
            "On Kaggle: attach the CIC-IoT-2023 dataset and point "
            "DataConfig.data_dir at /kaggle/input/<dataset-slug>."
        )

    files = sorted(data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {data_dir}")
    if cfg.max_files is not None:
        files = files[: cfg.max_files]

    logger.info("Reading %d CSV file(s) from %s", len(files), data_dir)

    frames: list[pd.DataFrame] = []
    total = 0
    for path in files:
        remaining = None if cfg.max_rows is None else cfg.max_rows - total
        if remaining is not None and remaining <= 0:
            break
        frame = pd.read_csv(path, nrows=remaining, low_memory=False)
        frames.append(frame)
        total += len(frame)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows x %d columns", len(df), df.shape[1])
    return df


def clean(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, coerce to numeric, and handle missing/infinite values.

    Returns the cleaned frame plus the statistics the manuscript's Table 2
    is supposed to report (and currently does not).
    """
    stats: dict = {"rows_initial": int(len(df))}

    # Drop exact duplicates — the manuscript claims this but never quantifies it.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicates_removed"] = int(before - len(df))

    labels = df[label_col].copy()
    features = df.drop(columns=[label_col])

    # Any non-numeric feature column is unusable for the MLP; record and drop.
    non_numeric = [
        c for c in features.columns
        if not pd.api.types.is_numeric_dtype(features[c])
    ]
    coerced = features.drop(columns=non_numeric) if non_numeric else features
    stats["non_numeric_columns_dropped"] = non_numeric

    # Flow-rate features divide by duration and produce inf on zero-length flows.
    n_inf = int(np.isinf(coerced.to_numpy(dtype=np.float64, na_value=np.nan)).sum())
    coerced = coerced.replace([np.inf, -np.inf], np.nan)
    stats["infinite_values_replaced"] = n_inf

    n_missing = int(coerced.isna().sum().sum())
    stats["missing_values_total"] = n_missing
    stats["missing_by_column"] = {
        str(c): int(v) for c, v in coerced.isna().sum().items() if v > 0
    }

    # Columns that are almost entirely missing can't be interpolated meaningfully.
    mostly_missing = [c for c in coerced.columns if coerced[c].isna().mean() > 0.5]
    if mostly_missing:
        coerced = coerced.drop(columns=mostly_missing)
    stats["columns_dropped_mostly_missing"] = mostly_missing

    # Median imputation. Interpolation (as the manuscript describes) assumes row
    # ordering is meaningful; these are independent flow records concatenated
    # across capture files, so ordering carries no signal and median is honest.
    coerced = coerced.fillna(coerced.median(numeric_only=True))
    coerced = coerced.fillna(0.0)
    stats["imputation_strategy"] = "column median (flow records are unordered)"

    # Constant columns carry no information and destabilise scaling.
    constant = [c for c in coerced.columns if coerced[c].nunique(dropna=False) <= 1]
    if constant:
        coerced = coerced.drop(columns=constant)
    stats["constant_columns_dropped"] = constant

    coerced[label_col] = labels.values
    stats["rows_final"] = int(len(coerced))
    stats["n_features"] = int(coerced.shape[1] - 1)

    return coerced, stats


def cap_per_class(
    df: pd.DataFrame, label_col: str, cap: int | None, seed: int
) -> pd.DataFrame:
    """Downsample over-represented classes.

    CIC-IoT-2023 is dominated by DDoS floods (millions of rows) while some Web
    attacks have only a few hundred. Without capping, a model that predicts
    "DDoS" unconditionally scores >90% accuracy, which is exactly the kind of
    hollow number the reviewers objected to.
    """
    if cap is None:
        return df
    return (
        df.groupby(label_col, group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.sample(min(len(g), cap), random_state=seed))
        .reset_index(drop=True)
    )


class Dataset:
    """Prepared, split, scaled arrays plus the metadata the paper needs."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        class_names: list[str],
        clean_stats: dict,
    ) -> None:
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.feature_names = feature_names
        self.class_names = class_names
        self.clean_stats = clean_stats

        # Set by the multi-site path; None under the legacy CIC-IoT pipeline.
        self.train_sites: np.ndarray | None = None
        self.test_sites: np.ndarray | None = None
        self.categorical_indices: list[int] | None = None
        # Incumbent-model scores held out of X, kept for comparison.
        self.baseline_scores: dict = {}

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "n_train": int(len(self.y_train)),
            "n_val": int(len(self.y_val)),
            "n_test": int(len(self.y_test)),
            "train_class_counts": _counts(self.y_train, self.class_names),
            "test_class_counts": _counts(self.y_test, self.class_names),
            "clean_stats": self.clean_stats,
        }


def _counts(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {class_names[int(v)]: int(c) for v, c in zip(values, counts)}


def prepare(cfg: DataConfig, seed: int = 42) -> Dataset:
    """Build a Dataset for the configured domain.

    Two pipelines live behind this. `ciciot` runs the original clean-then-split
    body unchanged, because the manuscript under review depends on its exact
    output. Every other domain runs the split-then-fit path in `_prepare_generic`,
    which learns imputation, encoding and scaling from training rows only.
    """
    from .datasets import get_spec  # local import: registry imports config

    spec = get_spec(getattr(cfg, "domain", "ciciot"))
    if spec.uses_legacy_pipeline:
        return _prepare_legacy(cfg, seed)
    return _prepare_generic(cfg, spec, seed)


def _prepare_generic(cfg: DataConfig, spec, seed: int = 42) -> Dataset:
    """Split first, then fit every transform on the training rows alone."""
    from .preprocess import Preprocessor
    from .splits import make_split

    df = spec.load(cfg)
    label_col = spec.label_column(df, cfg)
    y, class_names = spec.derive_labels(df[label_col], cfg)

    group_col = cfg.group_column or spec.group_column
    site_col = cfg.site_column or spec.site_column
    groups = df[group_col].to_numpy() if group_col in df.columns else None
    sites = df[site_col].to_numpy() if site_col and site_col in df.columns else None

    # Baseline scores are held out of X but retained for comparison: an
    # incumbent clinical score is a comparator, and leaving it among the
    # features would let every model read the outcome off one column.
    baselines = {
        c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)
        for c in spec.baseline_score_columns if c in df.columns
    }

    X_df = df.drop(columns=list(spec.excluded_columns(df, cfg)))

    split = make_split(
        y,
        mode=cfg.split_mode,
        groups=groups,
        sites=sites,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=seed,
        min_rows_per_site=cfg.min_rows_per_site,
    )

    pre = Preprocessor(
        categorical_columns=[c for c in spec.categorical_columns if c in X_df.columns],
        missing_drop_threshold=cfg.missing_drop_threshold,
        add_missing_indicators=cfg.add_missing_indicators,
        rare_level_min_count=cfg.rare_level_min_count,
        max_onehot_cardinality=cfg.categorical_max_cardinality,
        scaler=cfg.scaler,
    )
    train_df = X_df.iloc[split.train_idx]
    pre.fit(train_df)

    X_train = pre.transform(train_df)
    X_val = pre.transform(X_df.iloc[split.val_idx])
    X_test = pre.transform(X_df.iloc[split.test_idx])

    clean_stats = dict(pre.stats_)
    clean_stats["split"] = split.diagnostics
    clean_stats["rows_initial"] = int(len(df))
    clean_stats["rows_final"] = int(len(df))
    clean_stats["excluded_columns"] = sorted(spec.excluded_columns(df, cfg))

    ds = Dataset(
        X_train, y[split.train_idx],
        X_val, y[split.val_idx],
        X_test, y[split.test_idx],
        pre.feature_names_, class_names, clean_stats,
    )
    ds.train_sites = sites[split.train_idx] if sites is not None else None
    ds.test_sites = sites[split.test_idx] if sites is not None else None
    ds.categorical_indices = pre.categorical_indices_
    ds.baseline_scores = {
        name: {
            "val": values[split.val_idx],
            "test": values[split.test_idx],
        }
        for name, values in baselines.items()
    }

    logger.info(
        "Prepared [%s]: %d train / %d val / %d test, %d features, %d classes",
        spec.name, len(ds.y_train), len(ds.y_val), len(ds.y_test),
        ds.n_features, len(class_names),
    )
    return ds


def _prepare_legacy(cfg: DataConfig, seed: int = 42) -> Dataset:
    """Original CIC-IoT pipeline: load -> clean -> label -> cap -> split -> scale.

    Pinned by tests/test_ciciot_regression.py. Do not modify without deciding to
    re-run and re-report the IoT results.
    """
    df = load_raw(cfg)
    label_col = find_label_column(df)
    df, clean_stats = clean(df, label_col)

    df = cap_per_class(df, label_col, cfg.max_per_class, seed)

    raw_labels = df[label_col].astype(str)
    if cfg.label_mode == "binary":
        y_str = raw_labels.map(lambda s: BENIGN if to_binary(s) == 0 else "Malicious")
    elif cfg.label_mode == "category":
        y_str = raw_labels.map(to_category)
    else:
        y_str = raw_labels

    class_names = sorted(y_str.unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).to_numpy(dtype=np.int64)

    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    feature_names = [str(c) for c in df.drop(columns=[label_col]).columns]

    # Stratify so rare classes survive both splits.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=seed, stratify=y
    )
    val_fraction = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=seed, stratify=y_temp
    )

    # Fit the scaler on train only — fitting on the full set leaks test
    # statistics into training, a flaw reviewers do look for.
    if cfg.scaler != "none":
        scaler = StandardScaler() if cfg.scaler == "standard" else MinMaxScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(
        "Prepared: %d train / %d val / %d test, %d features, %d classes",
        len(y_train), len(y_val), len(y_test), X_train.shape[1], len(class_names),
    )

    return Dataset(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, class_names, clean_stats,
    )
