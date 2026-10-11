"""WiDS Datathon 2020 / GOSSIS — ICU mortality across 147 real hospitals.

91,713 ICU stays, 186 columns, target `hospital_death` at 8.63% prevalence.
The reason this dataset was chosen over a simulated multi-site benchmark is
`hospital_id`: the federation is observed rather than sampled, so the paper
reports measured cross-site heterogeneity instead of defending a Dirichlet
concentration parameter.

Facts established by inspecting the data, each of which drives a decision here:

* `patient_id` is row-unique (91,713 distinct over 91,713 rows), so group-aware
  splitting is a no-op on this dataset. It is still wired up because
  Diabetes-130 needs it, but nothing here depends on it.
* Mortality by hospital ranges 0.000 to 0.500, and ten hospitals have never
  recorded a death. Those become single-class federated clients — a real
  pathology that simulated partitions rarely produce.
* `apache_2_diagnosis` (44 levels) and `apache_3j_diagnosis` (399) are diagnosis
  codes stored as floats. Treating them as numeric implies an ordering that does
  not exist, so they are declared categorical.
* The `apache_4a_*_death_prob` columns are the APACHE IVa model's own risk
  prediction. See the note in this package's __init__ for why they are held out
  of X and used as a baseline instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import DataConfig
from . import DatasetSpec, register

logger = logging.getLogger(__name__)

TARGET = "hospital_death"

IDENTIFIERS = ("encounter_id", "patient_id", "hospital_id", "icu_id")

# APACHE IVa's predicted probabilities for this exact outcome.
BASELINE_SCORES = ("apache_4a_hospital_death_prob", "apache_4a_icu_death_prob")

# Constant in the public release, and a documented artefact rather than a signal.
KNOWN_DEGENERATE = ("readmission_status",)

CATEGORICAL = (
    "ethnicity", "gender", "hospital_admit_source", "icu_admit_source",
    "icu_stay_type", "icu_type", "apache_3j_bodysystem", "apache_2_bodysystem",
    "apache_2_diagnosis", "apache_3j_diagnosis",
)


def load(cfg: DataConfig) -> pd.DataFrame:
    files = sorted(cfg.data_dir.rglob(cfg.file_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching {cfg.file_glob!r} under {cfg.data_dir}. "
            "Expected the WiDS Datathon 2020 training_v2.csv."
        )
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if cfg.max_rows is not None and len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=0).reset_index(drop=True)
    logger.info("Loaded %d rows x %d columns from %d file(s)",
                len(df), df.shape[1], len(files))
    return df


def label_column(df: pd.DataFrame, cfg: DataConfig) -> str:
    col = cfg.label_column or TARGET
    if col not in df.columns:
        raise KeyError(f"Target column {col!r} not present")
    return col


def derive_labels(raw: pd.Series, cfg: DataConfig) -> tuple:
    """0/1 target -> indices with class 1 as death.

    The ordering is fixed explicitly rather than taken from sorted(unique),
    because several consumers treat index 1 as the adverse outcome and a
    silently reversed mapping would invert every reported metric.
    """
    y = pd.to_numeric(raw, errors="coerce")
    if y.isna().any():
        raise ValueError(f"{y.isna().sum()} missing values in the target column")
    values = set(np.unique(y).tolist())
    if not values <= {0.0, 1.0}:
        raise ValueError(f"Expected a 0/1 target, found values {sorted(values)}")
    return y.to_numpy(dtype=np.int64), ["survived", "died"]


def excluded_columns(df: pd.DataFrame, cfg: DataConfig) -> set:
    excluded = set(IDENTIFIERS) | set(BASELINE_SCORES) | set(KNOWN_DEGENERATE)
    excluded.add(label_column(df, cfg))
    return {c for c in excluded if c in df.columns}


SPEC = register(DatasetSpec(
    name="wids",
    load=load,
    label_column=label_column,
    derive_labels=derive_labels,
    excluded_columns=excluded_columns,
    categorical_columns=CATEGORICAL,
    group_column="patient_id",
    site_column="hospital_id",
    baseline_score_columns=BASELINE_SCORES,
    default_split_mode="random",
    default_max_per_class=None,   # capping would destroy the 8.63% base rate
))
