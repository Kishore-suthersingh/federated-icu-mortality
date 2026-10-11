"""Diabetes 130-US Hospitals — 30-day readmission, 1999-2008.

101,766 encounters. This is the cross-dataset validation partner to WiDS, and it
is chosen precisely because it fails differently: it has no hospital identifier,
so the federation must be simulated, but `patient_nbr` genuinely recurs (roughly
71,500 patients over 101,766 encounters). Group-aware splitting is therefore
load-bearing here — a random split would put the same patient in train and test
— which is what earns the machinery that WiDS does not exercise.

Target: `readmitted` is one of {"NO", "<30", ">30"}. The clinically meaningful
question is early readmission, so it is binarised as "<30" versus everything
else. Collapsing ">30" into the negative class is the standard framing in this
literature and is stated in the paper rather than left implicit.

Missingness is encoded as the literal string "?" rather than as empty cells,
which pandas reads as a value; it is converted on load so the preprocessor's
missing-value handling applies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import DataConfig
from . import DatasetSpec, register

logger = logging.getLogger(__name__)

TARGET = "readmitted"
EARLY = "<30"

IDENTIFIERS = ("encounter_id", "patient_nbr")

# Recorded after or as part of the discharge decision, so predictive of the
# outcome for reasons that would not be available at prediction time.
POST_OUTCOME = ("discharge_disposition_id",)

CATEGORICAL = (
    "race", "gender", "age", "weight", "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed", "admission_type_id", "admission_source_id",
)


def load(cfg: DataConfig) -> pd.DataFrame:
    files = sorted(cfg.data_dir.rglob(cfg.file_glob))
    files = [f for f in files if "ids_mapping" not in f.name.lower()]
    if not files:
        raise FileNotFoundError(
            f"No files matching {cfg.file_glob!r} under {cfg.data_dir}. "
            "Expected diabetic_data.csv."
        )
    df = pd.read_csv(files[0], low_memory=False)
    # "?" is this dataset's missing marker.
    df = df.replace("?", np.nan)
    if cfg.max_rows is not None and len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=0).reset_index(drop=True)
    logger.info("Loaded %d rows x %d columns", len(df), df.shape[1])
    return df


def label_column(df: pd.DataFrame, cfg: DataConfig) -> str:
    col = cfg.label_column or TARGET
    if col not in df.columns:
        raise KeyError(f"Target column {col!r} not present")
    return col


def derive_labels(raw: pd.Series, cfg: DataConfig) -> tuple:
    """Binarise to early readmission (<30 days) versus not."""
    y = (raw.astype(str).str.strip() == EARLY).to_numpy(dtype=np.int64)
    return y, ["not_readmitted_early", "readmitted_within_30d"]


def excluded_columns(df: pd.DataFrame, cfg: DataConfig) -> set:
    excluded = set(IDENTIFIERS) | set(POST_OUTCOME)
    excluded.add(label_column(df, cfg))
    return {c for c in excluded if c in df.columns}


SPEC = register(DatasetSpec(
    name="diabetes",
    load=load,
    label_column=label_column,
    derive_labels=derive_labels,
    excluded_columns=excluded_columns,
    categorical_columns=CATEGORICAL,
    group_column="patient_nbr",
    site_column=None,        # no hospital identifier in the public release
    baseline_score_columns=(),
    default_split_mode="group",   # patients genuinely recur
    default_max_per_class=None,
))
