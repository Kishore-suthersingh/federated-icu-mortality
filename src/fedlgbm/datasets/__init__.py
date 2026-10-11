"""Per-dataset specifications.

Adding a domain means writing a `DatasetSpec` and registering it, not editing
the pipeline. That matters here for a specific reason: the CIC-IoT manuscript is
under review and its numbers must stay reproducible, so the original path is
reached through a spec of its own and is never restructured.

A spec answers the questions the pipeline cannot infer safely:

* where the rows are and what the target column is called
* how raw label values become class indices, and which class is "positive"
* which columns must never become features

That last one carries the most weight. Identifiers leak trivially, and in the
clinical data there is a subtler case: `apache_4a_hospital_death_prob` is the
APACHE IVa model's predicted probability of the exact target. It is legitimately
available at prediction time, so it is not temporal leakage, but leaving it in
the feature matrix lets every model read the answer off one column. Accuracy
then saturates and federated and centralized results become indistinguishable —
which destroys the one measurement the study exists to make. It is excluded here
and used as a comparator baseline instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..config import DataConfig


@dataclass(frozen=True)
class DatasetSpec:
    """Everything domain-specific about one dataset."""

    name: str

    # data_dir + file_glob -> raw frame
    load: Callable[[DataConfig], pd.DataFrame]

    # frame -> name of the target column
    label_column: Callable[[pd.DataFrame, DataConfig], str]

    # (raw label series, config) -> (class index array, ordered class names).
    # Implementations must order names so that for a binary task index 1 is the
    # adverse/positive outcome; several downstream consumers rely on it.
    derive_labels: Callable[[pd.Series, DataConfig], tuple]

    # Columns that must never reach X: ids, the target, and anything encoding
    # the outcome. Returned as a set so callers can union freely.
    excluded_columns: Callable[[pd.DataFrame, DataConfig], set]

    # Columns to force to categorical even when stored as numbers (diagnosis
    # codes are floats but ordering them is meaningless).
    categorical_columns: tuple = ()

    group_column: str | None = None   # patient
    site_column: str | None = None    # hospital

    # Columns held out of X but kept aside as ready-made comparator scores,
    # e.g. an incumbent clinical risk model's predicted probability.
    baseline_score_columns: tuple = ()

    default_split_mode: str = "random"
    default_max_per_class: int | None = None
    uses_legacy_pipeline: bool = False


REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "none registered"
        raise KeyError(f"Unknown dataset domain {name!r} (known: {known})")
    return REGISTRY[name]


from . import ciciot, diabetes, wids  # noqa: E402,F401  (registration side-effect)

__all__ = ["DatasetSpec", "REGISTRY", "get_spec", "register"]
