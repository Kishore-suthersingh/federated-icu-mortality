"""CIC-IoT-2023 spec — a thin wrapper over the original pipeline.

This exists so the domain registry is complete, not so the CIC-IoT path can be
restructured. The manuscript built on this pipeline is under review, and
`tests/test_ciciot_regression.py` pins its output byte-for-byte, so the spec
delegates entirely: `uses_legacy_pipeline` routes `prepare()` to the original
body rather than through the split-then-fit clinical path.

The two pipelines differ in a way that matters and is deliberate. The original
cleans before splitting, so imputation and column-drop decisions see test rows.
That is near-harmless on flow records — few gaps, a large sample — and fixing it
would change published numbers. The clinical path does not inherit the flaw.
"""

from __future__ import annotations

from ..config import DataConfig
from . import DatasetSpec, register


def _unsupported(*_args, **_kwargs):
    raise RuntimeError(
        "The CIC-IoT path runs through data._prepare_legacy, not the generic "
        "spec callbacks. This spec exists for registry completeness only."
    )


SPEC = register(DatasetSpec(
    name="ciciot",
    load=_unsupported,
    label_column=_unsupported,
    derive_labels=_unsupported,
    excluded_columns=_unsupported,
    categorical_columns=(),
    group_column=None,
    site_column=None,
    default_split_mode="random",
    default_max_per_class=50_000,
    uses_legacy_pipeline=True,
))
