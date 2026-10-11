"""Golden regression test pinning the CIC-IoT-2023 pipeline.

The IoT manuscript is under review and every number in it comes from this
pipeline. The clinical port refactors `prepare()` into a dispatcher over several
domains, and the whole point of that refactor is that the CIC-IoT path comes out
byte-identical. This test is what makes that claim checkable instead of hoped
for: it hashes the full output of `prepare()` on the in-repo synthetic corpus,
across all three label modes.

If one of these fails after a refactor, the refactor changed the old paper's
numbers. That is a defect, not an expected diff — do not regenerate the literals
to make it pass unless you have decided deliberately to re-run and re-report the
IoT results.

Literals generated on commit 9f8462b (2026-08-12), before the clinical port.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import data, smoke_config  # noqa: E402

SYNTHETIC = Path(__file__).resolve().parent.parent / "data" / "synthetic"

# mode -> pinned digests of every array prepare() returns, plus the metadata a
# silent behaviour change would move.
GOLDEN: dict[str, dict] = {
    "binary": {
        "X_train": "c5d4930062061a21",
        "y_train": "a35dd90b9ddc8948",
        "X_val": "26b96caf4dce313e",
        "y_val": "410a33a45b836cfe",
        "X_test": "f934b37ce79c686e",
        "y_test": "41cea899ca47cfe4",
        "class_names": ["Benign", "Malicious"],
    },
    "category": {
        "X_train": "1d127ec714c627ae",
        "y_train": "624bb10988141a52",
        "X_val": "9e6e652aebc76ef4",
        "y_val": "08395aa1a23bf9c4",
        "X_test": "5dac2292d99d69e8",
        "y_test": "901fa6b9b8ade563",
        "class_names": [
            "Benign", "BruteForce", "DDoS", "DoS",
            "Mirai", "Recon", "Spoofing", "Web",
        ],
    },
    "fine": {
        "X_train": "e0c109fe032c0dd6",
        "y_train": "f871cd840adfba7e",
        "X_val": "06ac89f67094061f",
        "y_val": "8bca084a03d96ca9",
        "X_test": "2bec623007bee824",
        "y_test": "7cb66464ea9117ed",
        "class_names": [
            "Backdoor_Malware", "BenignTraffic", "DDoS-ICMP_Flood",
            "DDoS-SYN_Flood", "DDoS-TCP_Flood", "DDoS-UDP_Flood",
            "DNS_Spoofing", "DictionaryBruteForce", "DoS-SYN_Flood",
            "DoS-TCP_Flood", "DoS-UDP_Flood", "MITM-ArpSpoofing",
            "Mirai-greeth_flood", "Mirai-udpplain", "Recon-HostDiscovery",
            "Recon-PortScan", "SqlInjection", "XSS",
        ],
    },
}

EXPECTED_SHAPES = ((3478, 45), (497, 45), (994, 45))
EXPECTED_FEATURE_HASH = "a442d4ab944eccd4"
EXPECTED_CLEAN = {"rows_initial": 5000, "rows_final": 4969, "duplicates_removed": 31}


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _prepare(label_mode: str):
    cfg = smoke_config().data
    cfg.data_dir = SYNTHETIC
    cfg.label_mode = label_mode
    return data.prepare(cfg, seed=42)


@pytest.mark.parametrize("label_mode", sorted(GOLDEN))
def test_ciciot_prepare_is_bit_identical(label_mode: str) -> None:
    """Every array prepare() emits must hash to its pinned value."""
    ds = _prepare(label_mode)
    expected = GOLDEN[label_mode]

    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        assert _digest(getattr(ds, name)) == expected[name], (
            f"{label_mode}/{name} changed — the CIC-IoT pipeline no longer "
            "reproduces the results in the manuscript under review"
        )

    assert ds.class_names == expected["class_names"]


@pytest.mark.parametrize("label_mode", sorted(GOLDEN))
def test_ciciot_split_shapes_are_stable(label_mode: str) -> None:
    """Label mode must not move the split; only the labels differ."""
    ds = _prepare(label_mode)
    shapes = (ds.X_train.shape, ds.X_val.shape, ds.X_test.shape)
    assert shapes == EXPECTED_SHAPES


def test_ciciot_feature_set_is_stable() -> None:
    """Feature identity and order both matter: LightGBM splits on column index."""
    ds = _prepare("binary")
    digest = hashlib.sha256("|".join(ds.feature_names).encode()).hexdigest()[:16]
    assert digest == EXPECTED_FEATURE_HASH
    assert ds.n_features == 45


def test_ciciot_clean_stats_are_stable() -> None:
    """Cleaning counts are reported in the manuscript's Table 2."""
    ds = _prepare("binary")
    for key, value in EXPECTED_CLEAN.items():
        assert ds.clean_stats[key] == value, f"clean_stats[{key!r}] moved"


def test_scaler_is_fit_on_train_only() -> None:
    """Standardised training features should be ~zero-mean, ~unit-variance.

    If the scaler were fit on the full set instead, train statistics would drift
    off those targets. This is the one leakage property the current pipeline
    does hold, and the clinical port must not regress it.
    """
    ds = _prepare("binary")
    assert np.allclose(ds.X_train.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(ds.X_train.std(axis=0), 1.0, atol=1e-3)
