"""Correctness tests for the pieces where a silent bug would corrupt results."""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import config, data, partition  # noqa: E402
from fedlgbm.fedavg import average_weights  # noqa: E402
from fedlgbm.mlp import MLPClassifier, extract_embeddings  # noqa: E402


# --- label mapping ---------------------------------------------------

@pytest.mark.parametrize(
    "label,expected",
    [
        ("BenignTraffic", "Benign"),
        ("Benign", "Benign"),
        # The critical case: DDoS must not be swallowed by the "dos" prefix.
        ("DDoS-ICMP_Flood", "DDoS"),
        ("DDoS-SlowLoris", "DDoS"),
        ("DoS-UDP_Flood", "DoS"),
        ("DoS-SYN_Flood", "DoS"),
        ("Mirai-greeth_flood", "Mirai"),
        ("Recon-PortScan", "Recon"),
        ("VulnerabilityScan", "Recon"),
        ("MITM-ArpSpoofing", "Spoofing"),
        ("DNS_Spoofing", "Spoofing"),
        ("SqlInjection", "Web"),
        ("XSS", "Web"),
        ("DictionaryBruteForce", "BruteForce"),
    ],
)
def test_category_mapping(label, expected):
    assert data.to_category(label) == expected


def test_ddos_not_misclassified_as_dos():
    """Longest-prefix matching must hold — this would silently merge classes."""
    assert data.to_category("DDoS-TCP_Flood") != data.to_category("DoS-TCP_Flood")


@pytest.mark.parametrize(
    "label,expected", [("BenignTraffic", 0), ("Benign", 0),
                       ("DDoS-ICMP_Flood", 1), ("XSS", 1)],
)
def test_binary_mapping(label, expected):
    assert data.to_binary(label) == expected


def test_label_normalisation_tolerates_formatting():
    assert data.to_category("  ddos-icmp_flood ") == "DDoS"
    assert data.to_category("BENIGNTRAFFIC") == "Benign"


# --- FedAvg ----------------------------------------------------------

def test_average_weights_is_sample_weighted():
    """Eq. 6: the result must be the weighted mean, not the plain mean."""
    a = OrderedDict({"w": torch.tensor([0.0, 0.0])})
    b = OrderedDict({"w": torch.tensor([10.0, 20.0])})

    # 90 samples of zeros vs 10 of [10,20] -> [1.0, 2.0], not [5.0, 10.0].
    result = average_weights([a, b], [90, 10])
    assert torch.allclose(result["w"], torch.tensor([1.0, 2.0]), atol=1e-6)


def test_average_weights_equal_sizes_is_plain_mean():
    a = OrderedDict({"w": torch.tensor([0.0, 4.0])})
    b = OrderedDict({"w": torch.tensor([2.0, 8.0])})
    result = average_weights([a, b], [50, 50])
    assert torch.allclose(result["w"], torch.tensor([1.0, 6.0]), atol=1e-6)


def test_average_weights_preserves_integer_buffers():
    """BatchNorm's num_batches_tracked is integer and must not be corrupted."""
    a = OrderedDict({"n": torch.tensor(5, dtype=torch.long)})
    b = OrderedDict({"n": torch.tensor(9, dtype=torch.long)})
    result = average_weights([a, b], [10, 10])
    assert result["n"].dtype == torch.long


def test_average_weights_rejects_empty():
    with pytest.raises(ValueError):
        average_weights([], [])


def test_average_weights_matches_pooled_single_client():
    """One client => aggregation must be an identity operation."""
    sd = OrderedDict({"w": torch.tensor([1.5, -2.5, 3.0])})
    result = average_weights([sd], [42])
    assert torch.allclose(result["w"], sd["w"], atol=1e-6)


# --- partitioning ----------------------------------------------------

def _labels(n=3000, n_classes=5, seed=0):
    return np.random.default_rng(seed).integers(0, n_classes, size=n)


def test_partition_is_a_true_disjoint_cover():
    """No sample may be duplicated or dropped — clients must not share rows."""
    y = _labels()
    cfg = config.PartitionConfig(n_clients=8, scheme="dirichlet", dirichlet_alpha=0.5)
    parts = partition.partition(y, cfg, [str(i) for i in range(5)], seed=1)

    combined = np.concatenate(parts.client_indices)
    assert len(combined) == len(y)
    assert len(np.unique(combined)) == len(y)
    assert set(combined.tolist()) == set(range(len(y)))


def test_iid_partition_is_balanced():
    y = _labels()
    cfg = config.PartitionConfig(n_clients=10, scheme="iid")
    parts = partition.partition(y, cfg, [str(i) for i in range(5)], seed=1)
    assert max(parts.sizes) - min(parts.sizes) <= 1


def test_lower_alpha_produces_more_skew():
    """The non-IID knob must actually control non-IID-ness."""
    y = _labels()
    names = [str(i) for i in range(5)]

    skewed = partition.partition(
        y, config.PartitionConfig(n_clients=10, dirichlet_alpha=0.05,
                                  min_samples_per_client=1), names, seed=3)
    uniform = partition.partition(
        y, config.PartitionConfig(n_clients=10, dirichlet_alpha=100.0,
                                  min_samples_per_client=1), names, seed=3)

    assert skewed.summary()["mean_label_skew"] < uniform.summary()["mean_label_skew"]


def test_partition_scheme_validated():
    with pytest.raises(ValueError):
        partition.partition(_labels(), config.PartitionConfig(scheme="nonsense"),
                            [str(i) for i in range(5)], seed=1)


# --- cleaning --------------------------------------------------------

def test_clean_removes_duplicates_and_reports_count():
    import pandas as pd

    base = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0],
                         "label": ["Benign", "XSS", "Benign"]})
    df = pd.concat([base, base.iloc[[0]]], ignore_index=True)

    cleaned, stats = data.clean(df, "label")
    assert stats["duplicates_removed"] == 1
    assert len(cleaned) == 3


def test_clean_handles_inf_and_missing():
    import pandas as pd

    df = pd.DataFrame({
        "a": [1.0, np.inf, 3.0, 4.0],
        "b": [1.0, 2.0, np.nan, 4.0],
        "label": ["Benign", "XSS", "Benign", "XSS"],
    })
    cleaned, stats = data.clean(df, "label")

    assert stats["infinite_values_replaced"] == 1
    numeric = cleaned.drop(columns=["label"])
    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


def test_clean_drops_constant_columns():
    import pandas as pd

    df = pd.DataFrame({"varies": [1.0, 2.0, 3.0, 4.0],
                       "constant": [7.0, 7.0, 7.0, 7.0],
                       "label": ["Benign", "XSS", "Benign", "XSS"]})
    cleaned, stats = data.clean(df, "label")
    assert "constant" in stats["constant_columns_dropped"]
    assert "constant" not in cleaned.columns


def test_find_label_column_is_case_insensitive():
    import pandas as pd

    for name in ["label", "Label", "LABEL", " label "]:
        df = pd.DataFrame({"a": [1.0], name: ["Benign"]})
        assert data.find_label_column(df) == name


def test_find_label_column_raises_when_absent():
    import pandas as pd

    with pytest.raises(ValueError):
        data.find_label_column(pd.DataFrame({"a": [1.0], "b": [2.0]}))


# --- MLP -------------------------------------------------------------

def test_encoder_output_dimension_matches_config():
    cfg = config.MLPConfig(hidden_dims=(32, 16), embedding_dim=8)
    model = MLPClassifier(input_dim=20, n_classes=3, cfg=cfg)
    Z = extract_embeddings(model, np.random.randn(50, 20).astype(np.float32),
                           torch.device("cpu"))
    assert Z.shape == (50, 8)


def test_embeddings_are_deterministic_in_eval_mode():
    """Dropout/BatchNorm must be disabled — otherwise embeddings are unstable."""
    cfg = config.MLPConfig(hidden_dims=(32,), embedding_dim=8, dropout=0.5)
    model = MLPClassifier(input_dim=20, n_classes=3, cfg=cfg)
    X = np.random.randn(50, 20).astype(np.float32)
    device = torch.device("cpu")

    a = extract_embeddings(model, X, device)
    b = extract_embeddings(model, X, device)
    assert np.allclose(a, b)
