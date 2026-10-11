"""Tests for multi-seed statistics and communication accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import comms, stats  # noqa: E402


def _run(seed_values: dict[str, float]) -> dict:
    """Build a minimal results dict shaped like run_all's output."""
    return {
        "models": {
            name: {"f1_macro": value, "accuracy": value, "mcc": value}
            for name, value in seed_values.items()
        }
    }


# --- aggregation -----------------------------------------------------

def test_aggregate_computes_mean_across_seeds():
    runs = [_run({"a": 0.8}), _run({"a": 0.9}), _run({"a": 1.0})]
    agg = stats.aggregate_seeds(runs)
    assert agg["n_seeds"] == 3
    assert agg["models"]["a"]["f1_macro"]["mean"] == pytest.approx(0.9)


def test_confidence_interval_brackets_the_mean():
    runs = [_run({"a": v}) for v in (0.80, 0.85, 0.90, 0.88, 0.92)]
    entry = stats.aggregate_seeds(runs)["models"]["a"]["f1_macro"]
    assert entry["ci95_low"] < entry["mean"] < entry["ci95_high"]


def test_identical_values_give_zero_width_interval():
    runs = [_run({"a": 0.9}) for _ in range(5)]
    entry = stats.aggregate_seeds(runs)["models"]["a"]["f1_macro"]
    assert entry["std"] == pytest.approx(0.0)
    assert entry["ci95_low"] == pytest.approx(entry["ci95_high"])


def test_more_variance_widens_the_interval():
    tight = stats.aggregate_seeds([_run({"a": v}) for v in
                                   (0.89, 0.90, 0.91, 0.90, 0.90)])
    wide = stats.aggregate_seeds([_run({"a": v}) for v in
                                  (0.50, 0.90, 0.99, 0.70, 0.95)])

    def width(agg):
        e = agg["models"]["a"]["f1_macro"]
        return e["ci95_high"] - e["ci95_low"]

    assert width(wide) > width(tight)


def test_failed_models_are_excluded():
    runs = [{"models": {"good": {"f1_macro": 0.9},
                        "bad": {"error": "exploded"}}}]
    agg = stats.aggregate_seeds(runs)
    assert "good" in agg["models"]
    assert "bad" not in agg["models"]


# --- significance ----------------------------------------------------

def test_detects_a_consistent_improvement():
    """Reference wins on every seed by a clear margin — must be significant."""
    runs = [
        _run({"fed_hybrid_mlp_lgbm": r, "weak": w})
        for r, w in [(0.90, 0.70), (0.91, 0.73), (0.89, 0.68),
                     (0.92, 0.74), (0.90, 0.71)]
    ]
    result = stats.compare_models(runs, metric="f1_macro")
    entry = result["comparisons"]["weak"]

    assert entry["mean_difference"] > 0
    assert entry["wins"] == 5
    assert entry["t_test_p"] < 0.05
    assert entry["significant_holm"] is True


def test_constant_nonzero_difference_is_significant():
    """Zero variance with a real gap is the strongest possible evidence.

    scipy cannot compute this (it divides by zero), so it must be handled by
    its limit rather than discarded as nan.
    """
    runs = [_run({"fed_hybrid_mlp_lgbm": 0.90, "weak": 0.70}) for _ in range(5)]
    entry = stats.compare_models(runs, metric="f1_macro")["comparisons"]["weak"]

    assert entry["t_test_p"] == 0.0
    assert entry["significant_holm"] is True


def test_identical_models_are_not_significant():
    runs = [_run({"fed_hybrid_mlp_lgbm": 0.90, "twin": 0.90}) for _ in range(5)]
    entry = stats.compare_models(runs, metric="f1_macro")["comparisons"]["twin"]

    assert entry["t_test_p"] == 1.0
    assert entry["significant_holm"] is False


def test_does_not_claim_significance_on_noise():
    """Overlapping, inconsistent results must not be reported as a win."""
    rng = np.random.default_rng(0)
    runs = [
        _run({"fed_hybrid_mlp_lgbm": float(rng.normal(0.85, 0.05)),
              "similar": float(rng.normal(0.85, 0.05))})
        for _ in range(5)
    ]
    entry = stats.compare_models(runs, metric="f1_macro")["comparisons"]["similar"]
    assert entry["significant_holm"] is False


def test_holm_correction_is_applied():
    """With many comparisons, thresholds must be stricter than raw alpha."""
    runs = [
        _run({"fed_hybrid_mlp_lgbm": 0.90 + i * 0.001,
              **{f"b{j}": 0.70 + i * 0.001 for j in range(5)}})
        for i in range(5)
    ]
    result = stats.compare_models(runs, metric="f1_macro", alpha=0.05)
    thresholds = [e["holm_threshold"] for e in result["comparisons"].values()]
    assert min(thresholds) < 0.05
    assert result["correction"].startswith("Holm")


def test_missing_reference_is_reported():
    assert "error" in stats.compare_models([_run({"a": 0.9})], reference="absent")


def test_summary_table_is_markdown():
    agg = stats.aggregate_seeds([_run({"a": v}) for v in (0.8, 0.9, 0.85)])
    table = stats.summary_table(agg, metrics=("f1_macro",))
    assert table.startswith("|")
    assert "±" in table


# --- communication ---------------------------------------------------

class _Model(nn.Module):
    def __init__(self, in_dim=40, emb=32):
        super().__init__()
        self.encoder = nn.Linear(in_dim, emb)
        self.head = nn.Linear(emb, 2)


def test_only_encoder_parameters_are_counted_as_transmitted():
    model = _Model()
    assert comms.encoder_parameter_count(model) < comms.count_parameters(model)


def test_communication_scales_with_rounds():
    model = _Model()

    def total(rounds):
        return comms.analyse(
            model, n_clients=10, n_rounds=rounds, client_fraction=1.0,
            n_train_samples=1000, n_raw_features=40, embedding_dim=32,
        )["channel_1_model_updates"]["total_bytes"]

    assert total(20) == 2 * total(10)


def test_embedding_channel_scales_with_dataset_size():
    """This is the architecture's distinguishing cost — it must be measured."""
    model = _Model()

    def embedding_bytes(n):
        return comms.analyse(
            model, n_clients=10, n_rounds=10, client_fraction=1.0,
            n_train_samples=n, n_raw_features=40, embedding_dim=32,
        )["channel_2_embeddings"]["total_bytes"]

    assert embedding_bytes(20_000) == 2 * embedding_bytes(10_000)


def test_compression_ratio_reflects_feature_reduction():
    result = comms.analyse(
        _Model(), n_clients=5, n_rounds=5, client_fraction=1.0,
        n_train_samples=100_000, n_raw_features=40, embedding_dim=8,
    )
    assert result["comparison_to_centralized"]["embedding_compression_vs_raw"] == 5.0


def test_convergence_efficiency_finds_target_round():
    history = [{"round": i, "val_accuracy": a}
               for i, a in enumerate([0.5, 0.7, 0.85, 0.91, 0.93], start=1)]
    result = comms.convergence_efficiency(history, target=0.90)
    assert result["reached"] is True
    assert result["rounds_to_target"] == 4
    assert result["metric"] == "accuracy"


def test_convergence_efficiency_reports_failure_to_reach():
    history = [{"round": 1, "val_accuracy": 0.5}, {"round": 2, "val_accuracy": 0.6}]
    result = comms.convergence_efficiency(history, target=0.99)
    assert result["reached"] is False
    assert result["best_achieved"] == pytest.approx(0.6)


def test_convergence_efficiency_prefers_roc_auc_when_available():
    """Accuracy is satisfied by the majority-class predictor on rare outcomes.

    This history is what an 8.6%-prevalence run looks like: accuracy sits above
    0.90 from round 1 while the model has learned nothing, and only ROC-AUC
    shows when it actually starts discriminating.
    """
    history = [
        {"round": 1, "val_accuracy": 0.914, "val_roc_auc": 0.51},
        {"round": 2, "val_accuracy": 0.914, "val_roc_auc": 0.68},
        {"round": 3, "val_accuracy": 0.916, "val_roc_auc": 0.87},
    ]
    result = comms.convergence_efficiency(history, target=0.85)
    assert result["metric"] == "roc_auc"
    assert result["rounds_to_target"] == 3


def test_convergence_efficiency_metric_can_be_forced():
    history = [{"round": 1, "val_accuracy": 0.95, "val_roc_auc": 0.52}]
    forced = comms.convergence_efficiency(history, target=0.90, metric="accuracy")
    assert forced["reached"] is True
    assert forced["metric"] == "accuracy"
