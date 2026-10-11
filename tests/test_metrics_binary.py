"""Metric behaviour for rare binary outcomes.

Under 8.6% prevalence, accuracy is near-constant across models and carries
almost no information, so the reported table has to lead with ranking and
calibration. These tests pin that, and pin the multiclass columns unchanged so
the CIC-IoT tables in the manuscript under review still render identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import metrics  # noqa: E402


def _binary(n: int = 4000, prevalence: float = 0.086, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.int64)
    # A calibrated-ish score: positives drawn higher than negatives.
    scores = np.clip(rng.beta(2, 8, n) + 0.35 * y, 0.0, 1.0)
    proba = np.column_stack([1 - scores, scores])
    pred = (scores >= 0.5).astype(np.int64)
    return y, pred, proba


def test_binary_metrics_include_ranking_and_calibration() -> None:
    y, pred, proba = _binary()
    r = metrics.compute(y, pred, proba, class_names=["neg", "pos"])
    for key in ("roc_auc", "average_precision", "prevalence", "brier",
                "calibration_slope", "ece_10bin", "ap_normalized"):
        assert key in r, f"{key} missing from binary metrics"


def test_prevalence_matches_the_data() -> None:
    y, pred, proba = _binary()
    r = metrics.compute(y, pred, proba, class_names=["neg", "pos"])
    assert r["prevalence"] == pytest.approx(y.mean())


def test_average_precision_of_random_scores_is_about_prevalence() -> None:
    """The AUPRC floor is prevalence, which is why the table reports both."""
    rng = np.random.default_rng(3)
    y = (rng.random(6000) < 0.1).astype(np.int64)
    scores = rng.random(6000)
    proba = np.column_stack([1 - scores, scores])
    r = metrics.compute(y, (scores >= 0.5).astype(int), proba, ["neg", "pos"])
    assert r["average_precision"] == pytest.approx(y.mean(), abs=0.03)
    assert abs(r["ap_normalized"]) < 0.05


def test_normalized_ap_is_one_for_perfect_ranking() -> None:
    y = np.array([0] * 900 + [1] * 100)
    scores = np.concatenate([np.full(900, 0.01), np.full(100, 0.99)])
    proba = np.column_stack([1 - scores, scores])
    r = metrics.compute(y, (scores >= 0.5).astype(int), proba, ["neg", "pos"])
    assert r["average_precision"] == pytest.approx(1.0, abs=1e-6)
    assert r["ap_normalized"] == pytest.approx(1.0, abs=1e-6)


def test_brier_rewards_confident_correctness() -> None:
    y = np.array([0, 0, 1, 1])
    good = np.array([0.02, 0.03, 0.97, 0.98])
    bad = np.array([0.98, 0.97, 0.03, 0.02])
    r_good = metrics.compute(y, (good >= .5).astype(int),
                             np.column_stack([1 - good, good]), ["a", "b"])
    r_bad = metrics.compute(y, (bad >= .5).astype(int),
                            np.column_stack([1 - bad, bad]), ["a", "b"])
    assert r_good["brier"] < 0.01 < r_bad["brier"]


def test_calibration_slope_near_one_for_calibrated_scores() -> None:
    rng = np.random.default_rng(11)
    p = rng.uniform(0.02, 0.95, 20000)
    y = (rng.random(20000) < p).astype(np.int64)
    r = metrics.compute(y, (p >= .5).astype(int),
                        np.column_stack([1 - p, p]), ["neg", "pos"])
    assert r["calibration_slope"] == pytest.approx(1.0, abs=0.15)


def test_comparison_table_survives_a_failed_model() -> None:
    """One broken baseline must not destroy the table for the others."""
    y, pred, proba = _binary()
    good = metrics.compute(y, pred, proba, ["neg", "pos"])
    table = metrics.comparison_table({
        "centralized_lgbm": good,
        "broken_model": {"error": "CUDA unavailable"},
    })
    assert "FAILED" in table
    assert "centralized_lgbm" in table
    assert "AUROC" in table


def test_binary_table_leads_with_auroc_and_auprc() -> None:
    y, pred, proba = _binary()
    table = metrics.comparison_table({"m": metrics.compute(y, pred, proba, ["n", "p"])})
    assert "AUROC" in table and "AUPRC" in table


def test_multiclass_table_columns_are_unchanged() -> None:
    """Regression guard for the CIC-IoT tables in the paper under review."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, 500)
    pred = rng.integers(0, 4, 500)
    r = metrics.compute(y, pred, class_names=list("abcd"))
    table = metrics.comparison_table({"m": r})
    header = table.splitlines()[0]
    assert header == (
        "| Model | Accuracy | Bal. Acc | Precision (macro) | "
        "Recall (macro) | F1 (macro) | MCC |"
    )


def test_table_is_well_formed_markdown() -> None:
    """Header, one separator rule, one row per model — and no duplicated rule.

    The separator is built by repeating "|---", which silently absorbed the
    title row when the surrounding literals concatenated first.
    """
    y, pred, proba = _binary()
    r = metrics.compute(y, pred, proba, ["neg", "pos"])
    lines = metrics.comparison_table({"a": r, "b": r}).splitlines()
    assert len(lines) == 4, lines

    n_columns = lines[0].count("|") - 1
    assert lines[1] == ("|---" * n_columns) + "|"
    for row in lines[2:]:
        assert row.count("|") - 1 == n_columns


def test_headline_switches_on_task_type() -> None:
    y, pred, proba = _binary()
    assert "AUROC" in metrics.headline(metrics.compute(y, pred, proba, ["n", "p"]))

    rng = np.random.default_rng(6)
    y_m, pred_m = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    assert "Accuracy" in metrics.headline(
        metrics.compute(y_m, pred_m, class_names=list("abc"))
    )
