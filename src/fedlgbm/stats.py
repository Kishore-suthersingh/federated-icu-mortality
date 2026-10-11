"""Multi-seed aggregation and significance testing.

Single-run numbers are not publishable at a strong venue: with a non-IID
Dirichlet partition, run-to-run variance can exceed the gap between competing
methods, so one seed can show whichever winner you like. Every headline number
should be mean +/- 95% CI over seeds, with a significance test behind any claim
that one method beats another.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)

# Metrics summarised across seeds. Confusion matrices and per-class reports are
# structural, not scalar, so they are excluded here.
SCALAR_METRICS = (
    "accuracy", "balanced_accuracy",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "mcc", "roc_auc", "average_precision",
    "roc_auc_ovr_macro", "roc_auc_ovr_weighted",
)


def aggregate_seeds(runs: list[dict]) -> dict:
    """Collapse per-seed results into mean / std / 95% CI per model per metric.

    `runs` is a list of the `results` dicts returned by `run_all`, one per seed.
    """
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for run in runs:
        for model_name, model_results in run.get("models", {}).items():
            if "error" in model_results:
                continue
            for metric in SCALAR_METRICS:
                if metric in model_results:
                    collected[model_name][metric].append(float(model_results[metric]))

    summary: dict = {"n_seeds": len(runs), "models": {}}
    for model_name, metrics in collected.items():
        summary["models"][model_name] = {
            metric: _describe(np.array(values))
            for metric, values in metrics.items()
        }
    return summary


def _describe(values: np.ndarray) -> dict:
    """Mean, std and a t-based 95% CI."""
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci95_low": mean,
                "ci95_high": mean, "n": n, "values": values.tolist()}

    # ddof=1: sample std, since seeds are a sample of possible runs.
    std = float(values.std(ddof=1))
    half_width = float(sps.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "n": n,
        "values": values.tolist(),
    }


def compare_models(
    runs: list[dict],
    reference: str = "fed_hybrid_mlp_lgbm",
    metric: str = "f1_macro",
    alpha: float = 0.05,
) -> dict:
    """Paired significance tests: reference model vs every other model.

    Paired because all models see the identical split within a seed, so the
    seed effect is shared and pairing removes it — a paired test is both valid
    and more powerful here than an unpaired one.

    Reports a paired t-test, a Wilcoxon signed-rank test (no normality
    assumption, which matters at 5 seeds), and Cohen's d. Holm-Bonferroni
    correction is applied across comparisons, since testing one reference
    against many baselines inflates the false-positive rate.
    """
    per_model: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for name, results in run.get("models", {}).items():
            if "error" not in results and metric in results:
                per_model[name].append(float(results[metric]))

    if reference not in per_model:
        return {"error": f"Reference model {reference!r} not present in runs"}

    ref_values = np.array(per_model[reference])
    comparisons: dict = {}

    for name, values in per_model.items():
        if name == reference:
            continue
        other = np.array(values)
        if len(other) != len(ref_values) or len(ref_values) < 2:
            comparisons[name] = {"error": "Insufficient or mismatched seed counts"}
            continue

        differences = ref_values - other
        entry: dict = {
            "reference_mean": float(ref_values.mean()),
            "other_mean": float(other.mean()),
            "mean_difference": float(differences.mean()),
            "cohens_d": _cohens_d(differences),
            "wins": int((differences > 0).sum()),
            "n_seeds": len(differences),
        }

        # A paired t-test divides by the standard deviation of the differences,
        # which is zero when every seed gives the identical gap. That is a
        # degenerate input for scipy but not a degenerate result: the limit as
        # the variance vanishes with a non-zero mean is t -> inf, p -> 0. Treat
        # the two constant cases by their limits rather than reporting nan,
        # which would otherwise discard a perfectly consistent improvement.
        if np.allclose(differences, differences[0]):
            if np.isclose(differences[0], 0.0):
                entry["note"] = "Models are identical on every seed"
                entry["t_test_p"] = 1.0
                entry["wilcoxon_p"] = 1.0
            else:
                entry["note"] = (
                    "Difference is identical on every seed (zero variance); "
                    "p reported as its limiting value 0"
                )
                entry["t_test_p"] = 0.0
                entry["wilcoxon_p"] = 0.0
        else:
            entry["t_test_p"] = float(sps.ttest_rel(ref_values, other).pvalue)
            try:
                entry["wilcoxon_p"] = float(sps.wilcoxon(ref_values, other).pvalue)
            except ValueError as exc:
                entry["wilcoxon_p"] = float("nan")
                entry["wilcoxon_note"] = str(exc)

        comparisons[name] = entry

    _apply_holm(comparisons, alpha)

    return {
        "metric": metric,
        "reference": reference,
        "alpha": alpha,
        "correction": "Holm-Bonferroni across comparisons",
        "comparisons": comparisons,
    }


def _cohens_d(differences: np.ndarray) -> float:
    """Paired Cohen's d: mean difference over its standard deviation."""
    sd = differences.std(ddof=1)
    if sd == 0:
        return 0.0 if differences.mean() == 0 else float("inf")
    return float(differences.mean() / sd)


def _apply_holm(comparisons: dict, alpha: float) -> None:
    """Holm-Bonferroni step-down correction, applied in place."""
    testable = [
        (name, entry["t_test_p"])
        for name, entry in comparisons.items()
        if isinstance(entry.get("t_test_p"), float) and np.isfinite(entry["t_test_p"])
    ]
    if not testable:
        return

    testable.sort(key=lambda item: item[1])
    m = len(testable)
    still_rejecting = True

    for rank, (name, p) in enumerate(testable):
        threshold = alpha / (m - rank)
        # Holm is step-down: once a test fails, all later ones fail too.
        if still_rejecting and p <= threshold:
            comparisons[name]["significant_holm"] = True
        else:
            still_rejecting = False
            comparisons[name]["significant_holm"] = False
        comparisons[name]["holm_threshold"] = float(threshold)


def summary_table(aggregated: dict, metrics: tuple[str, ...] = (
    "accuracy", "balanced_accuracy", "f1_macro", "mcc",
)) -> str:
    """Markdown table of mean +/- CI, ready to paste into the manuscript."""
    header = "| Model | " + " | ".join(
        m.replace("_", " ").title() for m in metrics
    ) + " |"
    separator = "|---" * (len(metrics) + 1) + "|"

    rows = []
    for model_name, model_metrics in aggregated.get("models", {}).items():
        cells = []
        for metric in metrics:
            entry = model_metrics.get(metric)
            if entry is None:
                cells.append("—")
            else:
                half = (entry["ci95_high"] - entry["ci95_low"]) / 2.0
                cells.append(f"{entry['mean']:.4f} ± {half:.4f}")
        rows.append(f"| {model_name} | " + " | ".join(cells) + " |")

    caption = (
        f"\n\nMean ± half-width of the 95% confidence interval over "
        f"{aggregated.get('n_seeds', '?')} seeds."
    )
    return "\n".join([header, separator, *rows]) + caption
