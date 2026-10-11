"""Publication figures.

Replaces every figure in the manuscript with one generated from measured
results. Saved at 300 dpi as both PNG and PDF for journal submission.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless — required on Kaggle
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

logger = logging.getLogger(__name__)

DPI = 300
sns.set_theme(style="whitegrid", context="paper")


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", name)


def convergence(history: list[dict], out_dir: Path) -> None:
    """Per-round FedAvg convergence — replaces manuscript Figures 5 and 6.

    One consistent x-axis, so the 100-vs-200-epoch inconsistency the reviewers
    flagged cannot recur.
    """
    if not history:
        return
    rounds = [h["round"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rounds, [h["mean_client_train_loss"] for h in history],
                 marker="o", ms=3, label="Mean client training loss")
    axes[0].plot(rounds, [h["val_loss"] for h in history],
                 marker="s", ms=3, label="Global validation loss")
    axes[0].set(xlabel="Communication round", ylabel="Cross-entropy loss",
                title="FedAvg convergence — loss")
    axes[0].legend()

    axes[1].plot(rounds, [h["val_accuracy"] for h in history],
                 marker="o", ms=3, color="tab:green")
    axes[1].set(xlabel="Communication round", ylabel="Accuracy",
                title="FedAvg convergence — global validation accuracy")

    fig.tight_layout()
    _save(fig, out_dir, "fig_convergence")


def partition_heatmap(parts, out_dir: Path) -> None:
    """Per-client class distribution — makes the non-IID split visible."""
    matrix = parts.class_matrix()
    fig, ax = plt.subplots(figsize=(max(6, len(parts.class_names) * 0.8),
                                    max(4, len(parts) * 0.35)))
    sns.heatmap(
        matrix, annot=matrix.shape[1] <= 10, fmt="d", cmap="YlGnBu",
        xticklabels=parts.class_names,
        yticklabels=[f"Client {i}" for i in range(len(parts))],
        cbar_kws={"label": "Sample count"}, ax=ax,
    )
    ax.set(title=f"Non-IID client partition ({parts.scheme}, α={parts.alpha})",
           xlabel="Class", ylabel="Client")
    fig.tight_layout()
    _save(fig, out_dir, "fig_partition")


def confusion(results: dict, class_names: list[str], out_dir: Path,
              model: str = "fed_hybrid_mlp_lgbm") -> None:
    """Row-normalised confusion matrix for the proposed model."""
    entry = results["models"].get(model)
    if not entry or "confusion_matrix" not in entry:
        return

    cm = np.array(entry["confusion_matrix"])
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 0.9),
                                    max(4, len(class_names) * 0.8)))
    sns.heatmap(norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, cbar_kws={"label": "Proportion of true class"}, ax=ax)
    ax.set(title=f"Confusion matrix — {model}",
           xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    _save(fig, out_dir, "fig_confusion_matrix")


def roc_pr_curves(model_obj, X_test, y_test, out_dir: Path) -> None:
    """ROC and precision-recall for the proposed model (binary only)."""
    if model_obj is None or len(np.unique(y_test)) != 2:
        return
    proba = model_obj.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0],
                                     name="Fed Hybrid MLP-LightGBM")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1],
                                            name="Fed Hybrid MLP-LightGBM")
    axes[1].set_title("Precision-Recall curve")

    fig.tight_layout()
    _save(fig, out_dir, "fig_roc_pr")


def model_comparison(results: dict, out_dir: Path) -> None:
    """Grouped bar chart across models — replaces manuscript Figure 11."""
    ok = {k: v for k, v in results["models"].items() if "error" not in v}
    if not ok:
        return

    shown = ["accuracy", "balanced_accuracy", "precision_macro",
             "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Balanced Acc.", "Precision", "Recall", "F1"]
    names = list(ok.keys())

    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.5), 5))
    for i, (key, label) in enumerate(zip(shown, labels)):
        ax.bar(x + (i - len(shown) / 2) * width,
               [ok[n].get(key, 0) for n in names], width, label=label)

    ax.set(ylabel="Score", ylim=(0, 1.05),
           title="Model comparison (identical split, macro averaging)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(ncol=len(shown), loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig_model_comparison")


def embedding_tsne(model_obj, X_test, y_test, class_names: list[str],
                   out_dir: Path, max_points: int = 3000) -> None:
    """t-SNE of the *learned embeddings* — replaces manuscript Figure 10.

    Unlike the original, this visualises the representation the model actually
    produces, so it evidences that the federated encoder learned a separable
    space rather than merely that the raw data is separable.
    """
    if model_obj is None:
        return

    rng = np.random.default_rng(0)
    idx = (rng.choice(len(X_test), max_points, replace=False)
           if len(X_test) > max_points else np.arange(len(X_test)))

    Z = model_obj.embed(X_test[idx])
    perplexity = min(30, max(5, len(idx) // 4))
    coords = TSNE(n_components=2, perplexity=perplexity,
                  random_state=0, init="pca").fit_transform(Z)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        mask = y_test[idx] == c
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6, label=name)
    ax.set(title="t-SNE of federated MLP embeddings (test set)",
           xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig_tsne_embeddings")


def privacy_utility_curve(sweep_results: dict, out_dir: Path,
                          metric: str = "f1_macro") -> None:
    """Utility against privacy budget — the headline figure for plan item 7.

    `sweep_results` is the dict returned by ablations.sweep_privacy_utility.
    """
    points, no_dp = [], None
    for key, run in sweep_results.items():
        if isinstance(run, dict) and "error" in run:
            continue
        # Multi-seed runs nest under "aggregated"; single runs don't.
        if "aggregated" in run:
            entry = run["aggregated"]["models"].get("fed_hybrid_mlp_lgbm", {}).get(metric)
            if entry is None:
                continue
            value, err = entry["mean"], (entry["ci95_high"] - entry["ci95_low"]) / 2
        else:
            model = run.get("models", {}).get("fed_hybrid_mlp_lgbm", {})
            if metric not in model:
                continue
            value, err = model[metric], 0.0

        if key == "None":
            no_dp = (value, err)
        else:
            points.append((float(key), value, err))

    if not points:
        return
    points.sort()
    eps, values, errs = zip(*points)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(eps, values, yerr=errs, marker="o", capsize=4,
                label="Fed Hybrid MLP-LightGBM (DP)")
    if no_dp is not None:
        ax.axhline(no_dp[0], ls="--", color="tab:red",
                   label=f"No DP ({no_dp[0]:.3f})")
    ax.set(xscale="log", xlabel="Privacy budget ε (log scale)",
           ylabel=metric.replace("_", " ").title(),
           title="Privacy–utility trade-off")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig_privacy_utility")


def sweep_curve(sweep_results: dict, parameter: str, out_dir: Path,
                metric: str = "f1_macro", log_x: bool = False) -> None:
    """Generic one-parameter ablation curve (clients, alpha, embedding dim...)."""
    points = []
    for key, run in sweep_results.items():
        if not isinstance(run, dict) or "error" in run:
            continue
        if "aggregated" in run:
            entry = run["aggregated"]["models"].get("fed_hybrid_mlp_lgbm", {}).get(metric)
            if entry is None:
                continue
            points.append((float(key), entry["mean"],
                           (entry["ci95_high"] - entry["ci95_low"]) / 2))
        else:
            model = run.get("models", {}).get("fed_hybrid_mlp_lgbm", {})
            if metric in model:
                points.append((float(key), model[metric], 0.0))

    if not points:
        return
    points.sort()
    xs, values, errs = zip(*points)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, values, yerr=errs, marker="o", capsize=4)
    if log_x:
        ax.set_xscale("log")
    ax.set(xlabel=parameter.replace("_", " ").title(),
           ylabel=metric.replace("_", " ").title(),
           title=f"Effect of {parameter.replace('_', ' ')}")
    fig.tight_layout()
    _save(fig, out_dir, f"fig_sweep_{parameter}")


def generate_all(results, ds, parts, model_obj, out_dir: Path) -> None:
    """Generate every figure, tolerating individual failures."""
    out_dir = Path(out_dir)
    tasks = [
        ("convergence", lambda: convergence(results.get("fed_history", []), out_dir)),
        ("partition", lambda: partition_heatmap(parts, out_dir)),
        ("confusion", lambda: confusion(results, ds.class_names, out_dir)),
        ("roc_pr", lambda: roc_pr_curves(model_obj, ds.X_test, ds.y_test, out_dir)),
        ("comparison", lambda: model_comparison(results, out_dir)),
        ("tsne", lambda: embedding_tsne(model_obj, ds.X_test, ds.y_test,
                                        ds.class_names, out_dir)),
    ]
    for name, fn in tasks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Figure %r failed: %s", name, exc)
