"""Evaluation metrics.

Reports macro averages alongside weighted ones throughout. On a dataset this
imbalanced, weighted F1 mostly measures performance on the majority class;
macro F1 is what actually shows whether rare attack types are detected. The
manuscript currently reports only single unqualified numbers — a reviewer will
ask which averaging was used.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Full metric suite for one model on one split."""
    n_classes = len(np.unique(y_true))
    binary = n_classes <= 2

    results: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # MCC is robust to imbalance and hard to inflate — worth reporting.
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        results.update(_probability_metrics(y_true, y_proba, binary))

    if class_names is not None:
        results["per_class"] = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
            labels=list(range(len(class_names))),
        )

    return results


def _probability_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, binary: bool
) -> dict:
    """ROC-AUC, average precision and calibration, guarded against degenerate splits.

    Under heavy imbalance the discrimination metrics are not enough on their
    own. AUPRC depends on prevalence, so it is not comparable between splits
    with different base rates (notably within-site versus held-out-site), and
    prevalence is emitted alongside it so a reader can normalise. Calibration is
    reported because a clinical risk score that discriminates well but is
    systematically over-confident is not deployable, and because TRIPOD
    reporting expects it.
    """
    out: dict = {}
    try:
        if binary:
            scores = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            out["roc_auc"] = float(roc_auc_score(y_true, scores))
            out["average_precision"] = float(average_precision_score(y_true, scores))
            prevalence = float(np.mean(y_true))
            out["prevalence"] = prevalence
            # AUPRC of a random classifier equals prevalence; rescale so 0 is
            # chance and 1 is perfect, making values comparable across splits.
            if prevalence < 1.0:
                out["ap_normalized"] = float(
                    (out["average_precision"] - prevalence) / (1.0 - prevalence)
                )
            out["brier"] = float(np.mean((scores - y_true) ** 2))
            out.update(_calibration(y_true, scores))
        else:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
            out["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
    except ValueError as exc:
        # Happens when a class is absent from the split entirely.
        out["roc_auc_error"] = str(exc)
    return out


def _calibration(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Calibration slope, intercept and 10-bin expected calibration error.

    Slope and intercept come from regressing the outcome on the logit of the
    predicted probability: 1.0 and 0.0 are perfect. A slope below 1 means
    predictions are too extreme, which is the usual failure mode of a model
    trained with class reweighting.
    """
    out: dict = {}
    eps = 1e-6
    p = np.clip(scores, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    try:
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(logit.reshape(-1, 1), y_true)
        out["calibration_slope"] = float(lr.coef_[0][0])
        out["calibration_intercept"] = float(lr.intercept_[0])
    except (ValueError, np.linalg.LinAlgError):
        pass

    edges = np.linspace(0.0, 1.0, 11)
    bins = np.digitize(p, edges[1:-1])
    ece = 0.0
    for b in range(10):
        mask = bins == b
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(y_true[mask].mean() - p[mask].mean())
    out["ece_10bin"] = float(ece)
    return out


# Column sets per task type. Multiclass keeps exactly the original columns so
# the CIC-IoT tables in the manuscript under review still render identically.
MULTI_COLUMNS = (
    "accuracy", "balanced_accuracy", "precision_macro",
    "recall_macro", "f1_macro", "mcc",
)
# For a rare binary outcome, ranking metrics and calibration carry the result;
# accuracy is near-constant across models and says almost nothing.
BINARY_COLUMNS = (
    "roc_auc", "average_precision", "prevalence", "brier",
    "calibration_slope", "balanced_accuracy", "f1_macro", "mcc",
)

_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Bal. Acc",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "mcc": "MCC",
    "roc_auc": "AUROC",
    "average_precision": "AUPRC",
    "ap_normalized": "AUPRC (norm)",
    "prevalence": "Prevalence",
    "brier": "Brier",
    "calibration_slope": "Cal. slope",
    "ece_10bin": "ECE",
}


def _is_binary(results: dict) -> bool:
    return "roc_auc" in results or "average_precision" in results


def headline(results: dict) -> str:
    """One-line summary for logs, keyed to the task type."""
    columns = BINARY_COLUMNS if _is_binary(results) else MULTI_COLUMNS
    parts = [
        f"{_LABELS.get(c, c)}={results[c]:.4f}"
        for c in columns if isinstance(results.get(c), (int, float))
    ]
    return " ".join(parts)


def comparison_table(
    all_results: dict[str, dict], columns: tuple[str, ...] | None = None
) -> str:
    """Markdown table across models — drops straight into the manuscript.

    A model that failed is rendered as a FAILED row rather than raising, so one
    broken baseline cannot destroy the table for every other model.
    """
    ok = [r for r in all_results.values() if "error" not in r]
    if columns is None:
        columns = (
            BINARY_COLUMNS if ok and _is_binary(ok[0]) else MULTI_COLUMNS
        )

    titles = "| Model | " + " | ".join(_LABELS.get(c, c) for c in columns) + " |"
    # Parenthesised deliberately: without it the adjacent string literals
    # concatenate before the repeat and the title row is duplicated.
    rule = ("|---" * (len(columns) + 1)) + "|"
    header = titles + "\n" + rule

    def _cell(value) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:.4f}"
        return "n/a"

    rows = []
    for name, r in all_results.items():
        if "error" in r:
            rows.append(f"| {name} | " + " | ".join(["FAILED"] * len(columns)) + " |")
        else:
            rows.append(
                f"| {name} | " + " | ".join(_cell(r.get(c)) for c in columns) + " |"
            )
    return "\n".join([header, *rows])
