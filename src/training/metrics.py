"""Evaluation metrics and figures for forgery detection.

Provides the metric set required by the OSDFD paper and the project spec:
Accuracy, AUC, F1, Average Precision (AP), Equal Error Rate (EER), FPR and FNR,
plus confusion-matrix / ROC / PR figures for logging.

The metrics are computed from raw scores (sigmoid probabilities) and binary
labels with numpy / scikit-learn so they are logger-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute the Equal Error Rate and its operating threshold.

    EER is the ROC point where the False Positive Rate equals the False
    Negative Rate (1 - TPR).

    Args:
        labels: Binary ground-truth ``(N,)`` (1 = fake, 0 = real).
        scores: Predicted fake probabilities ``(N,)``.

    Returns:
        ``(eer, threshold)``.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full metric suite from scores and labels.

    Args:
        labels: Binary ground-truth ``(N,)``.
        scores: Predicted fake probabilities ``(N,)``.
        threshold: Decision threshold for the count-based metrics.

    Returns:
        Dict with ``acc``, ``auc``, ``f1``, ``precision``, ``recall``, ``ap``,
        ``eer``, ``fpr``, ``fnr``.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)

    metrics: dict[str, float] = {}
    metrics["acc"] = float(accuracy_score(labels, preds))
    # AUC / AP are undefined with a single class present in the batch.
    if len(np.unique(labels)) > 1:
        metrics["auc"] = float(roc_auc_score(labels, scores))
        metrics["ap"] = float(average_precision_score(labels, scores))
        metrics["eer"], _ = equal_error_rate(labels, scores)
    else:  # pragma: no cover - single-class edge case
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")
        metrics["eer"] = float("nan")
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
    metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
    metrics["recall"] = float(recall_score(labels, preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return metrics


def confusion_figure(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5):
    """Return a matplotlib confusion-matrix figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "Fake"])
    ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def roc_figure(labels: np.ndarray, scores: np.ndarray):
    """Return a matplotlib ROC-curve figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def pr_figure(labels: np.ndarray, scores: np.ndarray):
    """Return a matplotlib Precision-Recall-curve figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def log_figures(logger: Any, labels: np.ndarray, scores: np.ndarray, step: int, prefix: str) -> None:
    """Best-effort logging of confusion/ROC/PR figures to TB and/or W&B.

    Silently skips figures that cannot be produced (e.g. single-class batches).
    """
    import matplotlib.pyplot as plt

    if len(np.unique(np.asarray(labels).astype(int))) < 2:
        return

    figs = {
        f"{prefix}/confusion_matrix": confusion_figure(labels, scores),
        f"{prefix}/roc_curve": roc_figure(labels, scores),
        f"{prefix}/pr_curve": pr_figure(labels, scores),
    }
    for name, fig in figs.items():
        _log_single_figure(logger, name, fig, step)
        plt.close(fig)


def _log_single_figure(logger: Any, name: str, fig: Any, step: int) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return
    # TensorBoard
    if hasattr(experiment, "add_figure"):
        experiment.add_figure(name, fig, global_step=step)
    # Weights & Biases
    try:  # pragma: no cover - depends on optional wandb backend
        import wandb

        if hasattr(experiment, "log"):
            experiment.log({name: wandb.Image(fig)}, step=step)
    except Exception:
        pass
