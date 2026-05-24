import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    # zero_division=0 avoids warnings when a model predicts only one class.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC AUC is only defined when the test set contains both classes.
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(y_true, y_pred, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed labels keep the axes consistent even with rare missing classes.
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["0", "1"],
        cmap="Blues",
        normalize="all",
        values_format=".2f",
    )
    display.ax_.set_title(title)
    display.ax_.set_xlabel("Predicted label")
    display.ax_.set_ylabel("True label")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)
