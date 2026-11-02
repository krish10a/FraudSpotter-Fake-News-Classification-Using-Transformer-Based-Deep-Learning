from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # Handle 2D probability array or 1D positive class probabilities
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            p_pos = y_prob[:, 1]
        else:
            p_pos = y_prob

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, p_pos))
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, p_pos))
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics
