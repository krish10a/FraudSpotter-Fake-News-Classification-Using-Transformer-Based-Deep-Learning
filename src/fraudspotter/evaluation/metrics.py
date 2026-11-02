from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraudspotter.evaluation.calibration import (
    compute_brier_score,
    compute_expected_calibration_error,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics adhering to:

    Internal Label Mapping:
      0 = REAL
      1 = FAKE
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Class-wise metrics (0 = REAL, 1 = FAKE)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix: [[True Real, False Fake], [False Real, True Fake]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Specificity = TN / (TN + FP) = Recall for REAL (class 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        # Class 0: REAL
        "real_precision": float(precisions[0]) if len(precisions) > 0 else 0.0,
        "real_recall": float(recalls[0]) if len(recalls) > 0 else 0.0,
        "real_f1": float(f1s[0]) if len(f1s) > 0 else 0.0,
        # Class 1: FAKE
        "fake_precision": float(precisions[1]) if len(precisions) > 1 else 0.0,
        "fake_recall": float(recalls[1]) if len(recalls) > 1 else 0.0,
        "fake_f1": float(f1s[1]) if len(f1s) > 1 else 0.0,
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {
            "true_real": int(tn),
            "false_fake": int(fp),
            "false_real": int(fn),
            "true_fake": int(tp),
            "matrix": cm.tolist(),
        },
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # Probability of positive class (FAKE = 1)
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            p_fake = y_prob[:, 1]
        else:
            p_fake = y_prob

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, p_fake))
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, p_fake))
        except ValueError:
            metrics["pr_auc"] = 0.0

        # Calibration metrics
        metrics["brier_score"] = compute_brier_score(y_true, p_fake)
        ece, _ = compute_expected_calibration_error(y_true, y_prob)
        metrics["expected_calibration_error"] = float(ece)

    return metrics
