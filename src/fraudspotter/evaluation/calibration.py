from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate the Brier score for probabilistic predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    return float(brier_score_loss(y_true, y_prob))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, list]]:
    """Compute Expected Calibration Error (ECE) and reliability bin statistics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
        predictions = (y_prob >= 0.5).astype(int)

    accuracies = predictions == y_true
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = {
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
    }

    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        bin_stats["bin_counts"].append(int(bin_count))

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

            bin_stats["bin_accuracies"].append(float(bin_acc))
            bin_stats["bin_confidences"].append(float(bin_conf))
        else:
            bin_stats["bin_accuracies"].append(0.0)
            bin_stats["bin_confidences"].append(0.0)

    return float(ece), bin_stats
