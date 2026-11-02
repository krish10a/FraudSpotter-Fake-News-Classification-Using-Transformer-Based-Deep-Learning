from typing import Callable, Dict, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = lambda yt, yp: f1_score(
        yt, yp, average="macro"
    ),
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a given metric function."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)

    point_estimate = metric_func(y_true, y_pred)
    bootstrap_scores = []

    for _ in range(n_resamples):
        indices = rng.integers(0, n_samples, size=n_samples)
        sample_yt = y_true[indices]
        sample_yp = y_pred[indices]
        score = metric_func(sample_yt, sample_yp)
        bootstrap_scores.append(score)

    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = float(np.percentile(bootstrap_scores, alpha * 100))
    upper_bound = float(np.percentile(bootstrap_scores, (1.0 - alpha) * 100))

    return float(point_estimate), lower_bound, upper_bound


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_model_a: np.ndarray,
    y_pred_model_b: np.ndarray,
    correction: bool = True,
) -> Dict[str, float]:
    """Perform McNemar's test comparing two classifiers on the same test set."""
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_pred_model_a)
    y_b = np.asarray(y_pred_model_b)

    correct_a = y_a == y_true
    correct_b = y_b == y_true

    # Contingency Table:
    # b: A correct, B wrong
    # c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    total_discordant = b + c
    if total_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b_discordant": b,
            "c_discordant": c,
        }

    if correction:
        # Edwards continuity correction
        stat = (abs(b - c) - 1.0) ** 2 / total_discordant
    else:
        stat = (b - c) ** 2 / total_discordant

    p_value = 1.0 - stats.chi2.cdf(stat, df=1)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "b_discordant": b,
        "c_discordant": c,
    }
