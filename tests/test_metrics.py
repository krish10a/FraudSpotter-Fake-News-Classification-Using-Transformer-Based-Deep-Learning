import numpy as np

from fraudspotter.evaluation.calibration import (
    compute_brier_score,
    compute_expected_calibration_error,
)
from fraudspotter.evaluation.metrics import compute_classification_metrics
from fraudspotter.evaluation.statistical_tests import compute_bootstrap_ci, mcnemar_test


def test_classification_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4]])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "fake_f1" in metrics
    assert "real_f1" in metrics
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "mcc" in metrics
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_calibration_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6]])

    brier = compute_brier_score(y_true, y_prob)
    assert 0.0 <= brier <= 1.0

    ece, bin_stats = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(bin_stats["bin_counts"]) == 5


def test_statistical_tests():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred1 = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    y_pred2 = np.array([0, 1, 1, 1, 0, 1, 0, 0])

    pt, low, high = compute_bootstrap_ci(y_true, y_pred1, n_resamples=100)
    assert low <= pt <= high

    res = mcnemar_test(y_true, y_pred1, y_pred2)
    assert "statistic" in res
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0
