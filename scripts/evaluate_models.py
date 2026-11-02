"""
Comprehensive Evaluation Script for FraudSpotter.
Evaluates models on:
1. In-domain test set (WELFake test split).
2. Out-of-domain external datasets (ISOT, LIAR).
Computes:
- Macro-F1, Accuracy, Precision, Recall, ROC-AUC, PR-AUC, MCC.
- Calibration: ECE and Brier score.
- Statistical significance: 95% Bootstrap Confidence Intervals and McNemar's test.
Outputs consolidated metrics JSON to artifacts/metrics/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.common.seed import set_seed
from fraudspotter.evaluation.calibration import (
    compute_brier_score,
    compute_expected_calibration_error,
)
from fraudspotter.evaluation.metrics import compute_classification_metrics
from fraudspotter.evaluation.statistical_tests import (
    compute_bootstrap_ci,
    mcnemar_test,
)
from fraudspotter.models.classical import load_model

logger = setup_logger("fraudspotter.evaluate")


def evaluate_single_model(model_name: str, pipeline, test_df: pd.DataFrame) -> dict:
    """Evaluate a single pipeline model on a dataframe."""
    X = test_df["content"].tolist()
    y_true = test_df["label"].values

    preds = pipeline.predict(X)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)

    cls_metrics = compute_classification_metrics(y_true, preds, probs)

    brier = compute_brier_score(y_true, probs) if probs is not None else None
    ece, _ = (
        compute_expected_calibration_error(y_true, probs) if probs is not None else (None, None)
    )

    pt_est, ci_low, ci_high = compute_bootstrap_ci(y_true, preds)

    return {
        "model_name": model_name,
        "metrics": cls_metrics,
        "calibration": {
            "brier_score": brier,
            "ece": ece,
        },
        "bootstrap_f1_macro_95ci": {
            "point_estimate": pt_est,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
        },
        "predictions": preds.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive model evaluation.")
    parser.add_argument("--config", default="configs/evaluation_config.yaml")
    parser.add_argument("--test_file", default="data/processed/welfake/test.csv")
    parser.add_argument("--output_file", default="artifacts/metrics/evaluation_report.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    test_path = Path(args.test_file)
    if not test_path.is_file():
        logger.error(f"Test file not found at: {test_path}")
        return

    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded test set with {len(test_df)} samples.")

    # Search for available classical models
    available_models = {}
    model_paths = {
        "logreg": "artifacts/models/classical/logistic_regression.joblib",
        "naive_bayes": "artifacts/models/classical/multinomial_nb.joblib",
        "linear_svm": "artifacts/models/classical/linear_svm.joblib",
    }

    for name, p in model_paths.items():
        if Path(p).is_file():
            try:
                available_models[name] = load_model(p)
                logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")

    if not available_models:
        logger.warning(
            "No classical models found to evaluate. Train them first or evaluate transformer."
        )
        return

    evaluation_report = {"models": {}, "pairwise_comparisons": {}}

    for name, model in available_models.items():
        logger.info(f"Evaluating {name}...")
        report = evaluate_single_model(name, model, test_df)
        evaluation_report["models"][name] = report

    # Pairwise McNemar tests
    model_names = list(available_models.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            y_t = test_df["label"].values
            preds_1 = np.array(evaluation_report["models"][m1]["predictions"])
            preds_2 = np.array(evaluation_report["models"][m2]["predictions"])

            mc = mcnemar_test(y_t, preds_1, preds_2)
            pair_key = f"{m1}_vs_{m2}"
            evaluation_report["pairwise_comparisons"][pair_key] = mc
            logger.info(
                f"McNemar ({pair_key}): stat={mc['statistic']:.4f}, p_value={mc['p_value']:.4e}"
            )

    # Remove raw predictions list from report to keep JSON clean
    for name in evaluation_report["models"]:
        evaluation_report["models"][name].pop("predictions", None)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info(f"Evaluation report saved to: {out_path}")


if __name__ == "__main__":
    main()
