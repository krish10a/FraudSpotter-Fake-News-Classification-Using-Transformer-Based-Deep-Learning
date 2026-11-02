"""
FraudSpotter - Classical Baseline Training & Validation Script.
Executes training for:
  1. Majority Sanity Baseline
  2. Word TF-IDF + Multinomial Naive Bayes (alpha tuned on validation)
  3. Word + Char TF-IDF + Logistic Regression (C tuned on validation)
  4. (Optional) Word + Char TF-IDF + Linear SVM (calibrated, C tuned on validation)

Adheres strictly to canonical rules:
  - Internal labels: REAL=0, FAKE=1
  - Float32 sparse matrices for memory-efficient training
  - Hyperparameter tuning using validation data only (no test data access)
  - Saves full pipelines, vectorizers, and JSON validation metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure 'src' is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.common.seed import set_seed
from fraudspotter.data.loader import load_split_data
from fraudspotter.evaluation.metrics import compute_classification_metrics
from fraudspotter.models.classical import (
    MajorityClassifier,
    Pipeline,
    build_combined_feature_union,
    build_word_vectorizer,
    save_model,
    tune_linear_svm,
    tune_logistic_regression,
    tune_multinomial_nb,
)

logger = setup_logger("fraudspotter.train_classical")


def resolve_split_path(
    explicit_path: Optional[str],
    drive_dir: Optional[str],
    split_name: str,
) -> Path:
    """Resolve split path checking explicit path, drive_dir, parquet, and csv."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p

    if drive_dir:
        d_p = Path(drive_dir)
        pq_candidate = d_p / f"{split_name}.parquet"
        if pq_candidate.is_file():
            return pq_candidate
        csv_candidate = d_p / f"{split_name}.csv"
        if csv_candidate.is_file():
            return csv_candidate

    # Local default candidates
    local_dir = Path("data/processed/welfake")
    pq_local = local_dir / f"{split_name}.parquet"
    if pq_local.is_file():
        return pq_local
    csv_local = local_dir / f"{split_name}.csv"
    if csv_local.is_file():
        return csv_local

    # Fallback to explicit_path even if not found yet (will trigger error)
    return Path(explicit_path or f"data/processed/welfake/{split_name}.parquet")


def main():
    parser = argparse.ArgumentParser(
        description="Train classical ML baseline models for FraudSpotter."
    )
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train_file", default=None, help="Path to train.parquet or train.csv")
    parser.add_argument("--val_file", default=None, help="Path to val.parquet or val.csv")
    parser.add_argument(
        "--drive_dir",
        default=None,
        help="Optional Google Drive directory containing train/val Parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/models/classical",
        help="Directory to save trained models and vectorizers",
    )
    parser.add_argument(
        "--metrics_file",
        default="artifacts/metrics/classical_validation_metrics.json",
        help="JSON file path to save validation metrics",
    )
    parser.add_argument(
        "--include_svm",
        action="store_true",
        help="Include optional Linear SVM baseline in training",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    train_path = resolve_split_path(args.train_file, args.drive_dir, "train")
    val_path = resolve_split_path(args.val_file, args.drive_dir, "val")

    if not train_path.is_file():
        logger.error(f"Training split not found: {train_path}")
        return
    if not val_path.is_file():
        logger.error(f"Validation split not found: {val_path}")
        return

    logger.info(f"Loading training split from: {train_path}")
    train_df, y_train_series = load_split_data(str(train_path))
    logger.info(f"Loading validation split from: {val_path}")
    val_df, y_val_series = load_split_data(str(val_path))

    X_train_raw = train_df["content"].tolist()
    y_train = y_train_series.values
    X_val_raw = val_df["content"].tolist()
    y_val = y_val_series.values

    # Check and log label distribution (0=REAL, 1=FAKE)
    train_counts = np.bincount(y_train, minlength=2)
    val_counts = np.bincount(y_val, minlength=2)
    logger.info("=== Internal Label Distribution (0=REAL, 1=FAKE) ===")
    logger.info(
        f"Train: Total={len(y_train)}, REAL(0)={train_counts[0]} ({train_counts[0] / len(y_train):.1%}), "
        f"FAKE(1)={train_counts[1]} ({train_counts[1] / len(y_train):.1%})"
    )
    logger.info(
        f"Val:   Total={len(y_val)}, REAL(0)={val_counts[0]} ({val_counts[0] / len(y_val):.1%}), "
        f"FAKE(1)={val_counts[1]} ({val_counts[1] / len(y_val):.1%})"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    classical_cfg = config.get("classical", {})
    word_cfg = classical_cfg.get("word_tfidf", {})
    char_cfg = classical_cfg.get("char_tfidf", {})

    report: Dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "label_convention": {"0": "REAL", "1": "FAKE"},
            "train_file": str(train_path),
            "val_file": str(val_path),
        },
        "models": {},
    }

    # -------------------------------------------------------------
    # 1. Majority Sanity Baseline
    # -------------------------------------------------------------
    logger.info("\n--- [1/4] Training Majority Sanity Baseline ---")
    majority_clf = MajorityClassifier()
    majority_clf.fit(X_train_raw, y_train)

    maj_preds = majority_clf.predict(X_val_raw)
    maj_probs = majority_clf.predict_proba(X_val_raw)
    maj_metrics = compute_classification_metrics(y_val, maj_preds, maj_probs)

    maj_pipeline = Pipeline(
        [
            ("features", build_word_vectorizer(max_features=10)),
            ("clf", majority_clf),
        ]
    )
    maj_path = out_dir / "majority_baseline.joblib"
    save_model(maj_pipeline, str(maj_path))

    report["models"]["majority_baseline"] = {
        "selected_hyperparameters": {"majority_class": int(majority_clf.majority_class_)},
        "validation_metrics": maj_metrics,
        "artifact_path": str(maj_path),
    }
    logger.info(
        f"Majority Baseline -> Accuracy: {maj_metrics['accuracy']:.4f}, "
        f"Macro-F1: {maj_metrics['macro_f1']:.4f}"
    )

    # -------------------------------------------------------------
    # 2. Multinomial Naive Bayes (Word TF-IDF, Float32)
    # -------------------------------------------------------------
    logger.info("\n--- [2/4] Fitting Word TF-IDF (Float32) for Multinomial Naive Bayes ---")
    word_vec = build_word_vectorizer(
        ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
        min_df=word_cfg.get("min_df", 2),
        max_df=word_cfg.get("max_df", 0.95),
        max_features=word_cfg.get("max_features", 150000),
        sublinear_tf=word_cfg.get("sublinear_tf", True),
        dtype=np.float32,
    )
    X_train_word_sp = word_vec.fit_transform(X_train_raw)
    X_val_word_sp = word_vec.transform(X_val_raw)
    logger.info(
        f"Word TF-IDF vocabulary size: {X_train_word_sp.shape[1]:,} features "
        f"(dtype={X_train_word_sp.dtype})"
    )

    nb_cfg = classical_cfg.get("multinomial_nb", {})
    alpha_candidates = nb_cfg.get("alpha_candidates", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0])

    logger.info(f"Tuning MultinomialNB across alpha candidates: {alpha_candidates}")
    best_nb, nb_tuning_history = tune_multinomial_nb(
        X_train=X_train_word_sp,
        y_train=y_train,
        X_val=X_val_word_sp,
        y_val=y_val,
        alpha_candidates=alpha_candidates,
        fit_prior=nb_cfg.get("fit_prior", True),
    )

    nb_preds = best_nb.predict(X_val_word_sp)
    nb_probs = best_nb.predict_proba(X_val_word_sp)
    nb_metrics = compute_classification_metrics(y_val, nb_preds, nb_probs)

    nb_pipeline = Pipeline([("features", word_vec), ("clf", best_nb)])
    nb_path = out_dir / "multinomial_nb.joblib"
    nb_vec_path = out_dir / "multinomial_nb_vectorizer.joblib"
    save_model(nb_pipeline, str(nb_path))
    save_model(word_vec, str(nb_vec_path))

    report["models"]["multinomial_nb"] = {
        "selected_hyperparameters": {"alpha": float(best_nb.alpha)},
        "tuning_history": nb_tuning_history,
        "validation_metrics": nb_metrics,
        "artifact_path": str(nb_path),
        "vectorizer_path": str(nb_vec_path),
    }
    logger.info(
        f"MultinomialNB (Best alpha={best_nb.alpha}) -> "
        f"Accuracy: {nb_metrics['accuracy']:.4f}, Macro-F1: {nb_metrics['macro_f1']:.4f}, "
        f"ROC-AUC: {nb_metrics.get('roc_auc', 0.0):.4f}"
    )

    # -------------------------------------------------------------
    # 3. Logistic Regression (Combined Word + Char TF-IDF, Float32)
    # -------------------------------------------------------------
    logger.info(
        "\n--- [3/4] Fitting Combined Word + Char TF-IDF (Float32) for Logistic Regression ---"
    )

    combined_features = build_combined_feature_union(
        word_params={
            "ngram_range": tuple(word_cfg.get("ngram_range", [1, 2])),
            "min_df": word_cfg.get("min_df", 2),
            "max_df": word_cfg.get("max_df", 0.95),
            "max_features": word_cfg.get("max_features", 150000),
            "sublinear_tf": word_cfg.get("sublinear_tf", True),
            "dtype": np.float32,
        },
        char_params={
            "ngram_range": tuple(char_cfg.get("ngram_range", [3, 5])),
            "min_df": char_cfg.get("min_df", 2),
            "max_df": char_cfg.get("max_df", 0.95),
            "max_features": char_cfg.get("max_features", 150000),
            "sublinear_tf": char_cfg.get("sublinear_tf", True),
            "dtype": np.float32,
        },
    )

    X_train_comb_sp = combined_features.fit_transform(X_train_raw)
    X_val_comb_sp = combined_features.transform(X_val_raw)
    logger.info(
        f"Combined Word+Char TF-IDF total features: {X_train_comb_sp.shape[1]:,} "
        f"(dtype={X_train_comb_sp.dtype})"
    )

    lr_cfg = classical_cfg.get("logistic_regression", {})
    c_candidates = lr_cfg.get("C_candidates", [1.0, 2.0, 4.0])

    logger.info(f"Tuning Logistic Regression across C candidates: {c_candidates}")
    best_lr, lr_tuning_history = tune_logistic_regression(
        X_train=X_train_comb_sp,
        y_train=y_train,
        X_val=X_val_comb_sp,
        y_val=y_val,
        C_candidates=c_candidates,
        solver=lr_cfg.get("solver", "lbfgs"),
        max_iter=lr_cfg.get("max_iter", 2000),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        random_state=seed,
    )

    lr_preds = best_lr.predict(X_val_comb_sp)
    lr_probs = best_lr.predict_proba(X_val_comb_sp)
    lr_metrics = compute_classification_metrics(y_val, lr_preds, lr_probs)

    lr_pipeline = Pipeline([("features", combined_features), ("clf", best_lr)])
    lr_path = out_dir / "logistic_regression.joblib"
    lr_feat_path = out_dir / "logistic_regression_features.joblib"
    save_model(lr_pipeline, str(lr_path))
    save_model(combined_features, str(lr_feat_path))

    report["models"]["logistic_regression"] = {
        "selected_hyperparameters": {"C": float(best_lr.C)},
        "tuning_history": lr_tuning_history,
        "validation_metrics": lr_metrics,
        "artifact_path": str(lr_path),
        "features_path": str(lr_feat_path),
    }
    logger.info(
        f"Logistic Regression (Best C={best_lr.C}) -> "
        f"Accuracy: {lr_metrics['accuracy']:.4f}, Macro-F1: {lr_metrics['macro_f1']:.4f}, "
        f"ROC-AUC: {lr_metrics.get('roc_auc', 0.0):.4f}"
    )

    # -------------------------------------------------------------
    # 4. Optional Linear SVM (Combined Word + Char, Calibrated)
    # -------------------------------------------------------------
    if args.include_svm:
        logger.info("\n--- [4/4] Tuning Optional Calibrated Linear SVM ---")
        svm_cfg = classical_cfg.get("linear_svm", {})
        svm_c_candidates = svm_cfg.get("C_candidates", [0.5, 1.0, 2.0])

        best_svm, svm_tuning_history = tune_linear_svm(
            X_train=X_train_comb_sp,
            y_train=y_train,
            X_val=X_val_comb_sp,
            y_val=y_val,
            C_candidates=svm_c_candidates,
            max_iter=svm_cfg.get("max_iter", 2000),
            class_weight=svm_cfg.get("class_weight", "balanced"),
            random_state=seed,
        )

        svm_preds = best_svm.predict(X_val_comb_sp)
        svm_probs = best_svm.predict_proba(X_val_comb_sp)
        svm_metrics = compute_classification_metrics(y_val, svm_preds, svm_probs)

        svm_pipeline = Pipeline([("features", combined_features), ("clf", best_svm)])
        svm_path = out_dir / "linear_svm.joblib"
        save_model(svm_pipeline, str(svm_path))

        report["models"]["linear_svm"] = {
            "selected_hyperparameters": {"C": svm_tuning_history[-1].get("C")},
            "tuning_history": svm_tuning_history,
            "validation_metrics": svm_metrics,
            "artifact_path": str(svm_path),
        }
        logger.info(
            f"Linear SVM -> Accuracy: {svm_metrics['accuracy']:.4f}, "
            f"Macro-F1: {svm_metrics['macro_f1']:.4f}"
        )

    # Save metrics JSON report
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nAll validation metrics saved to: {metrics_path}")

    # Summary Table Display
    logger.info("\n==================== VALIDATION BENCHMARK SUMMARY ====================")
    logger.info(
        f"{'Model':<25} | {'Macro-F1':<10} | {'Accuracy':<10} | {'Fake-F1':<10} | {'ROC-AUC':<10} | {'Brier':<10}"
    )
    logger.info("-" * 85)
    for m_name, m_data in report["models"].items():
        v = m_data["validation_metrics"]
        roc_str = f"{v.get('roc_auc', 0.0):.4f}" if v.get("roc_auc") is not None else "N/A"
        brier_str = (
            f"{v.get('brier_score', 0.0):.4f}" if v.get("brier_score") is not None else "N/A"
        )
        logger.info(
            f"{m_name:<25} | {v['macro_f1']:<10.4f} | {v['accuracy']:<10.4f} | "
            f"{v.get('fake_f1', 0.0):<10.4f} | {roc_str:<10} | {brier_str:<10}"
        )
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
