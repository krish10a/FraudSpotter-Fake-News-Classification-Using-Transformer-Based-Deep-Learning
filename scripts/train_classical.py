"""
Classical Baselines Training Script for FraudSpotter.
Trains TF-IDF + Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
Saves models to artifacts/models/classical/.
"""

import argparse
from pathlib import Path

import pandas as pd

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.common.seed import set_seed
from fraudspotter.evaluation.metrics import compute_classification_metrics
from fraudspotter.models.classical import build_classical_pipeline, save_model

logger = setup_logger("fraudspotter.train_classical")


def main():
    parser = argparse.ArgumentParser(description="Train classical ML baseline models.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train_file", default="data/processed/welfake/train.csv")
    parser.add_argument("--val_file", default="data/processed/welfake/val.csv")
    parser.add_argument(
        "--model_type",
        choices=["logistic_regression", "multinomial_nb", "linear_svm", "all"],
        default="all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    if not train_path.is_file() or not val_path.is_file():
        logger.error("Training or validation file not found. Ensure preprocessing has run.")
        return

    logger.info("Loading train and validation datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train, y_train = train_df["content"].tolist(), train_df["label"].values
    X_val, y_val = val_df["content"].tolist(), val_df["label"].values

    models_to_train = (
        ["logistic_regression", "multinomial_nb", "linear_svm"]
        if args.model_type == "all"
        else [args.model_type]
    )

    tfidf_params = config.get("classical", {}).get("tfidf", {})

    for m_type in models_to_train:
        logger.info(f"--- Training {m_type} ---")
        m_params = config.get("classical", {}).get(m_type, {})
        save_path = m_params.get("output_path", f"artifacts/models/classical/{m_type}.joblib")

        pipeline = build_classical_pipeline(
            model_type=m_type,
            tfidf_params=tfidf_params,
            model_params=m_params,
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        probs = None
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_val)

        metrics = compute_classification_metrics(y_true=y_val, y_pred=preds, y_prob=probs)
        logger.info(f"Validation Results for {m_type}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        save_model(pipeline, save_path)
        logger.info(f"Saved {m_type} pipeline to {save_path}")


if __name__ == "__main__":
    main()
