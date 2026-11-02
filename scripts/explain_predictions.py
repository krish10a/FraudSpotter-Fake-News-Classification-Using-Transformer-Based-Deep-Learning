"""
Explainability Script for FraudSpotter.
Runs:
1. Global feature coefficient analysis on Logistic Regression.
2. Local explanation via LIME.
3. Local explanation via Captum Integrated Gradients (if transformer model available).
4. Faithfulness evaluation (comprehensiveness and sufficiency).
"""

import argparse
import json
from pathlib import Path

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.evaluation.faithfulness import (
    evaluate_comprehensiveness,
    evaluate_sufficiency,
)
from fraudspotter.explainability.lime_explainer import explain_with_lime
from fraudspotter.explainability.lr_coefficients import extract_top_lr_coefficients
from fraudspotter.models.classical import load_model

logger = setup_logger("fraudspotter.explain")


def main():
    parser = argparse.ArgumentParser(description="Generate model explanations.")
    parser.add_argument("--config", default="configs/explainability_config.yaml")
    parser.add_argument(
        "--model_path", default="artifacts/models/classical/logistic_regression.joblib"
    )
    parser.add_argument(
        "--sample_text",
        default="Breaking: Scientists discover miraculous herbal tea that cures all chronic diseases.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = Path(args.model_path)

    if not model_path.is_file():
        logger.error(f"Model file not found at: {model_path}")
        return

    logger.info(f"Loading pipeline from {model_path}...")
    pipeline = load_model(str(model_path))

    # 1. Global Logistic Regression interpretation
    if "clf" in pipeline.named_steps and hasattr(pipeline.named_steps["clf"], "coef_"):
        logger.info("Extracting global Logistic Regression top features...")
        top_k = (
            config.get("global_interpretation", {})
            .get("logistic_regression", {})
            .get("top_positive_features", 25)
        )
        global_features = extract_top_lr_coefficients(pipeline, top_k=top_k)

        out_json = Path(
            config.get("global_interpretation", {})
            .get("logistic_regression", {})
            .get("save_json_path", "artifacts/metrics/lr_global_features.json")
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(global_features, f, indent=2)
        logger.info(f"Global coefficients saved to {out_json}")

    # 2. Local LIME Explanation
    logger.info("Generating LIME local attribution...")
    lime_cfg = config.get("lime", {})

    def predict_fn(texts):
        return pipeline.predict_proba(texts)

    lime_result = explain_with_lime(
        text=args.sample_text,
        predict_fn=predict_fn,
        num_features=lime_cfg.get("num_features", 10),
        num_samples=lime_cfg.get("num_samples", 300),
    )
    logger.info(f"LIME Result for sample text: {lime_result['feature_attributions']}")

    # 3. Faithfulness Evaluation
    top_tokens = [item["token"] for item in lime_result["feature_attributions"][:5]]
    comp = evaluate_comprehensiveness(args.sample_text, top_tokens, predict_fn)
    suff = evaluate_sufficiency(args.sample_text, top_tokens, predict_fn)

    logger.info(f"Faithfulness Metrics (Comprehensiveness): {comp:.4f}")
    logger.info(f"Faithfulness Metrics (Sufficiency):       {suff:.4f}")


if __name__ == "__main__":
    main()
