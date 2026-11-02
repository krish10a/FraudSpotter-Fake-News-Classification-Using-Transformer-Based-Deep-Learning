from typing import Any, Dict, List

import numpy as np
from sklearn.pipeline import Pipeline


def extract_top_lr_coefficients(
    pipeline: Pipeline,
    top_k: int = 25,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top positive and negative n-gram coefficients from a fitted TF-IDF Logistic Regression pipeline."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    feature_names = np.array(tfidf.get_feature_names_out())
    coef = clf.coef_[0]

    # Top positive features (predictive of class 1 / REAL)
    top_pos_idx = np.argsort(coef)[-top_k:][::-1]
    top_positive = [
        {"feature": str(feature_names[i]), "weight": float(coef[i])} for i in top_pos_idx
    ]

    # Top negative features (predictive of class 0 / FAKE)
    top_neg_idx = np.argsort(coef)[:top_k]
    top_negative = [
        {"feature": str(feature_names[i]), "weight": float(coef[i])} for i in top_neg_idx
    ]

    return {
        "top_predictive_real": top_positive,
        "top_predictive_fake": top_negative,
    }
