from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_classical_pipeline(
    model_type: str = "logistic_regression",
    tfidf_params: Optional[Dict[str, Any]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a Scikit-Learn Pipeline combining TfidfVectorizer and a classifier."""
    if tfidf_params is None:
        tfidf_params = {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
        }

    vectorizer = TfidfVectorizer(**tfidf_params)
    m_type = model_type.lower()

    if m_type in ["logistic_regression", "logreg"]:
        params = {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        }
        if model_params:
            params.update(model_params)
        clf = LogisticRegression(**params)

    elif m_type in ["multinomial_nb", "naive_bayes", "nb"]:
        params = {"alpha": 0.1, "fit_prior": True}
        if model_params:
            params.update(model_params)
        clf = MultinomialNB(**params)

    elif m_type in ["linear_svm", "svm"]:
        params = {
            "C": 1.0,
            "loss": "squared_hinge",
            "max_iter": 2000,
            "random_state": 42,
        }
        if model_params:
            params.update(model_params)
        use_calibration = params.pop("use_calibration", False)
        calib_cv = params.pop("cv", 3)
        base_svm = LinearSVC(**params)
        if use_calibration:
            clf = CalibratedClassifierCV(estimator=base_svm, cv=calib_cv)
        else:
            clf = base_svm

    else:
        raise ValueError(f"Unsupported classical model type: {model_type}")

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    return pipeline


def predict_pipeline(pipeline: Pipeline, texts: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Predict class labels and calibrated probabilities."""
    preds = pipeline.predict(texts)

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)
    elif hasattr(pipeline.named_steps["clf"], "predict_proba"):
        probs = pipeline.named_steps["clf"].predict_proba(
            pipeline.named_steps["tfidf"].transform(texts)
        )
    else:
        # Fallback decision function
        dec = pipeline.decision_function(texts)
        p1 = 1 / (1 + np.exp(-dec))
        probs = np.column_stack([1 - p1, p1])

    return preds, probs


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Serialize pipeline model artifact to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(filepath: str) -> Pipeline:
    """Load serialized pipeline model from disk."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at: {filepath}")
    return joblib.load(path)
