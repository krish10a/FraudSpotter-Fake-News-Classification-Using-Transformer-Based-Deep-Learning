from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Majority class baseline classifier acting as the empirical performance floor."""

    def __init__(self):
        self.majority_class_: int = 0
        self.class_priors_: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
        self.classes_: np.ndarray = np.array([0, 1], dtype=int)

    def fit(self, X: Any, y: np.ndarray) -> "MajorityClassifier":
        y_arr = np.asarray(y, dtype=int)
        counts = np.bincount(y_arr, minlength=2)
        self.majority_class_ = int(np.argmax(counts))
        self.class_priors_ = (counts / len(y_arr)).astype(np.float32)
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.majority_class_, dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.tile(self.class_priors_, (n_samples, 1))


def build_word_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int = 100000,
    sublinear_tf: bool = True,
    dtype: np.dtype = np.float32,
) -> TfidfVectorizer:
    """Build word-level TF-IDF Vectorizer using float32 precision."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        dtype=dtype,
    )


def build_char_vectorizer(
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 3,
    max_df: float = 0.95,
    max_features: int = 50000,
    sublinear_tf: bool = True,
    dtype: np.dtype = np.float32,
) -> TfidfVectorizer:
    """Build character n-gram TF-IDF Vectorizer using float32 precision."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        dtype=dtype,
    )


def build_combined_feature_union(
    word_params: Optional[Dict[str, Any]] = None,
    char_params: Optional[Dict[str, Any]] = None,
) -> FeatureUnion:
    """Build combined word + character TF-IDF FeatureUnion in float32."""
    w_params = word_params or {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 100000,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    c_params = char_params or {
        "ngram_range": (3, 5),
        "min_df": 3,
        "max_df": 0.95,
        "max_features": 50000,
        "sublinear_tf": True,
        "dtype": np.float32,
    }

    word_vec = build_word_vectorizer(**w_params)
    char_vec = build_char_vectorizer(**c_params)

    return FeatureUnion(
        [
            ("word_tfidf", word_vec),
            ("char_tfidf", char_vec),
        ]
    )


def tune_multinomial_nb(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    alpha_candidates: List[float],
    fit_prior: bool = True,
) -> Tuple[MultinomialNB, List[Dict[str, Any]]]:
    """Tune Multinomial Naive Bayes alpha parameter using validation Macro-F1 only."""
    best_model: Optional[MultinomialNB] = None
    best_f1: float = -1.0
    history: List[Dict[str, Any]] = []

    for alpha in alpha_candidates:
        nb = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        nb.fit(X_train, y_train)
        val_preds = nb.predict(X_val)
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))

        record = {"alpha": float(alpha), "val_macro_f1": val_f1}
        history.append(record)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = nb

    assert best_model is not None
    return best_model, history


def tune_logistic_regression(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    C_candidates: List[float],
    solver: str = "lbfgs",
    solver_candidates: Optional[List[str]] = None,
    max_iter: int = 2000,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Tuple[LogisticRegression, float, str, List[Dict[str, Any]]]:
    """Tune Logistic Regression across C and solver candidates using validation Macro-F1 only.

    Returns:
        Tuple of (best_model, best_C, best_solver, history)
    """
    best_model: Optional[LogisticRegression] = None
    best_f1: float = -1.0
    best_C: float = float(C_candidates[0])
    best_solver: str = solver
    history: List[Dict[str, Any]] = []

    solvers = solver_candidates or [solver]

    for s in solvers:
        for C in C_candidates:
            clf = LogisticRegression(
                C=C,
                solver=s,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
            )
            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            val_f1 = float(f1_score(y_val, val_preds, average="macro"))

            record = {"solver": s, "C": float(C), "val_macro_f1": val_f1}
            history.append(record)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = clf
                best_C = float(C)
                best_solver = s

    assert best_model is not None
    setattr(best_model, "best_C_", best_C)
    setattr(best_model, "best_solver_", best_solver)
    return best_model, best_C, best_solver, history


def tune_linear_svm(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    C_candidates: List[float],
    max_iter: int = 2000,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Tuple[Any, float, List[Dict[str, Any]]]:
    """Tune LinearSVC with calibrated probabilities using validation Macro-F1 only.

    Returns:
        Tuple of (best_calibrated_model, best_C, tuning_history)
    """
    best_model: Optional[Any] = None
    best_f1: float = -1.0
    best_C: float = float(C_candidates[0])
    history: List[Dict[str, Any]] = []

    for C in C_candidates:
        base_svm = LinearSVC(
            C=C,
            loss="squared_hinge",
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )
        calibrated_clf = CalibratedClassifierCV(estimator=base_svm, cv=3)
        calibrated_clf.fit(X_train, y_train)
        val_preds = calibrated_clf.predict(X_val)
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))

        record = {"C": float(C), "val_macro_f1": val_f1}
        history.append(record)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = calibrated_clf
            best_C = float(C)

    assert best_model is not None
    setattr(best_model, "best_C_", best_C)
    return best_model, best_C, history


def build_classical_pipeline(
    model_type: str = "logistic_regression",
    tfidf_params: Optional[Dict[str, Any]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build an end-to-end inference Pipeline combining feature extraction and classifier."""
    m_type = model_type.lower()

    if m_type in ["majority", "majority_baseline"]:
        return Pipeline(
            [
                ("features", build_word_vectorizer(max_features=10)),
                ("clf", MajorityClassifier()),
            ]
        )

    elif m_type in ["multinomial_nb", "naive_bayes", "nb"]:
        vectorizer = build_word_vectorizer(**(tfidf_params or {}))
        clf_params = model_params or {"alpha": 0.1, "fit_prior": True}
        clf = MultinomialNB(**clf_params)
        return Pipeline([("features", vectorizer), ("clf", clf)])

    elif m_type in ["logistic_regression", "logreg"]:
        vectorizer = build_combined_feature_union()
        clf_params = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
        }
        if model_params:
            clf_params.update(model_params)
        clf = LogisticRegression(**clf_params)
        return Pipeline([("features", vectorizer), ("clf", clf)])

    elif m_type in ["linear_svm", "svm"]:
        vectorizer = build_combined_feature_union()
        params = {
            "C": 1.0,
            "loss": "squared_hinge",
            "max_iter": 2000,
            "random_state": 42,
        }
        if model_params:
            params.update(model_params)
        use_calibration = params.pop("use_calibration", True)
        calib_cv = params.pop("cv", 3)
        base_svm = LinearSVC(**params)
        if use_calibration:
            clf = CalibratedClassifierCV(estimator=base_svm, cv=calib_cv)
        else:
            clf = base_svm
        return Pipeline([("features", vectorizer), ("clf", clf)])

    else:
        raise ValueError(f"Unsupported classical model type: {model_type}")


def predict_pipeline(
    pipeline: Pipeline, texts: Union[List[str], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict class labels and calibrated probabilities (shape: [N, 2])."""
    preds = pipeline.predict(texts)

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)
    elif hasattr(pipeline.named_steps["clf"], "predict_proba"):
        feat_step = pipeline.steps[0][0]
        feats = pipeline.named_steps[feat_step].transform(texts)
        probs = pipeline.named_steps["clf"].predict_proba(feats)
    else:
        # Fallback using decision function sigmoid calibration
        feat_step = pipeline.steps[0][0]
        feats = pipeline.named_steps[feat_step].transform(texts)
        dec = pipeline.named_steps["clf"].decision_function(feats)
        p1 = 1 / (1 + np.exp(-dec))
        probs = np.column_stack([1 - p1, p1])

    return preds, probs


def save_model(pipeline: Any, filepath: str) -> None:
    """Serialize model artifact to disk using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(filepath: str) -> Any:
    """Load serialized model artifact from disk."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at: {filepath}")
    return joblib.load(path)
