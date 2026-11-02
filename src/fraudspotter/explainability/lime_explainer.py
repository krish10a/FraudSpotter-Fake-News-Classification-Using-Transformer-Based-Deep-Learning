from typing import Any, Callable, Dict, List

import numpy as np


def explain_with_lime(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    num_features: int = 15,
    num_samples: int = 500,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """Generate local token attribution explanation using LIME."""
    from lime.lime_text import LimeTextExplainer

    if class_names is None:
        class_names = ["FAKE", "REAL"]

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    feature_weights = [
        {"token": str(token), "weight": float(weight)} for token, weight in exp.as_list()
    ]

    return {
        "method": "LIME",
        "feature_attributions": feature_weights,
        "class_probabilities": exp.predict_proba.tolist(),
    }
