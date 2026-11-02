from typing import Callable, List

import numpy as np


def evaluate_comprehensiveness(
    text: str,
    top_tokens: List[str],
    predict_fn: Callable[[List[str]], np.ndarray],
    target_class: int = 1,
) -> float:
    """Measure comprehensiveness: drop in confidence when high-attribution tokens are removed."""
    orig_prob = predict_fn([text])[0][target_class]

    # Remove top tokens
    modified_text = text
    for tok in top_tokens:
        modified_text = modified_text.replace(tok, " ")

    modified_prob = predict_fn([modified_text])[0][target_class]
    return float(orig_prob - modified_prob)


def evaluate_sufficiency(
    text: str,
    top_tokens: List[str],
    predict_fn: Callable[[List[str]], np.ndarray],
    target_class: int = 1,
) -> float:
    """Measure sufficiency: drop in confidence when keeping only high-attribution tokens."""
    orig_prob = predict_fn([text])[0][target_class]
    isolated_text = " ".join(top_tokens)
    isolated_prob = predict_fn([isolated_text])[0][target_class]
    return float(orig_prob - isolated_prob)
