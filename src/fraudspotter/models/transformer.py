from typing import Any, Dict, Optional


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Load DistilBERT tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def get_transformer_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
):
    """Load DistilBERT Sequence Classification model."""
    from transformers import AutoModelForSequenceClassification

    if id2label is None:
        id2label = {0: "REAL", 1: "FAKE"}
    if label2id is None:
        label2id = {"REAL": 0, "FAKE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def compute_transformer_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute evaluation metrics for Hugging Face Trainer."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
