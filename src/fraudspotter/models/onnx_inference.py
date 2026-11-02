from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class ONNXClassifier:
    """Fast ONNX Runtime inference engine for quantized DistilBERT classification."""

    def __init__(
        self,
        onnx_model_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 512,
    ):
        path = Path(onnx_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")

        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Initialize ONNX Runtime Session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.input_names = [input_node.name for input_node in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        self.id2label = {0: "FAKE", 1: "REAL"}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute stable softmax values."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, texts: List[str]) -> Tuple[List[int], List[List[float]]]:
        """Predict binary labels and probabilities for input texts."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )

        ort_inputs = {}
        for name in self.input_names:
            if name in encoded:
                # Ensure int64
                ort_inputs[name] = encoded[name].astype(np.int64)

        ort_outs = self.session.run([self.output_name], ort_inputs)
        logits = ort_outs[0]
        probs = self._softmax(logits)
        predicted_classes = np.argmax(probs, axis=1).tolist()

        return predicted_classes, probs.tolist()

    def predict_one(self, text: str) -> Dict[str, Any]:
        """Convenience method for single text prediction."""
        classes, probs = self.predict([text])
        pred_class = classes[0]
        prob_fake = float(probs[0][0])
        prob_real = float(probs[0][1])

        return {
            "prediction_label": self.id2label[pred_class],
            "prediction_class": pred_class,
            "probability_fake": prob_fake,
            "probability_real": prob_real,
            "confidence": max(prob_fake, prob_real),
        }
