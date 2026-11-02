"""
ONNX Export and INT8 Quantization Script for FraudSpotter.
Converts fine-tuned DistilBERT to ONNX graph and applies INT8 dynamic quantization
for fast, lightweight CPU inference in FastAPI and Streamlit.
"""

import argparse
from pathlib import Path

from fraudspotter.common.logger import setup_logger

logger = setup_logger("fraudspotter.export_onnx")


def export_transformer_to_onnx(
    model_dir: str,
    output_onnx_path: str,
    quantized_onnx_path: str,
) -> None:
    """Export DistilBERT model to ONNX and quantize using ONNX Runtime / Optimum."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_path = Path(output_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading PyTorch model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy_text = "This is a sample sentence for ONNX tracing and export."
    inputs = tokenizer(dummy_text, return_tensors="pt")

    logger.info(f"Exporting model to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        f=str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    logger.info("ONNX graph export complete.")

    logger.info(f"Applying INT8 dynamic quantization: {quantized_onnx_path}...")
    quantize_dynamic(
        model_input=str(out_path),
        model_output=quantized_onnx_path,
        weight_type=QuantType.QInt8,
    )
    logger.info("INT8 quantization complete.")


def main():
    parser = argparse.ArgumentParser(description="Export transformer to ONNX and quantize to INT8.")
    parser.add_argument("--model_dir", default="artifacts/models/distilbert/best_model")
    parser.add_argument("--output_onnx", default="artifacts/models/distilbert_onnx/model.onnx")
    parser.add_argument(
        "--output_quantized", default="artifacts/models/distilbert_onnx/model_quantized.onnx"
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        logger.error(
            f"Trained model directory not found at {model_path}. Run training on Colab first."
        )
        return

    export_transformer_to_onnx(
        model_dir=str(model_path),
        output_onnx_path=args.output_onnx,
        quantized_onnx_path=args.output_quantized,
    )


if __name__ == "__main__":
    main()
