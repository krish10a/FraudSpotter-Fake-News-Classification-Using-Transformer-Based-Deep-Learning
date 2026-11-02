"""
Transformer Training & Sequence Length Experimentation Script for FraudSpotter.

Designed to execute on Google Colab with T4 GPU:
  - Sequence lengths: 128, 256, 384
  - Model: distilbert-base-uncased (Paired Title + Body)
  - Precision: FP16
  - Optimizer: AdamW with linear warmup schedule
  - Output: Checkpoints saved to Google Drive, best model exported
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add 'src' to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.common.seed import set_seed
from fraudspotter.models.transformer import (
    get_tokenizer,
    get_transformer_model,
)

logger = setup_logger("fraudspotter.train_transformer")


def resolve_split_path(
    explicit_path: Optional[str],
    drive_dir: Optional[str],
    split_name: str,
) -> Path:
    """Resolve split file checking explicit path, drive_dir, parquet, and csv."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p

    search_dirs = []
    if drive_dir:
        search_dirs.append(Path(drive_dir))
    search_dirs.extend(
        [
            Path("/content/drive/MyDrive/FraudSpotter/data/processed/welfake_v1"),
            Path("/content/drive/MyDrive/FraudSpotter_Data"),
            Path("data/processed/welfake_v1"),
            Path("data/processed/welfake"),
        ]
    )

    for s_dir in search_dirs:
        for alias in [split_name, "validation" if split_name == "val" else "val"]:
            for ext in [".parquet", ".pq", ".csv"]:
                candidate = s_dir / f"{alias}{ext}"
                if candidate.is_file():
                    return candidate

    return Path(
        f"/content/drive/MyDrive/FraudSpotter/data/processed/welfake_v1/{split_name}.parquet"
    )


def compute_metrics_fn(eval_pred: Any) -> Dict[str, float]:
    """Compute comprehensive classification metrics for HuggingFace Trainer."""
    import numpy as np
    from scipy.special import softmax
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = softmax(logits, axis=1)[:, 1]

    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    fake_f1 = float(f1_score(labels, preds, pos_label=1))
    real_f1 = float(f1_score(labels, preds, pos_label=0))
    prec = float(precision_score(labels, preds, average="macro", zero_division=0))
    rec = float(recall_score(labels, preds, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(labels, preds))

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except Exception:
        roc_auc = 0.5

    return {
        "accuracy": acc,
        "f1_macro": macro_f1,
        "eval_f1_macro": macro_f1,
        "fake_f1": fake_f1,
        "real_f1": real_f1,
        "macro_precision": prec,
        "macro_recall": rec,
        "mcc": mcc,
        "roc_auc": roc_auc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on Google Colab T4 GPU with sequence length experiments."
    )
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train_file", default=None, help="Path to train.parquet or train.csv")
    parser.add_argument(
        "--val_file", default=None, help="Path to validation.parquet or val.parquet"
    )
    parser.add_argument(
        "--drive_dir",
        default=None,
        help="Optional Google Drive base directory",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        choices=[128, 256, 384],
        help="Sequence length experiment: 128, 256, or 384",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for checkpoints (defaults to Drive if available)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Per-device train batch size override",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of fine-tuning epochs override",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    # Check GPU availability
    try:
        import pandas as pd
        import torch
        from datasets import Dataset
        from transformers import (
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        logger.error(
            f"Missing deep learning dependency: {e}. "
            "Transformer training is designated for Google Colab T4 GPU."
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target compute device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        logger.warning(
            "MANDATORY RULE NOTICE: Transformer training should be run on Google Colab T4 GPU, not on local CPU."
        )

    # Resolve sequence length configuration
    t_cfg = config.get("transformer", {})
    seq_len = args.seq_len
    exp_cfg = t_cfg.get("sequence_length_experiments", {}).get(seq_len, {})

    batch_size = args.batch_size or exp_cfg.get("batch_size", 16 if seq_len <= 256 else 8)
    eval_batch_size = exp_cfg.get("eval_batch_size", batch_size * 2)
    grad_accum = exp_cfg.get("gradient_accumulation_steps", 1 if seq_len <= 256 else 2)
    lr = float(exp_cfg.get("learning_rate", 2e-5))
    epochs = args.epochs or exp_cfg.get("num_train_epochs", 3)
    model_name = t_cfg.get("model_name", "distilbert/distilbert-base-uncased")

    # Resolve output directory
    default_out = (
        f"/content/drive/MyDrive/FraudSpotter/models/distilbert_seq{seq_len}"
        if Path("/content/drive/MyDrive").exists()
        else f"artifacts/models/distilbert_seq{seq_len}"
    )
    out_dir = Path(args.output_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve and Load Data
    train_path = resolve_split_path(args.train_file, args.drive_dir, "train")
    val_path = resolve_split_path(args.val_file, args.drive_dir, "val")

    logger.info(f"Loading training data from: {train_path}")
    if train_path.suffix.lower() in [".parquet", ".pq"]:
        train_df = pd.read_parquet(train_path, columns=["title", "body", "label"])
        val_df = pd.read_parquet(val_path, columns=["title", "body", "label"])
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

    # Fill nulls
    train_df["title"] = train_df["title"].fillna("").astype(str)
    train_df["body"] = (
        train_df["body"].fillna("").astype(str)
        if "body" in train_df.columns
        else train_df["text"].fillna("").astype(str)
    )
    val_df["title"] = val_df["title"].fillna("").astype(str)
    val_df["body"] = (
        val_df["body"].fillna("").astype(str)
        if "body" in val_df.columns
        else val_df["text"].fillna("").astype(str)
    )

    logger.info(
        f"Preparing Hugging Face datasets: Train={len(train_df):,} rows, Val={len(val_df):,} rows"
    )
    logger.info(
        f"Experiment Settings: seq_len={seq_len}, batch_size={batch_size}, grad_accum={grad_accum}, lr={lr}, epochs={epochs}"
    )

    tokenizer = get_tokenizer(model_name)

    # Paired tokenization: [CLS] title [SEP] body [SEP]
    def tokenize_paired(batch):
        return tokenizer(
            batch["title"],
            batch["body"],
            truncation="longest_first",
            max_length=seq_len,
        )

    train_ds = Dataset.from_pandas(
        train_df[["title", "body", "label"]], preserve_index=False
    ).map(tokenize_paired, batched=True)
    val_ds = Dataset.from_pandas(
        val_df[["title", "body", "label"]], preserve_index=False
    ).map(tokenize_paired, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    model = get_transformer_model(
        model_name=model_name,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
    )

    # Safe eval_strategy handling across transformers versions
    eval_strat = t_cfg.get("evaluation_strategy", t_cfg.get("eval_strategy", "epoch"))
    training_args_kwargs = {
        "output_dir": str(out_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "weight_decay": float(t_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(t_cfg.get("warmup_ratio", 0.1)),
        "save_strategy": t_cfg.get("save_strategy", "epoch"),
        "save_total_limit": 2,
        "load_best_model_at_end": t_cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": t_cfg.get("metric_for_best_model", "eval_f1_macro"),
        "greater_is_better": True,
        "fp16": (device == "cuda" and t_cfg.get("fp16", True)),
        "logging_steps": t_cfg.get("logging_steps", 100),
        "report_to": "none",
        "seed": seed,
    }
    import inspect

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_args_kwargs["eval_strategy"] = eval_strat
    else:
        training_args_kwargs["evaluation_strategy"] = eval_strat

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info(f"--- Launching DistilBERT Fine-Tuning (SeqLen={seq_len}) ---")
    trainer.train()

    eval_results = trainer.evaluate()
    logger.info(f"Final Validation Results (SeqLen={seq_len}): {eval_results}")

    # Save best model, tokenizer, and metrics
    best_model_dir = out_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    metrics_file = out_dir / f"validation_metrics_seq{seq_len}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "validation_metrics": eval_results,
            },
            f,
            indent=2,
        )

    logger.info(f"Best model & tokenizer successfully saved to: {best_model_dir}")
    logger.info(f"Validation metrics recorded at: {metrics_file}")


if __name__ == "__main__":
    main()
