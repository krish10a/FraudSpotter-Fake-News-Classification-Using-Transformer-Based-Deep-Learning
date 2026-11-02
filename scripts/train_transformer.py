"""
Transformer Training & Optuna Tuning Script for FraudSpotter.
Designed to run on Google Colab with T4 GPU.
Fine-tunes DistilBERT-base-uncased on WELFake and optionally runs Optuna search.
"""

import argparse
from pathlib import Path

import pandas as pd

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.common.seed import set_seed
from fraudspotter.models.transformer import (
    compute_transformer_metrics,
    get_tokenizer,
    get_transformer_model,
)

logger = setup_logger("fraudspotter.train_transformer")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Google Colab GPU.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train_file", default="data/processed/welfake/train.csv")
    parser.add_argument("--val_file", default="data/processed/welfake/val.csv")
    parser.add_argument("--output_dir", default="artifacts/models/distilbert")
    parser.add_argument(
        "--use_optuna", action="store_true", help="Run Optuna hyperparameter optimization"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    # Import Colab/GPU-specific libraries
    try:
        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
    except ImportError:
        logger.error(
            "PyTorch/Transformers not installed. This script is intended to run on Google Colab GPU."
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cpu":
        logger.warning(
            "Running transformer training on CPU is not recommended. Use Google Colab GPU."
        )

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)

    t_cfg = config.get("transformer", {})
    model_name = t_cfg.get("model_name", "distilbert-base-uncased")
    max_len = t_cfg.get("max_seq_length", 512)

    tokenizer = get_tokenizer(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["content"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    logger.info("Tokenizing datasets...")
    train_dataset = Dataset.from_pandas(train_df[["content", "label"]]).map(
        tokenize_fn, batched=True
    )
    val_dataset = Dataset.from_pandas(val_df[["content", "label"]]).map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("batch_size", 16),
        per_device_eval_batch_size=t_cfg.get("eval_batch_size", 32),
        learning_rate=float(t_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(t_cfg.get("warmup_ratio", 0.1)),
        evaluation_strategy=t_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=t_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_f1_macro"),
        fp16=torch.cuda.is_available() and t_cfg.get("fp16", True),
        logging_steps=t_cfg.get("logging_steps", 100),
        seed=seed,
    )

    model = get_transformer_model(model_name=model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_transformer_metrics,
        tokenizer=tokenizer,
    )

    logger.info("Starting DistilBERT fine-tuning...")
    trainer.train()

    eval_results = trainer.evaluate()
    logger.info(f"Final Validation Metrics: {eval_results}")

    # Save final best model and tokenizer
    save_path = Path(args.output_dir) / "best_model"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"Best model & tokenizer saved to {save_path}")


if __name__ == "__main__":
    main()
