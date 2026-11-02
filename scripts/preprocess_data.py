"""
Data Preprocessing Script for FraudSpotter.
Cleans raw text, handles HTML/URL removal, combines title and text,
and generates stratified Train, Validation, and Test splits.
"""

import argparse
from pathlib import Path

import pandas as pd

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.common.seed import set_seed
from fraudspotter.data.preprocessor import prepare_dataset_splits

logger = setup_logger("fraudspotter.preprocess")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split datasets.")
    parser.add_argument(
        "--config", default="configs/data_config.yaml", help="Path to data config YAML"
    )
    parser.add_argument("--input_file", default="data/raw/welfake.csv", help="Input raw CSV path")
    parser.add_argument(
        "--output_dir",
        default="data/processed/welfake",
        help="Output directory for processed splits",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    input_path = Path(args.input_file)
    if not input_path.is_file():
        logger.error(f"Input file not found at: {input_path}")
        logger.info(
            "Tip: Run `python scripts/download_datasets.py --sample` to create a sample fixture for local testing."
        )
        return

    logger.info(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} raw records.")

    dataset_cfg = config.get("datasets", {}).get("welfake", {})
    text_col = dataset_cfg.get("text_column", "text")
    title_col = dataset_cfg.get("title_column", "title")
    label_col = dataset_cfg.get("label_column", "label")
    ratios = dataset_cfg.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})

    logger.info("Cleaning text and creating stratified splits...")
    train_df, val_df, test_df = prepare_dataset_splits(
        df=df,
        text_col=text_col,
        title_col=title_col,
        label_col=label_col,
        train_size=ratios["train"],
        val_size=ratios["val"],
        test_size=ratios["test"],
        random_state=seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Splits saved successfully to {out_dir}:")
    logger.info(f"  - Train: {len(train_df)} samples ({train_path})")
    logger.info(f"  - Val:   {len(val_df)} samples ({val_path})")
    logger.info(f"  - Test:  {len(test_df)} samples ({test_path})")


if __name__ == "__main__":
    main()
