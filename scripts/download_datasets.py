"""
Dataset Downloader Script for FraudSpotter.
Designed to be executed on Google Colab or remote environments with high bandwidth.
Fetches WELFake, ISOT, and LIAR directly from online repositories without local laptop upload.
Supports '--sample' flag for local fixture generation.
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# Ensure 'src' is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress indication."""
    print(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest_path))
    print(f"Saved: {dest_path}")


def download_welfake(raw_dir: Path) -> None:
    """Download WELFake dataset."""
    dest = raw_dir / "welfake.csv"
    if dest.exists():
        print(f"WELFake already exists at {dest}")
        return

    # Zenodo direct URL or Kaggle
    url = "https://zenodo.org/record/4561253/files/WELFake_Dataset.csv?download=1"
    try:
        download_file(url, dest)
    except Exception as e:
        print(f"Direct download failed ({e}). If using Kaggle, run:")
        print("  kaggle datasets download -d saurabhshahane/fake-news-classification -p data/raw/")


def download_isot(raw_dir: Path) -> None:
    """Download ISOT dataset via Kaggle."""
    isot_dir = raw_dir / "isot"
    isot_dir.mkdir(parents=True, exist_ok=True)
    print("For ISOT dataset, download via Kaggle API:")
    print(
        "  kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw/isot/"
    )
    print("  unzip data/raw/isot/fake-and-real-news-dataset.zip -d data/raw/isot/")


def download_liar(raw_dir: Path) -> None:
    """Download LIAR dataset directly from UCSB repository."""
    liar_dir = raw_dir / "liar"
    liar_dir.mkdir(parents=True, exist_ok=True)
    zip_path = liar_dir / "liar_dataset.zip"

    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    try:
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(liar_dir)
        print(f"Extracted LIAR dataset to {liar_dir}")
    except Exception as e:
        print(f"Could not automatically download LIAR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets directly on Colab or create local samples."
    )
    parser.add_argument("--dataset", choices=["welfake", "isot", "liar", "all"], default="all")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate small local sample fixture instead of downloading full dataset",
    )
    parser.add_argument("--output_dir", default="data/raw")
    args = parser.parse_args()

    raw_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        from fraudspotter.data.loader import create_synthetic_sample_data

        print("Generating small local sample fixture for offline testing...")
        sample_df = create_synthetic_sample_data(num_samples=100)
        sample_path = raw_dir / "sample_welfake.csv"
        sample_df.to_csv(sample_path, index=False)
        print(f"Sample fixture created at: {sample_path}")
        return

    print("=== Downloading Datasets directly to environment ===")
    if args.dataset in ["welfake", "all"]:
        download_welfake(raw_dir)
    if args.dataset in ["isot", "all"]:
        download_isot(raw_dir)
    if args.dataset in ["liar", "all"]:
        download_liar(raw_dir)
    print("=== Download process complete ===")


if __name__ == "__main__":
    main()
