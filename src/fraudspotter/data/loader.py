from pathlib import Path
from typing import Optional
import pandas as pd


def load_raw_dataset(
    dataset_name: str,
    file_path: str,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load raw dataset CSV or TSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "welfake":
        df = pd.read_csv(path, encoding=encoding)
        # Ensure expected columns
        # Typical WELFake columns: index, title, text, label
        return df

    elif dataset_name_lower == "isot":
        # ISOT usually consists of True.csv and Fake.csv or pre-merged CSV
        df = pd.read_csv(path, encoding=encoding)
        return df

    elif dataset_name_lower == "liar":
        # LIAR is tab-separated with no headers in original distribution
        cols = [
            "id",
            "label",
            "statement",
            "subjects",
            "speaker",
            "job_title",
            "state_info",
            "party_affiliation",
            "barely_true_counts",
            "false_counts",
            "half_true_counts",
            "mostly_true_counts",
            "pants_on_fire_counts",
            "context",
        ]
        df = pd.read_csv(path, sep="\t", header=None, names=cols, encoding=encoding)
        return df

    else:
        return pd.read_csv(path, encoding=encoding)


def create_synthetic_sample_data(num_samples: int = 50) -> pd.DataFrame:
    """Generate deterministic synthetic news data for unit and integration testing."""
    records = []
    fake_titles = [
        "Shocking Discovery: Scientists Prove Earth is Completely Flat!",
        "Secret Government Aliens Discovered Living Under Area 51!",
        "Miracle Drink Cures All Diseases in 24 Hours, Doctors Speechless!",
        "Famous Politician Caught Stealing Gold from National Treasury!",
    ]
    real_titles = [
        "Federal Reserve Announces Updated Interest Rate Policy Decision",
        "New Climate Study Analyzes Global Precipitation Patterns",
        "Tech Company Unveils New High-Efficiency Solar Panel Array",
        "Public Health Officials Release Seasonal Flu Vaccination Guidance",
    ]

    for i in range(num_samples):
        if i % 2 == 0:
            title = fake_titles[(i // 2) % len(fake_titles)]
            text = f"{title} Unverified sources claim that shocking revelations were hidden from the public for centuries."
            label = 0  # Fake
        else:
            title = real_titles[(i // 2) % len(real_titles)]
            text = f"{title} Independent researchers and institutional authorities confirmed the statistical findings today."
            label = 1  # Real

        records.append({"title": title, "text": text, "label": label})

    return pd.DataFrame(records)

