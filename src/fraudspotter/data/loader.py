from pathlib import Path
from typing import Optional, Tuple
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
        return pd.read_csv(path, encoding=encoding)

    elif dataset_name_lower == "isot":
        return pd.read_csv(path, encoding=encoding)

    elif dataset_name_lower == "liar":
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
        return pd.read_csv(path, sep="\t", header=None, names=cols, encoding=encoding)

    else:
        return pd.read_csv(path, encoding=encoding)


def load_split_data(
    file_path: str,
    text_col: Optional[str] = None,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a frozen processed train, validation, or test split from Parquet or CSV.

    Ensures the presence of 'content' and integer binary labels conforming to:
    REAL = 0, FAKE = 1.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif suffix in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        # Fallback: attempt parquet first, then csv
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path)

    # Resolve text content column
    if text_col and text_col in df.columns:
        content_series = df[text_col].fillna("").astype(str)
    elif "content" in df.columns:
        content_series = df["content"].fillna("").astype(str)
    elif "title" in df.columns and "text" in df.columns:
        # Combine title and text if no pre-generated content column exists
        title_str = df["title"].fillna("").astype(str)
        text_str = df["text"].fillna("").astype(str)
        content_series = (title_str + " " + text_str).str.strip()
    elif "text" in df.columns:
        content_series = df["text"].fillna("").astype(str)
    else:
        raise ValueError(
            f"Unable to find text column in {file_path}. Columns present: {list(df.columns)}"
        )

    # Resolve label column
    target_label_col = label_col
    if target_label_col not in df.columns:
        if "internal_label" in df.columns:
            target_label_col = "internal_label"
        elif "labels" in df.columns:
            target_label_col = "labels"
        else:
            raise ValueError(
                f"Label column '{label_col}' not found in {file_path}. Columns: {list(df.columns)}"
            )

    labels = df[target_label_col].astype(int)

    # Validate internal label binary values
    unique_labels = set(labels.unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Labels in {file_path} must be binary {0, 1} (REAL=0, FAKE=1). Found: {unique_labels}"
        )

    clean_df = pd.DataFrame({
        "content": content_series,
        "label": labels,
    })

    # Drop any empty or invalid rows
    clean_df = clean_df.dropna(subset=["content", "label"])
    clean_df = clean_df[clean_df["content"].str.len() > 0].reset_index(drop=True)

    return clean_df, clean_df["label"]


def create_synthetic_sample_data(num_samples: int = 50) -> pd.DataFrame:
    """Generate deterministic synthetic news data for unit and integration testing.

    Conforms strictly to internal label convention:
    REAL = 0, FAKE = 1.
    """
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
            label = 1  # Internal label: 1 = FAKE
        else:
            title = real_titles[(i // 2) % len(real_titles)]
            text = f"{title} Independent researchers and institutional authorities confirmed the statistical findings today."
            label = 0  # Internal label: 0 = REAL

        records.append({
            "title": title,
            "text": text,
            "content": f"{title} {text}",
            "label": label,
        })

    return pd.DataFrame(records)
