import re
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(
    text: str,
    lowercase: bool = True,
    strip_html: bool = True,
    normalize_urls: bool = True,
) -> str:
    """Preprocess raw text by stripping HTML tags, normalizing URLs, and cleaning whitespace."""
    if not isinstance(text, str):
        return ""

    if strip_html:
        text = re.sub(r"<.*?>", " ", text)

    if normalize_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    if lowercase:
        text = text.lower()

    # Normalize excessive whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataset_splits(
    df: pd.DataFrame,
    text_col: str = "text",
    title_col: str = "title",
    label_col: str = "label",
    train_size: float = 0.80,
    val_size: float = 0.10,
    test_size: float = 0.10,
    combine_title_text: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean data, combine title/text if requested, and perform stratified train/val/test splits."""
    working_df = df.copy()

    # Fill nulls
    if title_col in working_df.columns:
        working_df[title_col] = working_df[title_col].fillna("").astype(str)
    else:
        working_df[title_col] = ""

    if text_col in working_df.columns:
        working_df[text_col] = working_df[text_col].fillna("").astype(str)
    else:
        working_df[text_col] = ""

    # Drop rows without label or invalid label
    working_df = working_df.dropna(subset=[label_col])
    working_df[label_col] = working_df[label_col].astype(int)

    # Combine title and text
    if combine_title_text:
        working_df["content"] = working_df.apply(
            lambda row: clean_text(f"{row[title_col]} {row[text_col]}"), axis=1
        )
    else:
        working_df["content"] = working_df[text_col].apply(clean_text)

    # Filter out empty content
    working_df = working_df[working_df["content"].str.len() > 10].reset_index(drop=True)

    # Stratified split: Train vs Temp (val + test)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        working_df,
        test_size=temp_size,
        random_state=random_state,
        stratify=working_df[label_col],
    )

    # Split Temp into Val and Test
    val_relative_ratio = val_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_state,
        stratify=temp_df[label_col],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

