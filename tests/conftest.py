import json
from pathlib import Path

import pandas as pd
import pytest

from fraudspotter.models.classical import build_classical_pipeline


@pytest.fixture
def sample_news_data():
    """Load test fixtures from sample_news.json."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_news.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "content" not in df.columns:
        df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    return df


@pytest.fixture
def trained_dummy_classical_pipeline(sample_news_data):
    """Build and fit a fast TF-IDF Logistic Regression pipeline on fixture data."""
    df = sample_news_data
    texts = (df["title"] + " " + df["text"]).tolist()
    labels = df["label"].values

    pipeline = build_classical_pipeline("logistic_regression")
    pipeline.fit(texts, labels)
    return pipeline
