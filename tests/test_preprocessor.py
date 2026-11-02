from fraudspotter.data.loader import create_synthetic_sample_data
from fraudspotter.data.preprocessor import clean_text, prepare_dataset_splits


def test_clean_text():
    raw = "<div>BREAKING: Visit https://example.com/fake-news for shocking updates!   </div>\n\n"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "https://" not in cleaned
    assert "[url]" in cleaned
    assert "breaking:" in cleaned
    assert "  " not in cleaned


def test_prepare_dataset_splits():
    df = create_synthetic_sample_data(num_samples=40)
    train_df, val_df, test_df = prepare_dataset_splits(
        df=df,
        text_col="text",
        title_col="title",
        label_col="label",
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(train_df) == 32
    assert len(val_df) == 4
    assert len(test_df) == 4
    assert "content" in train_df.columns
    assert set(train_df["label"].unique()).issubset({0, 1})
