import numpy as np
import pandas as pd
import pytest

from fraudspotter.data.loader import load_split_data
from fraudspotter.models.classical import (
    MajorityClassifier,
    Pipeline,
    build_char_vectorizer,
    build_classical_pipeline,
    build_combined_feature_union,
    build_word_vectorizer,
    load_model,
    predict_pipeline,
    save_model,
    tune_linear_svm,
    tune_logistic_regression,
    tune_multinomial_nb,
)


def test_majority_baseline_classifier():
    y = np.array([0, 1, 1, 1, 0, 1], dtype=int)  # Majority class is 1 (FAKE)
    X = ["sample news"] * len(y)

    clf = MajorityClassifier()
    clf.fit(X, y)

    assert clf.majority_class_ == 1
    assert np.allclose(clf.class_priors_, [2 / 6, 4 / 6])

    preds = clf.predict(["new text 1", "new text 2"])
    assert np.array_equal(preds, [1, 1])

    probs = clf.predict_proba(["text 1", "text 2"])
    assert probs.shape == (2, 2)
    assert np.allclose(probs[:, 1], 4 / 6)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_vectorizer_float32_sparse_matrices(sample_news_data):
    texts = sample_news_data["content"].tolist()

    word_vec = build_word_vectorizer(ngram_range=(1, 2), min_df=1, max_features=100)
    X_word = word_vec.fit_transform(texts)
    assert X_word.dtype == np.float32

    char_vec = build_char_vectorizer(ngram_range=(3, 4), min_df=1, max_features=100)
    X_char = char_vec.fit_transform(texts)
    assert X_char.dtype == np.float32

    combined_union = build_combined_feature_union(
        word_params={"ngram_range": (1, 2), "min_df": 1, "max_features": 50},
        char_params={"ngram_range": (3, 4), "min_df": 1, "max_features": 50},
    )
    X_comb = combined_union.fit_transform(texts)
    assert X_comb.dtype == np.float32
    assert X_comb.shape[1] > 0


def test_validation_only_tuning(sample_news_data):
    df = sample_news_data
    texts = df["content"].tolist()
    labels = df["label"].values

    word_vec = build_word_vectorizer(ngram_range=(1, 2), min_df=1, max_features=200)
    X_train_sp = word_vec.fit_transform(texts)
    X_val_sp = word_vec.transform(texts)

    # Test MultinomialNB tuning
    best_nb, nb_history = tune_multinomial_nb(
        X_train=X_train_sp,
        y_train=labels,
        X_val=X_val_sp,
        y_val=labels,
        alpha_candidates=[0.1, 0.5, 1.0],
    )
    assert len(nb_history) == 3
    assert best_nb.alpha in [0.1, 0.5, 1.0]

    # Test LogisticRegression tuning
    best_lr, lr_history = tune_logistic_regression(
        X_train=X_train_sp,
        y_train=labels,
        X_val=X_val_sp,
        y_val=labels,
        C_candidates=[0.5, 1.0, 2.0],
        max_iter=100,
    )
    assert len(lr_history) == 3
    assert best_lr.C in [0.5, 1.0, 2.0]


def test_build_and_train_classical_pipelines(sample_news_data, tmp_path):
    df = sample_news_data
    texts = df["content"].tolist()
    labels = df["label"].values

    models = ["majority", "multinomial_nb", "logistic_regression", "linear_svm"]
    for model_type in models:
        model_params = None
        if model_type == "logistic_regression":
            model_params = {"max_iter": 100}
        elif model_type in ["linear_svm", "svm"]:
            model_params = {"cv": 2, "max_iter": 100}

        pipeline = build_classical_pipeline(
            model_type=model_type,
            tfidf_params={"ngram_range": (1, 2), "min_df": 1, "max_features": 100},
            model_params=model_params,
        )
        pipeline.fit(texts, labels)

        preds, probs = predict_pipeline(pipeline, texts[:2])
        assert len(preds) == 2
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert set(preds).issubset({0, 1})

        # Test model serialization and reloading
        save_file = tmp_path / f"{model_type}.joblib"
        save_model(pipeline, str(save_file))
        assert save_file.exists()

        loaded_pipeline = load_model(str(save_file))
        loaded_preds, loaded_probs = predict_pipeline(loaded_pipeline, texts[:2])
        assert np.array_equal(preds, loaded_preds)
        assert np.allclose(probs, loaded_probs)


def test_load_split_data_parquet_and_csv(tmp_path):
    df_raw = pd.DataFrame(
        {
            "title": ["Real News Title", "Fake Story Headline"],
            "text": ["Fact verified by researchers.", "Conspiracy alien claims."],
            "label": [0, 1],  # 0=REAL, 1=FAKE
        }
    )

    # Test CSV loading
    csv_file = tmp_path / "train.csv"
    df_raw.to_csv(csv_file, index=False)
    loaded_csv, y_csv = load_split_data(str(csv_file))
    assert len(loaded_csv) == 2
    assert "content" in loaded_csv.columns
    assert list(y_csv.values) == [0, 1]

    # Test Parquet loading
    pq_file = tmp_path / "val.parquet"
    df_raw.to_parquet(pq_file, index=False)
    loaded_pq, y_pq = load_split_data(str(pq_file))
    assert len(loaded_pq) == 2
    assert "content" in loaded_pq.columns
    assert list(y_pq.values) == [0, 1]


def test_label_consistency_invalid_labels_rejected(tmp_path):
    invalid_df = pd.DataFrame(
        {
            "content": ["Valid text content", "Another valid article text"],
            "label": [0, 2],  # 2 is an invalid label
        }
    )
    invalid_file = tmp_path / "invalid.csv"
    invalid_df.to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="Labels in .* must be binary"):
        load_split_data(str(invalid_file))


def test_saved_majority_pipeline_reload_and_predict_raw_text(sample_news_data, tmp_path):
    """Verify Majority baseline pipeline has fitted TF-IDF step when saved and can predict raw text."""
    texts = sample_news_data["content"].tolist()
    labels = sample_news_data["label"].values

    maj_pipeline = Pipeline(
        [
            ("features", build_word_vectorizer(max_features=10)),
            ("clf", MajorityClassifier()),
        ]
    )
    # Fit the entire pipeline on raw text strings
    maj_pipeline.fit(texts, labels)

    save_path = tmp_path / "majority_baseline.joblib"
    save_model(maj_pipeline, str(save_path))
    assert save_path.exists()

    # Reload from disk and execute predictions on raw unvectorized text
    loaded_pipe = load_model(str(save_path))
    raw_unseen_texts = [
        "Breaking: Major medical breakthrough cures disease in record time.",
        "Local government council meets to discuss quarterly infrastructure budget.",
    ]

    preds, probs = predict_pipeline(loaded_pipe, raw_unseen_texts)
    assert len(preds) == 2
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert set(preds).issubset({0, 1})


def test_tune_linear_svm_records_actual_best_c(sample_news_data):
    """Verify tune_linear_svm tracks and returns the actual best C rather than the last attempted candidate."""
    texts = sample_news_data["content"].tolist()
    labels = sample_news_data["label"].values

    # Expand sample data slightly for 3-fold cross validation
    texts = texts * 3
    labels = np.tile(labels, 3)

    word_vec = build_word_vectorizer(ngram_range=(1, 2), min_df=1, max_features=100)
    X_train_sp = word_vec.fit_transform(texts)
    X_val_sp = word_vec.transform(texts)

    candidates = [0.01, 0.5, 1.0, 5.0]
    best_svm, best_c, history = tune_linear_svm(
        X_train=X_train_sp,
        y_train=labels,
        X_val=X_val_sp,
        y_val=labels,
        C_candidates=candidates,
        max_iter=100,
    )

    # Find candidate that actually achieved maximum Macro-F1
    expected_best = max(history, key=lambda x: x["val_macro_f1"])["C"]
    assert best_c == expected_best
    assert hasattr(best_svm, "best_C_")
    assert best_svm.best_C_ == expected_best
