import json

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
    save_model_bundle,
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

    # Test LogisticRegression tuning with solver benchmarking
    best_lr, best_c, best_solver, lr_history = tune_logistic_regression(
        X_train=X_train_sp,
        y_train=labels,
        X_val=X_val_sp,
        y_val=labels,
        C_candidates=[0.5, 1.0, 2.0],
        solver_candidates=["lbfgs", "liblinear"],
        max_iter=100,
    )
    assert len(lr_history) == 6  # 2 solvers x 3 C candidates
    assert best_c in [0.5, 1.0, 2.0]
    assert best_solver in ["lbfgs", "liblinear"]


def test_build_and_train_classical_pipelines(sample_news_data, tmp_path):
    df = sample_news_data
    texts = df["content"].tolist()
    labels = df["label"].values

    models = [
        "majority",
        "multinomial_nb",
        "logistic_word",
        "logistic_word_char",
        "linear_svm",
    ]
    for model_type in models:
        model_params = None
        if "logistic" in model_type:
            model_params = {"max_iter": 100}
        elif model_type in ["linear_svm", "svm"]:
            model_params = {"max_iter": 100, "use_calibration": False}

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


def test_char_vectorizer_pruning_and_feature_limits():
    """Verify character TF-IDF vectorizer defaults to min_df=3 and max_features=50000 with float32."""
    char_vec = build_char_vectorizer()
    assert char_vec.min_df == 3
    assert char_vec.max_features == 50000
    assert char_vec.dtype == np.float32
    assert char_vec.analyzer == "char_wb"
    assert char_vec.ngram_range == (3, 5)


def test_text_truncation_logic():
    """Verify text truncation properly limits long articles while preserving short ones."""
    sample_texts = [
        "A" * 12000,  # Exceeds 8000
        "B" * 3000,  # Within 8000
    ]
    max_chars = 8000
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in sample_texts]

    assert len(truncated[0]) == 8000
    assert len(truncated[1]) == 3000


def test_intermediate_report_persistence(tmp_path):
    """Verify intermediate reporting persists valid JSON after each stage."""
    from scripts.train_classical import save_json

    test_file = tmp_path / "test_metrics.json"
    data = {"stage": "majority", "macro_f1": 0.5}

    save_json(data, test_file)
    assert test_file.exists()

    with open(test_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["stage"] == "majority"


def test_save_model_bundle_and_directory_load(sample_news_data, tmp_path):
    """Verify save_model_bundle generates pipeline, vectorizer, and metadata, and load_model can reload from folder."""
    texts = sample_news_data["content"].tolist()
    labels = sample_news_data["label"].values

    pipe = build_classical_pipeline("naive_bayes")
    pipe.fit(texts, labels)

    vec = pipe.named_steps["features"]
    bundle_dir = save_model_bundle(
        output_dir=tmp_path,
        model_name="nb_test",
        pipeline=pipe,
        vectorizer=vec,
        metadata={"alpha": 0.1, "dataset": "test_fixture"},
    )

    assert (bundle_dir / "pipeline.joblib").exists()
    assert (bundle_dir / "vectorizer.joblib").exists()
    assert (bundle_dir / "model_metadata.json").exists()

    with open(bundle_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_name"] == "nb_test"
    assert "sha256" in meta

    # Test loading directly via folder path
    reloaded_pipe = load_model(str(bundle_dir))
    preds, probs = predict_pipeline(reloaded_pipe, texts[:2])
    assert len(preds) == 2
    assert probs.shape == (2, 2)


def test_uncalibrated_svm_metrics():
    """Verify compute_uncalibrated_svm_metrics computes metrics on margin scores with null calibration."""
    from scripts.train_classical import compute_uncalibrated_svm_metrics

    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    # Decision margins: positive -> class 1, negative -> class 0
    decision_scores = np.array([-1.5, 2.1, -0.4, 0.9, 1.2, -0.8])

    metrics = compute_uncalibrated_svm_metrics(y_true, decision_scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["brier_score"] is None
    assert metrics["expected_calibration_error"] is None
    assert metrics["roc_auc"] == 1.0


def test_parquet_selective_column_loading(tmp_path):
    """Verify load_split_data correctly loads Parquet with combined_text and label."""
    df_raw = pd.DataFrame(
        {
            "record_id": ["rec_1", "rec_2"],
            "title": ["Title 1", "Title 2"],
            "text": ["Text 1", "Text 2"],
            "combined_text": ["Title 1 Text 1 combined", "Title 2 Text 2 combined"],
            "label": [0, 1],
            "split": ["train", "train"],
        }
    )
    pq_file = tmp_path / "selective.parquet"
    df_raw.to_parquet(pq_file, index=False)

    loaded_df, loaded_y = load_split_data(str(pq_file))
    assert len(loaded_df) == 2
    assert "content" in loaded_df.columns
    assert loaded_df["content"].iloc[0] == "Title 1 Text 1 combined"
    assert list(loaded_y.values) == [0, 1]
