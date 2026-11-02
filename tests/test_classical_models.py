import numpy as np

from fraudspotter.models.classical import (
    build_classical_pipeline,
    load_model,
    predict_pipeline,
    save_model,
)


def test_build_and_train_classical_models(sample_news_data, tmp_path):
    df = sample_news_data
    texts = (df["title"] + " " + df["text"]).tolist()
    labels = df["label"].values

    for model_type in ["logistic_regression", "multinomial_nb", "linear_svm"]:
        pipeline = build_classical_pipeline(model_type)
        pipeline.fit(texts, labels)

        preds, probs = predict_pipeline(pipeline, texts[:2])
        assert len(preds) == 2
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Test save and load
        save_file = tmp_path / f"{model_type}.joblib"
        save_model(pipeline, str(save_file))
        assert save_file.exists()

        loaded_pipeline = load_model(str(save_file))
        loaded_preds, loaded_probs = predict_pipeline(loaded_pipeline, texts[:2])
        assert np.array_equal(preds, loaded_preds)
