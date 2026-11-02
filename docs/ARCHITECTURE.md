# FraudSpotter — Architecture and Component Contracts

## High-level flow

```text
Remote datasets -> Colab data audit/clean/deduplicate/group split
                         |
                         v
         NB / TF-IDF Logistic Regression / DistilBERT
                         |
                         v
       Validation selection -> threshold -> calibration
                         |
                         v
         Frozen test evaluation + robustness + statistics
                         |
                         v
           Captum / LIME / faithfulness diagnostics
                         |
                         v
      PyTorch reference -> ONNX FP32 -> optional ONNX INT8
                         |
                         v
            FastAPI ModelRegistry and endpoints
                         |
                         v
                 Streamlit dashboard
```

## Separation of concerns

### Data layer
- `load_data.py`: load and validate data files.
- `preprocess.py`: deterministic text normalization and field construction.
- `deduplicate.py`: exact and near-duplicate grouping.
- `split_data.py`: stratified group-aware partitions.
- `manifest.py`: hashes, counts, split IDs and provenance.

### Modeling layer
- `train_nb.py`: MultinomialNB baseline.
- `train_logistic.py`: word + character TF-IDF Logistic Regression.
- `train_svm.py`: optional LinearSVC and validation calibration.
- `train_distilbert.py`: paired title/body fine-tuning.
- `registry.py`: artifact loading/version/config validation.
- `inference.py`: common probability/prediction contract.

### Evaluation layer
- `metrics.py`: classification metrics and confusion matrix.
- `threshold.py`: validation-only threshold selection.
- `calibration.py`: temperature fitting, ECE, Brier and reliability data.
- `bootstrap.py`, `statistical_tests.py`: paired confidence intervals and McNemar.
- `robustness.py`: controlled text transforms and information ablations.
- `external_validation.py`: separate ISOT/LIAR reporting.
- `benchmark.py`: model load, warm latency, P95, throughput and memory.

### Explainability layer
- `captum_ig.py`: Layer Integrated Gradients on PyTorch model.
- `token_alignment.py`: special-token removal and WordPiece-to-word alignment.
- `lime_explainer.py`: local black-box explanation.
- `lr_coefficients.py`: global coefficient associations.
- `faithfulness.py`: deletion and sufficiency experiments.

### Serving layer
- FastAPI validates request schemas and calls ModelRegistry.
- Streamlit calls FastAPI; it must not duplicate model-specific inference code.
- Model training must never run when the application starts.
- Captum requires a PyTorch gradient-capable model; ONNX Runtime is for optimized prediction, not Captum gradients. Keep both artifacts if using ONNX serving with PyTorch explanations.

## API contract

### `POST /predict`
Request:
```json
{
  "title": "Example headline",
  "text": "Example article body",
  "model": "distilbert",
  "mode": "balanced"
}
```
Response shape (values below are illustrative):
```json
{
  "model": "DistilBERT",
  "model_version": "fraudspotter-v1",
  "prediction": "NEEDS_REVIEW",
  "predicted_class_id": null,
  "fake_probability": 0.51,
  "real_probability": 0.49,
  "threshold": 0.50,
  "needs_review": true,
  "latency_ms": 0.0,
  "explanation_available": true
}
```

Other endpoints:
- `GET /health`
- `GET /models`
- `POST /explain`
- `POST /batch_predict`
- `GET /metrics`

## Execution boundary

| Work | Local laptop | Colab T4 |
|---|---:|---:|
| Source code, configs, docs | Primary | Pull from GitHub |
| Large dataset download | No | Yes |
| Deduplication/split at full scale | Optional/small tests | Primary |
| Classical baseline | Small tests/dev | Full experiment |
| DistilBERT training | No | Yes |
| Captum/LIME experiments | Small smoke test | Full evaluation |
| FastAPI/Streamlit | Primary | Not needed |
| ONNX CPU benchmark | Primary | Optional GPU/CPU comparison |
| Docker/CI/docs | Primary | No |

## Artifact bundle contract

Every deployable model bundle must include:
- model weights or ONNX file;
- tokenizer/config where relevant;
- `inference_config.json` containing label map, threshold, calibration temperature, max length, model version and serving mode;
- preprocessing version/config;
- source checkpoint hash and exported model hash;
- package/runtime versions;
- evaluation summary and parity report.
