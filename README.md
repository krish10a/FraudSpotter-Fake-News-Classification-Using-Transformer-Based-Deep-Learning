# 🛡️ FraudSpotter

> **Explainable Fake News Classification System**  
> High-performance fake news detection powered by DistilBERT, classical baselines, ONNX INT8 optimization, Captum Integrated Gradients, and LIME.

---

## Architecture Overview

FraudSpotter is architected with strict separation between **local development / serving** and **GPU cloud training (Google Colab)**:

- **Local Machine:** Fast CPU inference, FastAPI REST API, Streamlit dashboard, classical baseline modeling, unit testing, containerization.
- **Google Colab (T4 GPU):** Direct remote dataset downloading (no GB-sized uploads), DistilBERT transformer fine-tuning, Optuna optimization, Captum/LIME batch attribution, and ONNX INT8 quantization.

### Canonical Technical Stack
- **Primary Dataset:** WELFake (72,134 news articles).
- **External Evaluation:** ISOT and LIAR (evaluated separately for out-of-domain robustness).
- **Core Transformer:** DistilBERT (`distilbert-base-uncased`).
- **Classical Baselines:** Multinomial Naive Bayes, TF-IDF Logistic Regression, Linear SVM.
- **Explainability:** Captum Integrated Gradients (token-level), LIME (local surrogate), Logistic Regression coefficients (global feature importance).
- **Evaluation:** Macro-F1, Accuracy, Precision, Recall, ROC-AUC, PR-AUC, MCC, Expected Calibration Error (ECE), Brier Score, Bootstrap 95% Confidence Intervals, and McNemar's Test.
- **Optimization:** ONNX Runtime INT8 Dynamic Quantization (<150 MB footprint, CPU-optimized latency).
- **Serving & UI:** FastAPI backend + Streamlit frontend.

---

## Repository Structure

```
├── .env.example                      # Template for environment variables
├── .gitignore                        # Git exclusion rules (prevents committing large data/weights)
├── AGENTS.md                         # Permanent project rules for AI and developers
├── COLAB_WORKFLOW.md                 # Detailed guide for Colab GPU execution & artifact sync
├── Makefile                          # Cross-platform development command shortcuts
├── README.md                         # Project overview and documentation
├── TASKS.md                          # Phase-wise development roadmap
├── requirements-local.txt            # Local development requirements (CPU / FastAPI / Streamlit)
├── requirements-colab.txt            # Colab training requirements (PyTorch GPU / Transformers / Optuna)
├── configs/                          # Modular YAML configuration files
│   ├── data_config.yaml              # Dataset paths, columns, and split ratios
│   ├── training_config.yaml          # Hyperparameters for classical & transformer models
│   ├── evaluation_config.yaml        # Metrics, calibration, and statistical testing specs
│   ├── explainability_config.yaml    # Captum, LIME, and faithfulness settings
│   └── serving_config.yaml           # API and ONNX runtime configuration
├── docs/                             # Project documentation (MASTER_SPEC.md goes here)
├── notebooks/
│   └── colab/
│       └── FraudSpotter_Training.ipynb # Dedicated Colab T4 GPU execution notebook
├── scripts/                          # Modular CLI automation scripts
│   ├── download_datasets.py          # Remote dataset fetcher (WELFake, ISOT, LIAR)
│   ├── preprocess_data.py            # Text cleaning & stratified splitting
│   ├── train_classical.py            # Train Naive Bayes, Logistic Regression, Linear SVM
│   ├── train_transformer.py          # Fine-tune DistilBERT on GPU
│   ├── evaluate_models.py            # Comprehensive evaluation & McNemar test
│   ├── explain_predictions.py        # Captum IG, LIME & Faithfulness evaluation
│   └── export_onnx.py                # DistilBERT to ONNX + INT8 quantization
├── src/
│   └── fraudspotter/                 # Main Python package
│       ├── common/                   # Config loader, logger, deterministic seed
│       ├── data/                     # Text preprocessors and dataset loaders
│       ├── models/                   # Classical pipelines, transformer, ONNX runtime
│       ├── explainability/           # Integrated Gradients, LIME, LR coefficients
│       ├── evaluation/               # Metrics, calibration, statistical tests, faithfulness
│       ├── api/                      # FastAPI application, schemas, routes
│       └── web/                      # Streamlit interactive frontend
├── tests/                            # Pytest test suite
│   ├── conftest.py                   # Shared fixtures
│   ├── test_preprocessor.py          # Tests for text cleaning & splitting
│   ├── test_classical_models.py      # Tests for classical pipelines & serialization
│   ├── test_metrics.py               # Tests for metrics, ECE, CIs, McNemar
│   ├── test_api.py                   # Tests for FastAPI endpoints
│   └── fixtures/                     # Small synthetic news fixtures
├── data/                             # Raw and processed datasets (ignored by Git)
└── artifacts/                        # Serialized models, figures, and metrics (ignored by Git)
```

---

## Local Setup & Quickstart

### 1. Environment Setup
Using `uv` or standard Python:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install local dependencies
pip install -r requirements-local.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Generate Local Sample Data & Train Baselines
```bash
# Generate small sample fixture
python scripts/download_datasets.py --sample

# Preprocess fixture
python scripts/preprocess_data.py --input_file data/raw/sample_welfake.csv --output_dir data/processed/welfake

# Train classical baselines
python scripts/train_classical.py --model_type all

# Run evaluation on test split
python scripts/evaluate_models.py
```

### 4. Run API & UI
```bash
# Start FastAPI backend
uvicorn src.fraudspotter.api.app:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit frontend
streamlit run src/fraudspotter/web/app.py
```

---

## Google Colab Training Workflow
See [`COLAB_WORKFLOW.md`](file:///c:/Users/ASUS/Desktop/nlp/COLAB_WORKFLOW.md) for full instructions on running `notebooks/colab/FraudSpotter_Training.ipynb` on a T4 GPU.

