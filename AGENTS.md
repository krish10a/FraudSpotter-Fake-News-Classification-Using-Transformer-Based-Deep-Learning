# AGENTS.md - Permanent Project Rules for FraudSpotter

## 1. Project Overview & Identity
FraudSpotter is an end-to-end, explainable fake news classification platform. All agents and developers working on this repository must adhere strictly to the rules, technical constraints, and canonical decisions defined herein.

## 2. Canonical Technical Stack & Decisions
Any modification to these decisions requires explicit user authorization:
- **Primary Dataset:** WELFake (72k+ articles).
- **External Evaluation:** ISOT and LIAR (evaluated separately for generalization and cross-domain robustness).
- **Core Transformer:** DistilBERT (`distilbert-base-uncased`).
- **Classical Baselines:** Multinomial Naive Bayes (`MultinomialNB`) and TF-IDF Logistic Regression (`LogisticRegression`).
- **Optional Baseline:** Linear Support Vector Classifier (`LinearSVC`).
- **Explainability:**
  - Local: Captum Integrated Gradients (token-level attribution) + LIME.
  - Global: Logistic Regression feature coefficient inspection (top informative n-grams).
- **Evaluation Framework:**
  - Standard: Macro-F1, Accuracy, Precision, Recall, ROC-AUC, PR-AUC, MCC.
  - Calibration: Expected Calibration Error (ECE) and Brier Score.
  - Statistical Testing: Bootstrap Confidence Intervals (95% CI) and McNemar's Test for pairwise model comparison.
  - Reliability: Robustness perturbation testing and Explanation Faithfulness (comprehensiveness & sufficiency).
- **Serving & Optimization:**
  - Backend: FastAPI.
  - Frontend: Streamlit.
  - Optimization: ONNX Runtime with INT8 dynamic quantization for low-latency CPU inference.
- **Testing:** Pytest with unit and integration test fixtures.
- **Deployment:** Docker containerization + GitHub Actions CI/CD.
- **Training Environment:** Google Colab (T4 GPU).

---

## 3. Mandatory Computing Separation
Strict separation of duties must be maintained across environments:

### Local Laptop Environment
- Modular codebase structure, source files, and utility modules.
- Preprocessing pipelines, data cleaning scripts, and small test fixtures.
- Classical ML baseline development and lightweight debugging.
- FastAPI backend and Streamlit frontend development.
- Unit and integration tests (Pytest).
- Docker configuration, GitHub Actions workflows, documentation.
- Local inference verification using saved ONNX INT8 / joblib models.
- **NEVER** download full multi-gigabyte datasets locally.
- **NEVER** attempt deep transformer fine-tuning on the local CPU/laptop.

### Google Colab Environment (T4 GPU)
- Direct remote download of datasets (WELFake, ISOT, LIAR).
- Transformer fine-tuning and checkpointing (DistilBERT).
- Optuna hyperparameter optimization.
- GPU-accelerated evaluation across internal and external test sets.
- Integrated Gradients and LIME compute-intensive batch explanations.
- ONNX export and INT8 post-training quantization.
- Export of final small models (<150 MB INT8 ONNX, metrics JSON, figures) to Google Drive or Hugging Face Hub.

---

## 4. Critical Storage & Transfer Constraint
- **NEVER** use `google.colab.files.upload()` or instruct the user to upload gigabyte-sized files from laptop to Colab.
- Colab notebooks and scripts must stream or download datasets directly from online repositories (Zenodo, Kaggle API, Hugging Face Datasets, or direct URLs).
- Only small serialized artifacts (e.g., INT8 ONNX weights, vectorizer pickles, metrics JSONs) are synced from Colab back to the repository for local serving.

---

## 5. Security & Secrets Management
- Never commit API tokens, passwords, Kaggle keys, or Hugging Face tokens to version control.
- Use `.env` files locally and Colab Secrets (`google.colab.userdata`) in Colab.
- Reference `.env.example` as the source of truth for required environment variables.

---

## 6. Development & Coding Standards
- Type hints (`typing`) and docstrings are required for all core functions and classes.
- Code formatting and linting must pass `ruff` standards.
- Every new module must have a corresponding test suite under `tests/`.
- No silent deletions or modifications of approved architectural decisions.
- Do not commit or push to remote Git without explicit user confirmation.

