# TASKS.md - Phase-Wise Development Roadmap

## Status Legend
- [x] Completed
- [/] In Progress
- [ ] Planned

---

## Phase 0: Project Setup & Environment Architecture (Current)
- [x] Inspect OS, Python runtime, Git status, and development tools.
- [x] Establish modular repository structure (`src/`, `configs/`, `scripts/`, `notebooks/`, `tests/`).
- [x] Establish permanent project guidelines in `AGENTS.md`.
- [x] Author core YAML configs (`data`, `training`, `evaluation`, `explainability`, `serving`).
- [x] Create separate `requirements-local.txt` and `requirements-colab.txt`.
- [x] Create `.gitignore`, `.env.example`, `Makefile`, and `README.md`.
- [x] Author executable scripts for downloading, preprocessing, classical baselines, transformer fine-tuning, evaluation, explainability, and ONNX export.
- [x] Prepare dedicated Colab notebook at `notebooks/colab/FraudSpotter_Training.ipynb`.
- [x] Document GPU execution and artifact synchronization protocol in `COLAB_WORKFLOW.md`.
- [x] Create comprehensive Pytest test suite with fixture data.
- [ ] Await user input of complete Master Specification into `docs/MASTER_SPEC.md`.

---

## Phase 1: Data Acquisition & Preprocessing Validation
- [ ] Initialize local sample test fixtures (`scripts/download_datasets.py --sample`).
- [ ] Validate stratified splitting logic (WELFake 80/10/10) with verified label distribution.
- [ ] Validate ISOT and LIAR dataset ingestion schemas.
- [ ] Implement text normalization benchmarks and edge-case handling.

---

## Phase 2: Classical Baselines & Statistical Benchmarks
- [ ] Train TF-IDF Multinomial Naive Bayes baseline.
- [ ] Train TF-IDF Logistic Regression baseline with balanced class weights.
- [ ] Train Linear SVM with probability calibration.
- [ ] Extract global Logistic Regression n-gram coefficients (top 25 positive and negative).
- [ ] Implement McNemar's test and 95% Bootstrap CI routines across classical models.

---

## Phase 3: Colab GPU Execution & Transformer Fine-Tuning
- [ ] Execute Colab training notebook (`FraudSpotter_Training.ipynb`) on T4 GPU.
- [ ] Direct download of full WELFake dataset on Colab instance.
- [ ] Fine-tune DistilBERT-base-uncased with fp16 mixed precision.
- [ ] Run Optuna hyperparameter trials (learning rate, batch size, warmup).
- [ ] Evaluate DistilBERT checkpoint on WELFake test set and log Macro-F1, ROC-AUC, PR-AUC, MCC.
- [ ] Perform out-of-distribution evaluation on ISOT and LIAR datasets.

---

## Phase 4: Explainability & Interpretability Pipeline
- [ ] Implement token-level attribution using Captum Integrated Gradients for DistilBERT.
- [ ] Implement local instance attribution using LIME TextExplainer.
- [ ] Compute explanation faithfulness: Comprehensiveness score.
- [ ] Compute explanation faithfulness: Sufficiency score.
- [ ] Generate comparative attribution visualization diagrams.

---

## Phase 5: Optimization & INT8 ONNX Export
- [ ] Export best DistilBERT PyTorch checkpoint to ONNX graph (opset 14).
- [ ] Apply INT8 dynamic post-training quantization.
- [ ] Benchmark inference latency & memory footprint: PyTorch FP32 vs ONNX FP32 vs ONNX INT8.
- [ ] Verify accuracy/F1 parity between PyTorch model and ONNX INT8 model.
- [ ] Sync lightweight INT8 ONNX model (<150 MB) to Google Drive.

---

## Phase 6: Local Backend Serving & Full API
- [ ] Place quantized INT8 model in local `artifacts/models/distilbert_onnx/`.
- [ ] Implement complete FastAPI routes (`/predict`, `/explain`, `/metrics`, `/health`).
- [ ] Add input validation, error handling, and payload size constraints.
- [ ] Write integration tests for API endpoints with pytest.

---

## Phase 7: Interactive Streamlit Frontend & Visualizations
- [ ] Implement interactive UI with news text input and veracity classification gauges.
- [ ] Render interactive token attribution heatmaps (Captum & LIME).
- [ ] Render global feature importance charts for Logistic Regression.
- [ ] Display reliability diagrams and calibration curves.
- [ ] Connect Streamlit frontend to FastAPI backend.

---

## Phase 8: Containerization & CI/CD Deployment
- [ ] Author multi-stage `Dockerfile` for lightweight production deployment.
- [ ] Create `docker-compose.yml` for unified API and Frontend orchestration.
- [ ] Configure GitHub Actions workflow (`.github/workflows/ci.yml`) for automated testing and linting.
- [ ] Final end-to-end audit and documentation sign-off.

