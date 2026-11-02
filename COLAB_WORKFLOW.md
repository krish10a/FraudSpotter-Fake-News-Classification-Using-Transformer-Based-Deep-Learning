# Google Colab GPU Workflow & Synchronization Guide

This document outlines the protocol for executing compute-heavy and GPU-dependent tasks on **Google Colab (T4 GPU)** while maintaining strict synchronization with your local environment.

---

## 1. Core Principle: Zero Laptop-to-Colab Large File Uploads

> [!IMPORTANT]
> **Never upload multi-gigabyte datasets, checkpoints, or weights from your local machine to Colab.**
> - Colab has superior download speeds (hundreds of megabits per second).
> - All datasets (WELFake, ISOT, LIAR) and base transformer weights (`distilbert-base-uncased`) are downloaded directly inside Colab from remote sources (Zenodo, Kaggle, Hugging Face).

---

## 2. End-to-End Workflow Diagram

```
+-------------------------------------------------------------+
|                      LOCAL LAPTOP                           |
|  - Edit source code, tests, configs, FastAPI & Streamlit    |
|  - Git push commits to GitHub                               |
+------------------------------+------------------------------+
                               | git push / git pull
                               v
+-------------------------------------------------------------+
|                     GITHUB REPOSITORY                       |
|  - Houses codebase, configs, scripts, test fixtures         |
+------------------------------+------------------------------+
                               | git clone / git pull
                               v
+-------------------------------------------------------------+
|                   GOOGLE COLAB (T4 GPU)                     |
|  1. Pull latest code from GitHub                            |
|  2. Download raw datasets directly (Zenodo / Kaggle)        |
|  3. Run preprocessing & stratified splitting                |
|  4. Train classical models & DistilBERT transformer         |
|  5. Run Optuna, Captum IG, LIME, McNemar & calibration      |
|  6. Export & quantize to INT8 ONNX (~65 MB)                 |
|  7. Copy small artifacts (<150 MB) to Google Drive          |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                      GOOGLE DRIVE                           |
|  /MyDrive/FraudSpotter_Artifacts/                           |
|    ├── distilbert_onnx/model_quantized.onnx                 |
|    ├── classical/logistic_regression.joblib                 |
|    └── metrics/evaluation_report.json                       |
+------------------------------+------------------------------+
                               |
                               v Download small artifacts to laptop
+-------------------------------------------------------------+
|                      LOCAL LAPTOP                           |
|  - Place model_quantized.onnx in artifacts/models/          |
|  - Test FastAPI and Streamlit locally on CPU                |
+-------------------------------------------------------------+
```

---

## 3. Step-by-Step Colab Execution Instructions

### Step 3.1: Open Notebook in Google Colab
1. Navigate to Google Colab: [colab.research.google.com](https://colab.research.google.com).
2. Select **Upload** and upload `notebooks/colab/FraudSpotter_Training.ipynb` (or connect directly via GitHub).
3. Ensure GPU runtime is active:
   - Click **Runtime** > **Change runtime type**
   - Select **T4 GPU** (or available GPU accelerator)
   - Click **Save**

### Step 3.2: Mount Google Drive
Execute Cell 1 to mount Google Drive. This creates a persistent directory at:
```
/content/drive/MyDrive/FraudSpotter_Artifacts/
```
All training results and models will be saved here so you do not lose progress if the Colab session disconnects.

### Step 3.3: Clone Code & Install Dependencies
Run:
```bash
!git clone https://github.com/<your-username>/fraudspotter.git /content/fraudspotter
%cd /content/fraudspotter
!pip install -r requirements-colab.txt
```

### Step 3.4: Download Datasets Directly in Colab
Execute the dataset acquisition cell. WELFake is automatically retrieved from Zenodo:
```python
!python scripts/download_datasets.py --dataset welfake
```
For ISOT or Kaggle datasets, configure Kaggle API keys via Colab Secrets (`userdata.get('KAGGLE_KEY')`) or upload `kaggle.json`.

### Step 3.5: Run Preprocessing & Training
Execute:
```bash
# 1. Stratified Preprocessing
!python scripts/preprocess_data.py --input_file data/raw/welfake.csv --output_dir data/processed/welfake

# 2. Train Classical Baselines
!python scripts/train_classical.py --model_type all

# 3. Fine-Tune DistilBERT on T4 GPU
!python scripts/train_transformer.py --train_file data/processed/welfake/train.csv --val_file data/processed/welfake/val.csv

# 4. Comprehensive Evaluation
!python scripts/evaluate_models.py --test_file data/processed/welfake/test.csv

# 5. Explainability & Faithfulness
!python scripts/explain_predictions.py

# 6. ONNX INT8 Quantization
!python scripts/export_onnx.py
```

### Step 3.6: Sync Artifacts to Local Machine
The final notebook cell copies the quantized ONNX model (`model_quantized.onnx` ~65 MB) and evaluation JSON reports to your Google Drive.

On your local laptop:
1. Open Google Drive and locate `FraudSpotter_Artifacts`.
2. Download `model_quantized.onnx` into `artifacts/models/distilbert_onnx/`.
3. Download `evaluation_report.json` into `artifacts/metrics/`.
4. Run FastAPI and Streamlit locally:
   ```bash
   uvicorn src.fraudspotter.api.app:app --reload
   streamlit run src/fraudspotter/web/app.py
   ```

