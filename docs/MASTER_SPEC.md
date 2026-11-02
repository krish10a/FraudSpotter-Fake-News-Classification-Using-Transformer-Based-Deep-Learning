# FraudSpotter — Master Specification v1.0

**Project title:** Explainable Fake News Classification Using Transformer-Based Deep Learning  
**Status:** Canonical implementation specification  
**Primary task:** Binary classification of a news headline and article body into REAL or FAKE, with model-attribution explanations and explicit uncertainty handling.

## 1. Objective and product boundary

Build a reproducible research system that:
1. Audits and prepares news datasets with leakage controls.
2. Compares classical NLP baselines against a fine-tuned transformer.
3. Evaluates predictive performance, calibration, robustness, generalization, explanation faithfulness, and inference efficiency.
4. Exposes predictions through FastAPI and a Streamlit dashboard.
5. Communicates limitations and does not claim to independently fact-check claims.

**Input:** headline/title and article body (one may be absent only under a documented input policy).  
**Output:** `LIKELY_REAL`, `LIKELY_FAKE`, or `NEEDS_REVIEW`, class probabilities/scores, threshold, model version, latency, and optional token attribution.

The model learns patterns associated with dataset labels. It does not establish the truth of a real-world claim or replace independent verification.

## 2. Frozen technical decisions

| Area | Decision |
|---|---|
| Primary dataset | WELFake |
| External diagnostics | ISOT and LIAR, evaluated separately |
| Internal labels | REAL=0, FAKE=1 |
| Main transformer | `distilbert/distilbert-base-uncased` |
| Tokenization | Fast DistilBERT WordPiece tokenizer |
| Required baselines | Majority sanity baseline, MultinomialNB, TF-IDF Logistic Regression |
| Optional baseline | Linear SVM with validation-based probability calibration |
| Main explanation | Captum Layer Integrated Gradients |
| Secondary explanation | LIME |
| Global feature association | Logistic Regression coefficients |
| Primary metric | Macro-F1 |
| Split target | 80% train / 10% validation / 10% test, group-aware |
| Seed | 42 for the main reproducible run |
| Training | PyTorch + Hugging Face Transformers, Colab T4 |
| API/UI | FastAPI + Streamlit |
| Optimization | ONNX FP32 then INT8 after validation |
| Tests | Pytest; lint with Ruff |

DeBERTa-v3-base, SHAP, BiLSTM, RAG, live crawling, agentic workflows, authentication, and databases are not core deliverables. Only consider optional experiments after all required milestones are complete and reproducible.

## 3. Label convention

WELFake's source label convention is treated as `0=FAKE`, `1=REAL`. Convert once during ingestion:

```python
internal_label = 1 - original_label
ID2LABEL = {0: "REAL", 1: "FAKE"}
LABEL2ID = {"REAL": 0, "FAKE": 1}
```

Persist and test this mapping. Never independently remap labels in model, API, UI, or evaluation code.

## 4. Dataset roles

### WELFake — primary benchmark
- Use title and article text as model inputs.
- Validate the downloaded file's schema and label values rather than assuming them.
- The reported upstream dataset size is approximately 72,134 records; actual loaded, retained, deduplicated, and split counts must be measured and written to the manifest.
- Do not treat dataset labels as perfect ground truth.

### ISOT — separate external-domain diagnostic
- Keep ISOT out of WELFake training and model selection.
- Remove verified source/byline prefixes using conservative, auditable rules.
- Do not use `subject` or `date` as classifier inputs; retain them only for optional slice analysis.
- Check exact/near-duplicate overlap with WELFake before external evaluation.
- Report all metrics separately and describe domain/label differences.

### LIAR — separate short-statement diagnostic
- Do not merge LIAR with article datasets.
- Primary explicit mapping: `pants-fire`, `false`, `barely-true` -> FAKE; `half-true`, `mostly-true`, `true` -> REAL.
- Also run a sensitivity diagnostic excluding `half-true` if data permits.
- Report LIAR results separately; do not compare its score as if it were the same full-article task.

## 5. Data pipeline

1. Download data directly in Colab from an official or verified remote source; record source, access date, file hash, and license/redistribution notes.
2. Validate required columns, data types, label values, nulls, empty text, and malformed records.
3. Normalize with Unicode NFKC, whitespace cleanup, and conservative HTML/URL handling.
4. Strip only verified source/byline boilerplate with anchored rules. Do not remove normal mentions of publishers or entities from article body text.
5. Retain title and body as distinct fields. Do not stem, lemmatize, or remove stopwords from transformer inputs; preserve negation and sentence order.
6. Compute normalized exact fingerprints and identify near-duplicate candidates (e.g. MinHash/LSH). Review sampled similarity matches before freezing the threshold.
7. Assign duplicate clusters to a single partition. Create reproducible stratified group-aware 80/10/10 partitions with seed 42; if exact ratios cannot be met, report actual ratios and class distributions.
8. Run exact and near-duplicate overlap audits across train/validation/test.
9. Save raw/processed counts, removed-record reasons, duplicate cluster counts, class ratios, split IDs, hashes, configuration, and source provenance in a manifest.
10. Freeze partitions. Do not alter them after model comparison begins.

No reliable publisher identifier is guaranteed for every WELFake article; group splitting reduces duplicate contamination but does not establish publisher-disjoint generalization.

## 6. Models

### Classical baselines
- Majority classifier for sanity floor.
- MultinomialNB using word TF-IDF (1–2 grams), `min_df` starting at 2, alpha tuned on validation.
- Logistic Regression using word TF-IDF (1–2 grams) plus character TF-IDF (3–5 grams); initial `min_df=2`, word and character feature caps around 150k each, `max_iter=2000`, C candidates 1, 2, 4. Tune on validation only.
- Optional LinearSVC on the same representation; use validation-fitted calibration if probability-based metrics are required.
- Fit vectorizers only on training records. Serialize the whole preprocessing/vectorizer/classifier pipeline.

### DistilBERT
- Checkpoint: `distilbert/distilbert-base-uncased`.
- Use `AutoTokenizer(use_fast=True)` and `AutoModelForSequenceClassification(num_labels=2)`.
- Pair tokenize `(title, body)`; prioritize title when truncating the second sequence. Confirm tokenizer support and model input fields; do not assume BERT token-type IDs are required by DistilBERT.
- Run max-length experiments at 128, 256, and 384; also run headline-only ablation.
- Initial config: learning rate 2e-5, maximum 3 epochs, train batch 8, gradient accumulation 2 (effective batch 16), eval batch 16, weight decay 0.01, warmup ratio 0.10, linear scheduler, FP16 where supported, gradient clipping 1.0, early stopping patience 1–2.
- Use dynamic padding and save best checkpoint by validation Macro-F1.
- Optuna: limited 5–10 trials after a working baseline; tune LR, weight decay, max length and modest batch/dropout choices. Keep tuning budget bounded.
- Run additional seeds on the selected configuration if compute permits; report variability.

## 7. Model selection, threshold and calibration

- Select model and sequence length on validation data, not test data.
- Report default threshold 0.50 and a validation-selected threshold.
- Search fake-probability thresholds from 0.10 to 0.90 in 0.01 increments; primary threshold selection maximizes Macro-F1, with a documented deterministic tie-break.
- Optionally configure a conservative operating point with a target fake precision constraint only if validation achieves it; report the actual value.
- Fit temperature scaling on validation predictions only. Freeze temperature and thresholds before test evaluation.
- Define `NEEDS_REVIEW` using a validation-assessed uncertainty margin; report coverage and error rate. Do not invent a confidence guarantee.

## 8. Explainability

### Captum Integrated Gradients
- Use LayerIntegratedGradients on the transformer embedding layer.
- Explain the fake-vs-real logit margin `z_fake - z_real` so attribution direction is consistent.
- Aggregate embedding dimensions into token scores; remove padding/special tokens; align WordPieces to readable words using offsets where possible.
- Positive attribution pushes toward FAKE; negative pushes toward REAL. These are model attributions, not causal/factual evidence.

### LIME
- Provide raw text -> two-class probability function.
- Generate per-article explanations on demand, not for every batch row.
- Display local explanation fit score and positive/negative contributions.

### Global interpretation
- Extract and visualize top positive/negative Logistic Regression coefficients. Describe as feature associations, not universal causal signals.

### Numerical explanation diagnostics
- Use at least 200 stratified held-out records if compute permits.
- Compare deletion of top 1, 3, 5 attributed tokens with random-token deletion of equal counts.
- Report comprehensiveness: original predicted-class score minus score after removing top tokens.
- Report sufficiency: original predicted-class score minus score using only top tokens.
- Report mean, variability, and confidence intervals. Perturbation results are diagnostics, not causal proof.

## 9. Evaluation contract

For every model, use the same frozen WELFake test records and save raw probabilities/predictions.

Report: accuracy; per-class precision, recall and F1; Macro-F1; weighted-F1; balanced accuracy; specificity; ROC-AUC; PR-AUC; MCC; confusion matrix; Brier score; ECE; NLL where available.

Additional analyses:
- Threshold curves and confusion matrices.
- 95% stratified paired bootstrap confidence intervals (1,000 resamples) for key metrics and model differences.
- McNemar test for paired model error differences; account for multiple comparisons where several models are tested.
- Calibration reliability diagram and temperature scaling.
- Robustness suite: capitalization, whitespace, punctuation, conservative URL/HTML formatting, Unicode normalization and controlled spelling noise; measure prediction consistency, probability deviation and metric change.
- Information ablations: title-only, body-only, 128/256/384 truncation and controlled word dropout; report performance degradation rather than assuming invariance.
- Leakage analysis: compare raw/minimally-cleaned vs source-marker-cleaned text where applicable; inspect ISOT subject/date associations only as diagnostics.
- External ISOT and LIAR metrics in separate reports.
- Review at least 50 test errors, preferably balanced false positives/false negatives, and categorize them.
- Benchmark cold load, warm median/P95 latency, throughput, model size, RAM/VRAM and training time with hardware/runtime/sequence length/batch size recorded.

Never put target numbers in results as if achieved. Targets are aspirational; measurements must come from code.

## 10. Serving and UI

FastAPI endpoints: `/health`, `/predict`, `/explain`, `/batch_predict`, `/metrics`, `/models`.

Use a central ModelRegistry and common prediction contract containing model/version, predicted label, class scores, threshold, review state, latency, and explanation availability.

Streamlit pages: Overview, News Analyzer, Explainability, Model Laboratory, Robustness Laboratory, Batch Analyzer, About & Limitations.

Use caching so the model loads once. Validate inputs, cap article length and batch size, handle unsupported-language/empty inputs honestly, escape untrusted text in UI, and avoid storing raw user articles by default.

## 11. Optimization

- Keep the PyTorch checkpoint as reference and for Captum gradients.
- Export ONNX FP32 and compare logits/probabilities/predictions against PyTorch.
- Apply ONNX Runtime dynamic INT8 quantization after FP32 parity checks.
- Re-evaluate quality and benchmark speed. Keep INT8 only if measured benefit and acceptable quality are demonstrated.
- Do not promise a fixed size reduction or latency without measurements.

## 12. Compute and artifact transfer

- Local laptop: coding, lightweight tests, API, UI, Docker, docs, CPU inference.
- Colab T4: remote dataset download, preprocessing at scale, model training, evaluation, explainability and ONNX export.
- No GB-scale laptop-to-Colab uploads. Colab clones small source code from GitHub and downloads data/pretrained weights from remote sources itself.
- Persist raw/processed datasets and checkpoints to Google Drive; optionally publish model artifacts to Hugging Face Hub only with user approval.
- Keep raw data and large weights out of GitHub.
- Export/download model artifacts from Colab to local only when needed for serving.

## 13. Completion definition

Completion is based on reproducibility and verified deliverables, not a guaranteed 82% Macro-F1 or 30/30 marks. See `ACCEPTANCE_CRITERIA.md` for phase gates.
