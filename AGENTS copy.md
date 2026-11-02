# FraudSpotter — Permanent Agent Instructions

You are the implementation agent for FraudSpotter. Treat `docs/MASTER_SPEC.md` as the canonical technical specification and the other files in `docs/` as its implementation contracts.

## Non-negotiable project decisions
- Primary benchmark: WELFake.
- ISOT and LIAR are separate external diagnostic evaluations; do not merge them into WELFake training or combine their scores with the WELFake test score.
- Main transformer: `distilbert/distilbert-base-uncased`.
- Required classical baselines: majority-class sanity baseline, Multinomial Naive Bayes, TF-IDF Logistic Regression. Linear SVM is optional.
- Internal labels: REAL=0, FAKE=1. WELFake source labels: 0=FAKE, 1=REAL; map with `internal_label = 1 - original_label`.
- Main metric: Macro-F1. Report class-wise precision/recall and false positives/negatives as well.
- Explanations: Captum Integrated Gradients for the transformer, LIME as a secondary local method, and Logistic Regression coefficients for global feature associations.
- Application: FastAPI backend + Streamlit frontend.
- Optimization: ONNX FP32 then INT8 only after numerical parity and quality checks.

## Execution boundaries
- Local Windows laptop: repository, coding, lightweight tests, classical development if practical, FastAPI, Streamlit, Docker, documentation, and CPU inference checks.
- Google Colab T4: large dataset download, GPU training, transformer experiments, GPU evaluation, explanation experiments if needed, and model export.
- Never require uploading large local datasets/checkpoints into Colab. Colab must download source data and pretrained checkpoints directly from their remote sources. Persistent outputs go to Google Drive and/or an explicitly approved Hugging Face Hub repository.
- Never download multi-GB data or model weights onto the laptop unless explicitly requested.

## Data and experiment integrity
- Do not change the dataset, model, label mapping, split policy, or evaluation design without explaining the reason and asking for approval.
- Deduplicate before splitting. Keep duplicate groups inside a single partition. Use a reproducible, stratified, group-aware 80/10/10 split; verify actual proportions and class balance.
- Fit TF-IDF/vectorizers on training data only.
- Use validation only for model selection, hyperparameter tuning, threshold selection, and calibration fitting.
- Never tune or select using test results. Evaluate the frozen system on test once for final reporting.
- No fabricated metrics, test counts, screenshots, or claims of successful execution. Distinguish planned targets from measured results.
- Do not claim that this system verifies truth. Use “Likely Real”, “Likely Fake”, or “Needs Review”.

## Coding and workflow rules
- Inspect existing files before creating or overwriting them.
- Work one phase at a time. State the files changed, commands run, actual test output, unresolved issues, and next step.
- Add tests for new behavior. Keep configuration in YAML rather than duplicating constants across modules.
- Keep training code separate from serving/UI code.
- Save model, tokenizer, preprocessing, label mapping, threshold, calibration temperature, dataset/split hashes, package versions, and experiment configuration together as a versioned artifact set.
- Never commit raw data, checkpoints, credentials, `.env`, or Colab runtime files.
- Never run destructive cleanup or publish/upload artifacts without explicit user approval.
- When manual action is required, give exact short instructions and wait.

## Definition of done
A phase is complete only when its acceptance criteria in `docs/ACCEPTANCE_CRITERIA.md` pass and the actual evidence is recorded. A created file or generated code is not proof that a feature works.
