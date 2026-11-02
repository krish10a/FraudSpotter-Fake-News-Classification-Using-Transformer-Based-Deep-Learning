# FraudSpotter — Acceptance Criteria and Phase Gates

A file being created is not proof that the feature works. Each gate needs execution evidence (command output, test result, report, hash, or screenshot).

| Gate | Acceptance criteria | Evidence |
|---|---|---|
| G0 Scaffold | Python 3.12 environment works; config parses; Git ignore reviewed; docs present | environment report, pytest, Ruff |
| G1 Dataset | WELFake downloaded in Colab; source/schema/labels verified | source URL, checksum, audit JSON |
| G2 Data integrity | Exact duplicates removed; near-duplicate groups audited; groups do not cross splits | split manifest and overlap assertions |
| G3 Baselines | Majority, NB and LR train; validation probabilities saved | model artifacts, logs, validation metrics |
| G4 Transformer | DistilBERT fine-tuning completes and can reload | checkpoint, tokenizer, training log |
| G5 Selection | Length/model chosen on validation only | comparison and selection record |
| G6 Calibration | Threshold and temperature fitted on validation only | inference_config.json |
| G7 Final evaluation | Frozen models evaluated on identical untouched test rows | final_evaluation.json, prediction file |
| G8 Statistics | Paired bootstrap and McNemar generated | statistical report |
| G9 XAI | Captum and LIME return valid aligned explanations | explanation examples and tests |
| G10 Faithfulness | Top-k vs random-k deletion and sufficiency evaluated | faithfulness.csv/plot |
| G11 Robustness | Transformations/ablations complete with separate interpretations | robustness.csv |
| G12 External | ISOT and LIAR tested separately with documented mappings | external_validation.json |
| G13 Error analysis | At least 50 errors reviewed and categorized where available | error_analysis.csv |
| G14 API | Health, prediction, explanation and batch routes tested | API test report |
| G15 UI | Single and batch workflows work; limitations visible | screenshots and smoke tests |
| G16 Optimization | ONNX parity measured; INT8 quality and speed measured | parity and benchmark report |
| G17 Packaging | Clean startup, Docker/CI checks and instructions work | CI run and README verification |
| G18 Submission | Report, slides, model/data cards and viva notes complete | submission checklist |

## Final quality rules
- No fabricated metrics or test counts.
- No test-set tuning.
- No raw datasets, credentials or large checkpoints in Git.
- All model scores include dataset/split, threshold and label convention.
- All latency numbers include hardware and measurement protocol.
- Any unmet target is reported honestly; a target is not an acceptance gate unless explicitly agreed by the project owner.
