# FraudSpotter — Experiments and Evaluation Contract

## Main comparison

All models must be evaluated on identical frozen WELFake test records.

| Model | Accuracy | Macro-F1 | Fake precision | Fake recall | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | measured | measured | measured | measured | measured/NA | measured/NA | measured |
| MultinomialNB | measured | measured | measured | measured | measured | measured | measured |
| TF-IDF Logistic Regression | measured | measured | measured | measured | measured | measured | measured |
| LinearSVC (optional) | measured | measured | measured | measured | measured | measured | measured |
| DistilBERT | measured | measured | measured | measured | measured | measured | measured |

## Research experiments

| ID | Experiment | Comparison/output |
|---|---|---|
| E1 | Classical vs transformer | Macro-F1 and class-wise differences |
| E2 | Headline vs headline+body | Change in Macro-F1, precision and recall |
| E3 | Context length 128/256/384 | Performance vs latency/truncation |
| E4 | Raw vs source-marker-cleaned | Leakage/shortcut impact |
| E5 | Threshold sweep | Precision, recall, Macro-F1 vs threshold |
| E6 | Calibration | ECE, Brier, NLL, reliability diagram |
| E7 | Robustness | Transformed performance and prediction consistency |
| E8 | Explanation faithfulness | Top-k deletion vs random-k; sufficiency |
| E9 | External generalization | Separate ISOT and LIAR reports |
| E10 | Efficiency | Model size, latency, P95, throughput, RAM/VRAM |
| E11 | Statistical comparison | Paired bootstrap CI and McNemar |
| E12 | Error analysis | At least 50 manually reviewed test errors |

## Robustness transformations

### Meaning-preserving or near-preserving
- Case changes.
- Redundant whitespace.
- Trailing punctuation.
- Controlled punctuation duplication.
- Safe HTML/URL representation changes.
- Unicode NFKC normalization.
- Controlled spelling noise (report as noisy-input stress test; verify examples).

Measure transformed Macro-F1, prediction consistency, probability deviation and class-specific degradation.

### Information ablations
- Headline-only.
- Body-only.
- Maximum length 128/256/384.
- 10% word dropout.
- Emoji-heavy/noisy input.

Do not assume these preserve meaning; measure performance change.

## Explainability diagnostics

For a stratified sample of at least 200 test examples where feasible:
1. Explain the original predicted-class margin.
2. Remove top 1, 3 and 5 attributed tokens.
3. Remove random tokens of equal counts.
4. Keep only the top attributed tokens and measure sufficiency.
5. Aggregate results with confidence intervals.

Comprehensiveness: `score(original) - score(input_without_top_tokens)`.

Sufficiency gap: `score(original) - score(input_only_top_tokens)`.

Report the exact score being measured (prefer the original predicted-class probability or logit margin) and keep it consistent. This is an explanation diagnostic, not proof of causality.

## Statistical protocol

- Use paired, stratified bootstrap sampling of the same test rows (1,000 resamples).
- Report 95% confidence intervals for key metrics and model differences.
- Use McNemar for paired classification errors.
- If testing multiple model pairs, use an appropriate multiple-comparison correction.
- Report p-values and effect sizes; do not equate statistical significance with practical importance.

## Calibration and thresholds

- Select thresholds on validation predictions only.
- Fit temperature scaling on validation only.
- Report the 0.50 threshold and selected threshold separately.
- Report coverage and error rate of `NEEDS_REVIEW`.
- Do not refit calibration on the test set.

## Benchmark protocol

Record CPU/GPU model, runtime, precision, sequence length, batch size, thread count, warm-up count and timing method. Measure model load time, cold inference, warm median, P95, throughput, peak RAM/VRAM and artifact size. Do not compare unlike hardware configurations without disclosing the difference.

## Result storage

Write metrics programmatically to JSON/CSV. Suggested files:
- `metrics.json`
- `comparison.csv`
- `final_evaluation.json`
- `bootstrap_results.json`
- `statistical_tests.json`
- `robustness.csv`
- `faithfulness.csv`
- `external_validation.json`
- `error_analysis.csv`
- `benchmark.json`
- plots under `reports/figures/`

Never use illustrative numbers as if they were measured results.
