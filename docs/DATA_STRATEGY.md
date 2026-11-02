# FraudSpotter — Data Strategy

## Dataset roles

| Dataset | Role | Training? | Reporting |
|---|---|---:|---|
| WELFake | Primary full-news-article benchmark | Yes | Main train/validation/test results |
| ISOT | External full-article domain diagnostic | No for the primary model | Separate external metrics |
| LIAR | Short political-statement diagnostic | No for the primary model | Separate metrics and mapping note |

## Label mapping

Internal convention: REAL=0, FAKE=1.

For WELFake, verify original labels and apply `internal_label = 1 - original_label` only after confirming source convention `0=Fake, 1=Real`.

For LIAR, primary mapping: `pants-fire`, `false`, `barely-true` -> FAKE; `half-true`, `mostly-true`, `true` -> REAL. Run a separate sensitivity diagnostic excluding `half-true` if feasible. Do not merge LIAR with WELFake.

## WELFake data processing

1. Download in Colab from a verified remote source; save source URL, access date, file hash and license notes.
2. Validate columns, types and labels.
3. Drop records with no usable title and no usable body. Retain partial records under a consistent policy.
4. Apply Unicode NFKC and whitespace cleanup. Remove HTML artifacts conservatively. Preserve negation, punctuation, sentence order and ordinary entity/publisher mentions.
5. Strip only validated source/byline boilerplate using anchored patterns. Log every transformation rule and affected-row count.
6. Create exact fingerprints over normalized title/body. Identify near-duplicate candidates with MinHash/LSH or similar; review sampled pairs before choosing threshold.
7. Keep every duplicate cluster in one partition.
8. Generate an 80/10/10 stratified group-aware split with seed 42. Report actual ratios and class balance.
9. Assert zero exact and zero known near-duplicate group overlap across partitions.
10. Save split manifest with row IDs, hashes, group IDs, counts, class distribution, seed, code/config revision and dataset hash.

## Leakage safeguards

- Fit TF-IDF only on training data.
- Never use source, date, subject or IDs as model features.
- Keep source/topic/date fields only for audit and slice analysis when available.
- Tune hyperparameters, thresholds and calibration using validation data only.
- Do not inspect test metrics to choose model settings.
- Check WELFake/ISOT duplicate overlap before external evaluation.

## External evaluation

### ISOT
- Keep all ISOT records outside primary WELFake training and validation.
- Exclude subject/date from model input.
- Remove verified publisher/byline prefixes conservatively.
- Audit duplicates with WELFake and report excluded overlaps.
- Evaluate on the remaining external records and report separately.

### LIAR
- Use statement text only.
- State the six-to-two label mapping explicitly.
- Report the middle-label sensitivity test separately.
- Explain that short political claims are not equivalent to full news articles.

## Required data artifacts

```text
data/
  raw/                 # Colab/Drive only; do not commit
  processed/
    welfake/
      train.parquet
      validation.parquet
      test.parquet
      split_manifest.json
  external/
    isot/
    liar/
```

Use Parquet where practical. Keep raw and processed large files out of GitHub. Record hashes and reproducible download instructions.
