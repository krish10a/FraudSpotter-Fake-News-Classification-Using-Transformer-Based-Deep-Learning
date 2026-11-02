# FraudSpotter — Google Colab T4 Workflow

## Core constraint

**Do not upload large local files into Colab.** Colab must obtain source code from GitHub and datasets/pretrained weights from remote sources directly. This avoids GB-scale laptop-to-Colab uploads.

## Responsibilities

### Local Windows laptop
- Antigravity edits source code and documentation.
- Git stores source/config/tests only.
- Local tests, API, Streamlit, Docker and CPU inference.
- No large dataset/checkpoint downloads unless explicitly requested.

### Colab T4
- Clone/pull source from GitHub.
- Download WELFake and pretrained DistilBERT directly.
- Run data prep, baselines, transformer training, evaluation and explanations.
- Persist data/checkpoints/results to Google Drive.
- Export model artifacts to Drive or approved Hugging Face Hub repository.

## Initial manual setup (once)

1. In Colab select **Runtime -> Change runtime type -> T4 GPU** (if available).
2. Execute `drive.mount('/content/drive')` and authorize Google Drive.
3. Add GitHub/Kaggle/Hugging Face credentials only through Colab Secrets when needed. Never hardcode tokens in notebook cells.
4. Set the GitHub repository URL in the notebook configuration.

## Workspace layout in Drive

```text
MyDrive/FraudSpotter/
  data/raw/
  data/processed/
  models/checkpoints/
  models/final/
  models/onnx/
  reports/eda/
  reports/evaluation/
  reports/figures/
  logs/
  manifests/
```

Keep large data/checkpoints on Drive, not only in `/content`; Colab temporary storage may be lost when the runtime resets. Use `/content` as temporary working storage for speed where appropriate, then copy verified outputs to Drive.

## Notebook execution order

1. Inspect runtime: Python, torch, CUDA, GPU name/VRAM, RAM, CPU cores and disk.
2. Mount Drive safely. If already mounted, detect it and reuse it; do not forcibly delete `/content/drive` or run destructive cleanup.
3. Clone repository or pull latest code.
4. Read `AGENTS.md`, `MASTER_SPEC.md` and config files.
5. Install only missing/incompatible Colab dependencies. Do not blindly replace the Colab CUDA-enabled PyTorch build.
6. Download WELFake directly from verified remote source; record provenance and checksum.
7. Run audit, EDA, preprocessing, deduplication and frozen split creation.
8. Train classical baselines.
9. Fine-tune DistilBERT with periodic checkpoints and resume support.
10. Run validation selection and calibration.
11. Run final test evaluation only after settings are frozen.
12. Run Captum/LIME/faithfulness, robustness, external diagnostics and benchmarks.
13. Export ONNX FP32 and then INT8 only after parity checks.
14. Copy verified results/artifacts to Drive and generate an export manifest.

## Git workflow

Local:
```powershell
git status
git add AGENTS.md docs configs src scripts tests notebooks
# Review staged files carefully; ensure data/, artifacts/weights, .env and secrets are ignored.
git commit -m "docs: add canonical FraudSpotter specification"
git push -u origin main
```

Colab:
```bash
!git clone https://github.com/USERNAME/FraudSpotter.git
%cd FraudSpotter
!git pull
```

If private, use a secure authentication method. Do not paste a token into notebook source. Use Colab Secrets or an approved credential helper.

## Artifact export contract

Save at least:
- selected model checkpoint;
- tokenizer files;
- `inference_config.json` (label map, threshold, temperature, max length, model version);
- preprocessing configuration/version;
- ONNX FP32 and INT8 only if verified;
- metrics JSON/CSV, plots, benchmark and parity report;
- SHA-256 manifest and package versions.

Suggested archive (only if size is practical): `fraudspotter_release_v1.zip`. Keep a separate model directory if archive is too large. Upload/publish to Hugging Face Hub only with user approval.

## Resume and failure recovery

- Save checkpoints at epoch/step intervals to Drive.
- Save optimizer/scheduler state if training is expected to resume exactly.
- Record last completed phase and experiment ID in a manifest.
- After runtime reset: mount Drive, clone/pull code, verify checkpoint hash, resume from latest valid checkpoint.
- Never overwrite the only known-good checkpoint.

## Manual intervention rules

Stop and give the user short instructions when:
- Google Drive authorization is needed;
- runtime GPU selection is required;
- repository URL is missing;
- a private repository or dataset needs credentials;
- a license/terms decision is required;
- artifact publication needs approval.

Never request GB-scale local uploads.
