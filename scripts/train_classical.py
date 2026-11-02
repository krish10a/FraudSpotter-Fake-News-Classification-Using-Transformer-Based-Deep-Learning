"""
FraudSpotter - Memory-Efficient Classical Baseline Training Script.

Executes sequential training on local Windows CPU with strict memory management:
  1. Majority Baseline (Sanity Floor) -> artifacts/models/classical/majority/
  2. Multinomial Naive Bayes (Word TF-IDF, Float32) -> artifacts/models/classical/naive_bayes/
  3. Logistic Regression (Word-only TF-IDF, Float32) -> artifacts/models/classical/logistic_word/
  4. Logistic Regression (Combined Word + Char TF-IDF, Float32) -> artifacts/models/classical/logistic_word_char/
  5. Linear SVM (Optional Baseline, Float32) -> artifacts/models/classical/linear_svm/

Evaluation reports:
  - reports/evaluations/majority_validation.json
  - reports/evaluations/naive_bayes_validation.json
  - reports/evaluations/logistic_word_validation.json
  - reports/evaluations/logistic_word_char_validation.json
  - reports/evaluations/linear_svm_validation.json
  - reports/evaluations/classical_validation_metrics.json
  - reports/evaluations/memory_profile.json
  - reports/evaluations/truncation_report.json
  - reports/evaluations/model_comparison.csv

Key Memory Safeguards:
  - Thread affinity capped at 4 (OMP, OPENBLAS, MKL)
  - Float32 sparse CSR matrices exclusively
  - Word features capped at 100,000 max; Char features capped at 50,000 max with min_df=3
  - Configurable text length truncation (default 8,000 chars)
  - Memory profiling (Process RSS + Available RAM) at every pipeline stage
  - Intermediate model artifact and metrics persistence
  - Zero access to test partition (only train and val/validation)
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Enforce thread limits before importing BLAS-dependent libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Ensure 'src' is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from fraudspotter.common.config import load_config
from fraudspotter.common.logger import setup_logger
from fraudspotter.common.seed import set_seed
from fraudspotter.data.loader import load_split_data
from fraudspotter.evaluation.metrics import compute_classification_metrics
from fraudspotter.models.classical import (
    MajorityClassifier,
    build_char_vectorizer,
    build_word_vectorizer,
    save_model_bundle,
    tune_linear_svm,
    tune_logistic_regression,
    tune_multinomial_nb,
)

logger = setup_logger("fraudspotter.train_classical")


def get_memory_info() -> Dict[str, float]:
    """Get current process memory (RSS) and system available memory in GB."""
    process_rss_gb = 0.0
    system_available_gb = 0.0

    try:
        import psutil

        proc = psutil.Process()
        process_rss_gb = proc.memory_info().rss / (1024**3)
        system_available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        try:
            status_path = Path("/proc/self/status")
            if status_path.exists():
                with open(status_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            process_rss_gb = kb / (1024**2)
                            break
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.exists():
                with open(meminfo_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "MemAvailable:" in line:
                            kb = int(line.split()[1])
                            system_available_gb = kb / (1024**2)
                            break
        except Exception:
            pass

    return {
        "process_rss_gb": round(process_rss_gb, 3),
        "system_available_gb": round(system_available_gb, 3),
    }


def log_memory_stage(stage_name: str) -> Dict[str, float]:
    """Log current process memory and available system RAM at a given milestone."""
    mem = get_memory_info()
    rss = mem["process_rss_gb"]
    avail = mem["system_available_gb"]
    avail_str = f"{avail:.2f} GB" if avail > 0 else "N/A"
    logger.info(
        f"[RAM Stage: {stage_name}] Process RSS: {rss:.2f} GB | Available System RAM: {avail_str}"
    )
    return mem


def resolve_split_path(
    explicit_path: Optional[str],
    drive_dir: Optional[str],
    split_name: str,
) -> Path:
    """Resolve split path checking explicit path, drive_dir, parquet, and csv.

    Strictly supports training and validation splits. The test partition is intentionally
    excluded to guarantee zero test leakage during baseline model selection and tuning.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p

    aliases = [split_name]
    if split_name in ["val", "validation"]:
        aliases = ["validation", "val"]
    elif split_name in ["train", "training"]:
        aliases = ["train", "training"]

    search_dirs = []
    if drive_dir:
        search_dirs.append(Path(drive_dir))
    search_dirs.extend(
        [
            Path("data/processed/welfake_v1"),
            Path("data/processed/welfake"),
            Path("/content/drive/MyDrive/FraudSpotter/data/processed/welfake_v1"),
            Path("/content/drive/MyDrive/FraudSpotter_Data"),
        ]
    )

    for s_dir in search_dirs:
        for alias in aliases:
            for ext in [".parquet", ".pq", ".csv"]:
                candidate = s_dir / f"{alias}{ext}"
                if candidate.is_file():
                    return candidate

    return Path(explicit_path or f"data/processed/welfake_v1/{split_name}.parquet")


def run_dynamic_memory_preflight(
    X_train: List[str],
    X_val: List[str],
    max_chars: Optional[int],
    char_min_df: int,
) -> Dict[str, Any]:
    """Execute rigorous preflight checking real text length distributions and system RAM."""
    gc.collect()
    mem = get_memory_info()

    lengths = [len(t) for t in X_train]
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    median_len = float(np.median(lengths)) if lengths else 0.0
    p95_len = float(np.percentile(lengths, 95)) if lengths else 0.0
    max_len = int(np.max(lengths)) if lengths else 0

    over_limit_count = sum(1 for ln in lengths if ln > (max_chars or 100000))

    logger.info("================ RESOURCE & MEMORY PREFLIGHT AUDIT ================")
    logger.info(f"Training partition size:     {len(X_train):,} rows")
    logger.info(f"Validation partition size:   {len(X_val):,} rows")
    logger.info(
        f"Text character distribution: Mean={mean_len:.0f}, Median={median_len:.0f}, P95={p95_len:.0f}, Max={max_len:,}"
    )
    if max_chars:
        logger.info(
            f"Configured max_chars limit:  {max_chars:,} (Articles exceeding limit: {over_limit_count:,} / {len(X_train):,})"
        )
    logger.info(f"Character TF-IDF min_df:     {char_min_df} (filters rare character noise)")
    logger.info(f"Current Process Memory:      {mem['process_rss_gb']:.2f} GB")
    logger.info(
        f"Available System RAM:        {mem['system_available_gb']:.2f} GB"
        if mem["system_available_gb"] > 0
        else "Available System RAM:        N/A"
    )

    # Estimate sparse CSR memory footprint (float32 precision = 4 bytes data + 4 bytes index = 8 bytes per nnz)
    effective_char_len = min(mean_len, max_chars) if max_chars else mean_len
    estimated_word_nnz = min(600, int(effective_char_len * 0.08))
    estimated_char_nnz = min(1000, int(effective_char_len * 0.12))

    projected_word_csr_mb = (len(X_train) * estimated_word_nnz * 8) / (1024**2)
    projected_comb_csr_mb = (len(X_train) * (estimated_word_nnz + estimated_char_nnz) * 8) / (
        1024**2
    )

    logger.info(f"Projected Word CSR Memory:   ~{projected_word_csr_mb:.1f} MB (float32)")
    logger.info(
        f"Projected Comb CSR Memory:   ~{projected_comb_csr_mb:.1f} MB (float32, word+char)"
    )

    # Preflight safety assessment
    is_safe = True
    warning_msg = None
    if mem["system_available_gb"] > 0 and mem["system_available_gb"] < 3.0:
        is_safe = False
        warning_msg = (
            f"Available RAM ({mem['system_available_gb']:.2f} GB) is below 3.0 GB safety threshold. "
            "Character TF-IDF vocabulary construction may risk memory pressure."
        )
        logger.warning(f"PREFLIGHT ALERT: {warning_msg}")
    elif mem["system_available_gb"] > 0 and mem["system_available_gb"] < 4.5:
        logger.warning(
            f"Available RAM ({mem['system_available_gb']:.2f} GB) is moderate. "
            "Running sequential isolated models with aggressive gc.collect() is required."
        )
    else:
        logger.info(
            "Preflight Safety Status:     [PASS] Memory headroom verified for isolated execution."
        )

    logger.info("===================================================================\n")

    return {
        "mean_length": mean_len,
        "median_length": median_len,
        "p95_length": p95_len,
        "max_length": max_len,
        "over_limit_count": over_limit_count,
        "projected_word_csr_mb": projected_word_csr_mb,
        "projected_comb_csr_mb": projected_comb_csr_mb,
        "system_available_gb_at_preflight": mem["system_available_gb"],
        "process_rss_gb_at_preflight": mem["process_rss_gb"],
        "is_safe": is_safe,
        "warning_msg": warning_msg,
    }


def compute_uncalibrated_svm_metrics(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
) -> Dict[str, Any]:
    """Compute classification metrics for uncalibrated models (e.g. Linear SVM with raw margins).

    Predictions are determined by sign(margin) > 0.
    ROC-AUC and PR-AUC are computed on continuous decision margins.
    Brier score and ECE are returned as None (not applicable to uncalibrated margin scores).
    """
    preds = (decision_scores > 0).astype(int)
    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    fake_f1 = float(f1_score(y_true, preds, pos_label=1))
    real_f1 = float(f1_score(y_true, preds, pos_label=0))
    prec = float(precision_score(y_true, preds, average="macro"))
    rec = float(recall_score(y_true, preds, average="macro"))
    mcc = float(matthews_corrcoef(y_true, preds))

    try:
        roc_auc = float(roc_auc_score(y_true, decision_scores))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, decision_scores))
    except Exception:
        pr_auc = 0.5

    report_dict = classification_report(
        y_true, preds, target_names=["REAL", "FAKE"], output_dict=True
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "fake_f1": fake_f1,
        "real_f1": real_f1,
        "macro_precision": prec,
        "macro_recall": rec,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": None,
        "expected_calibration_error": None,
        "classification_report": report_dict,
    }


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Safely persist JSON dictionary."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_file = filepath.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_file.replace(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient classical baseline training for FraudSpotter on local Windows CPU."
    )
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train_file", default=None, help="Path to train.parquet or train.csv")
    parser.add_argument(
        "--val_file", default=None, help="Path to validation.parquet or val.parquet"
    )
    parser.add_argument(
        "--drive_dir",
        default=None,
        help="Optional fallback directory for train/val Parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/models/classical",
        help="Base directory to save trained model bundles",
    )
    parser.add_argument(
        "--reports_dir",
        default="reports/evaluations",
        help="Directory to save validation reports, metrics JSONs, and CSV summary",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "majority",
            "naive_bayes",
            "logistic_word",
            "logistic_word_char",
            "linear_svm",
        ],
        help="Specific models to train, or 'all'",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help="Optional character length truncation threshold (default: 8000)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional pilot sub-sample size (e.g. 15000) for exploratory benchmarking",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        default=False,
        help="Run memory and dataset preflight checks and exit without training",
    )
    parser.add_argument(
        "--include_svm",
        action="store_true",
        default=False,
        help="Include optional Linear SVM baseline in training",
    )
    args = parser.parse_args()

    # Determine which models to run
    models_to_run = set(args.models)
    if "all" in models_to_run:
        models_to_run = {"majority", "naive_bayes", "logistic_word", "logistic_word_char"}
        if args.include_svm:
            models_to_run.add("linear_svm")
    elif args.include_svm:
        models_to_run.add("linear_svm")

    # Load configuration
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    classical_cfg = config.get("classical", {})
    cfg_text_length = classical_cfg.get("text_length", {})
    max_chars = (
        args.max_chars
        if args.max_chars is not None
        else (
            cfg_text_length.get("max_chars", 8000) if cfg_text_length.get("enabled", True) else None
        )
    )

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(args.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve and Load Datasets
    train_path = resolve_split_path(args.train_file, args.drive_dir, "train")
    val_path = resolve_split_path(args.val_file, args.drive_dir, "val")

    if not train_path.is_file():
        logger.error(f"Training split not found: {train_path}")
        return
    if not val_path.is_file():
        logger.error(f"Validation split not found: {val_path}")
        return

    memory_timeline: Dict[str, Dict[str, float]] = {}
    memory_timeline["init"] = log_memory_stage("Init - Before Loading Data")

    logger.info(f"Loading training split from: {train_path}")
    train_df, y_train_series = load_split_data(str(train_path))
    logger.info(f"Loading validation split from: {val_path}")
    val_df, y_val_series = load_split_data(str(val_path))

    X_train_raw = train_df["content"].tolist()
    y_train = y_train_series.values
    X_val_raw = val_df["content"].tolist()
    y_val = y_val_series.values

    # Free DataFrame memory immediately
    del train_df, val_df, y_train_series, y_val_series
    gc.collect()

    memory_timeline["after_data_load"] = log_memory_stage("After Data Load")

    # Optional Pilot Sub-sampling
    is_pilot = args.sample_size is not None and args.sample_size < len(y_train)
    if is_pilot:
        logger.info(f"--- Running Exploratory Pilot Run ({args.sample_size:,} samples) ---")
        X_train_raw, _, y_train, _ = train_test_split(
            X_train_raw,
            y_train,
            train_size=args.sample_size,
            random_state=seed,
            stratify=y_train,
        )
        experiment_type = f"pilot_sample_{args.sample_size}"
    else:
        logger.info(f"--- Running Full-Dataset Benchmark ({len(y_train):,} samples) ---")
        experiment_type = "full_dataset_final"

    # Text length handling & truncation tracking
    train_truncated = 0
    val_truncated = 0
    if max_chars:
        logger.info(f"Applying text length truncation limit of {max_chars:,} characters...")
        X_train_processed = []
        for text in X_train_raw:
            if len(text) > max_chars:
                train_truncated += 1
                X_train_processed.append(text[:max_chars])
            else:
                X_train_processed.append(text)
        X_train_raw = X_train_processed

        X_val_processed = []
        for text in X_val_raw:
            if len(text) > max_chars:
                val_truncated += 1
                X_val_processed.append(text[:max_chars])
            else:
                X_val_processed.append(text)
        X_val_raw = X_val_processed

    train_counts = np.bincount(y_train, minlength=2)
    val_counts = np.bincount(y_val, minlength=2)

    logger.info("=== Internal Label Distribution (0=REAL, 1=FAKE) ===")
    logger.info(
        f"Train: Total={len(y_train):,}, REAL(0)={train_counts[0]:,} ({train_counts[0] / len(y_train):.1%}), "
        f"FAKE(1)={train_counts[1]:,} ({train_counts[1] / len(y_train):.1%})"
    )
    logger.info(
        f"Val:   Total={len(y_val):,}, REAL(0)={val_counts[0]:,} ({val_counts[0] / len(y_val):.1%}), "
        f"FAKE(1)={val_counts[1]:,} ({val_counts[1] / len(y_val):.1%})"
    )
    if max_chars:
        logger.info(
            f"Truncation: Train={train_truncated:,}/{len(y_train):,} ({train_truncated / len(y_train):.1%}), "
            f"Val={val_truncated:,}/{len(y_val):,} ({val_truncated / len(y_val):.1%})"
        )

    # Truncation report
    truncation_report = {
        "max_chars": max_chars,
        "train_total": len(y_train),
        "train_truncated_count": train_truncated,
        "train_truncated_pct": round(train_truncated / len(y_train), 4) if y_train.size else 0.0,
        "val_total": len(y_val),
        "val_truncated_count": val_truncated,
        "val_truncated_pct": round(val_truncated / len(y_val), 4) if y_val.size else 0.0,
    }
    save_json(truncation_report, rep_dir / "truncation_report.json")

    # 2. Dynamic Preflight Check
    char_cfg = classical_cfg.get("char_tfidf", {})
    char_min_df = char_cfg.get("min_df", 3)
    preflight_info = run_dynamic_memory_preflight(
        X_train_raw, X_val_raw, max_chars=max_chars, char_min_df=char_min_df
    )

    if args.preflight:
        logger.info("Preflight flag requested. Exiting without training.")
        return

    word_cfg = classical_cfg.get("word_tfidf", {})
    word_max_feat = word_cfg.get("max_features", 100000)
    char_max_feat = char_cfg.get("max_features", 50000)

    consolidated_metrics: Dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "experiment_type": experiment_type,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "train_label_distribution": {
                "REAL_0": int(train_counts[0]),
                "FAKE_1": int(train_counts[1]),
            },
            "val_label_distribution": {"REAL_0": int(val_counts[0]), "FAKE_1": int(val_counts[1])},
            "label_convention": {"0": "REAL", "1": "FAKE"},
            "truncation_report": truncation_report,
            "feature_limits": {
                "word_max_features": word_max_feat,
                "char_max_features": char_max_feat,
                "char_min_df": char_min_df,
                "dtype": classical_cfg.get("dtype", "float32"),
            },
            "preflight_audit": preflight_info,
            "train_file": str(train_path),
            "val_file": str(val_path),
        },
        "models": {},
    }

    # Helper for saving unified metrics and memory
    def flush_reports():
        save_json(consolidated_metrics, rep_dir / "classical_validation_metrics.json")
        save_json(memory_timeline, rep_dir / "memory_profile.json")

    # -------------------------------------------------------------
    # STAGE 1: Majority Sanity Baseline
    # -------------------------------------------------------------
    if "majority" in models_to_run:
        logger.info("\n--- [STAGE 1] Training Majority Sanity Baseline ---")
        maj_pipeline = Pipeline(
            [
                ("features", build_word_vectorizer(max_features=10, min_df=1)),
                ("clf", MajorityClassifier()),
            ]
        )
        maj_pipeline.fit(X_train_raw, y_train)

        maj_preds = maj_pipeline.predict(X_val_raw)
        maj_probs = maj_pipeline.predict_proba(X_val_raw)
        maj_metrics = compute_classification_metrics(y_val, maj_preds, maj_probs)

        maj_meta = {
            "selected_hyperparameters": {
                "majority_class": int(maj_pipeline.named_steps["clf"].majority_class_)
            },
            "label_mapping": {"0": "REAL", "1": "FAKE"},
            "train_samples": len(y_train),
        }
        maj_dir = save_model_bundle(
            output_dir=out_base,
            model_name="majority",
            pipeline=maj_pipeline,
            vectorizer=maj_pipeline.named_steps["features"],
            metadata=maj_meta,
        )

        maj_report = {
            "model_name": "majority",
            "artifact_dir": str(maj_dir),
            "hyperparameters": maj_meta["selected_hyperparameters"],
            "validation_metrics": maj_metrics,
        }
        save_json(maj_report, rep_dir / "majority_validation.json")
        consolidated_metrics["models"]["majority"] = maj_report

        memory_timeline["after_majority"] = log_memory_stage("After Majority Baseline")
        flush_reports()
        logger.info(
            f"Majority Baseline -> Accuracy: {maj_metrics['accuracy']:.4f}, Macro-F1: {maj_metrics['macro_f1']:.4f}"
        )

        del maj_pipeline, maj_preds, maj_probs
        gc.collect()

    # -------------------------------------------------------------
    # STAGE 2: Multinomial Naive Bayes (Word TF-IDF, Float32)
    # -------------------------------------------------------------
    if "naive_bayes" in models_to_run:
        logger.info("\n--- [STAGE 2] Fitting Word TF-IDF for Multinomial Naive Bayes ---")
        word_vec_nb = build_word_vectorizer(
            ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
            min_df=word_cfg.get("min_df", 2),
            max_df=word_cfg.get("max_df", 0.95),
            max_features=word_max_feat,
            sublinear_tf=word_cfg.get("sublinear_tf", True),
            dtype=np.float32,
        )

        memory_timeline["nb_before_vectorize"] = log_memory_stage("NB - Before Word Vectorize")
        X_train_nb_sp = word_vec_nb.fit_transform(X_train_raw)
        X_val_nb_sp = word_vec_nb.transform(X_val_raw)
        memory_timeline["nb_after_vectorize"] = log_memory_stage("NB - After Word Vectorize")

        logger.info(
            f"Word TF-IDF vocabulary size: {X_train_nb_sp.shape[1]:,} features (dtype={X_train_nb_sp.dtype})"
        )

        nb_cfg = classical_cfg.get("multinomial_nb", {})
        alpha_candidates = nb_cfg.get("alpha_candidates", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0])

        logger.info(f"Tuning MultinomialNB across alpha candidates: {alpha_candidates}")
        best_nb, nb_tuning_history = tune_multinomial_nb(
            X_train=X_train_nb_sp,
            y_train=y_train,
            X_val=X_val_nb_sp,
            y_val=y_val,
            alpha_candidates=alpha_candidates,
            fit_prior=nb_cfg.get("fit_prior", True),
        )

        nb_preds = best_nb.predict(X_val_nb_sp)
        nb_probs = best_nb.predict_proba(X_val_nb_sp)
        nb_metrics = compute_classification_metrics(y_val, nb_preds, nb_probs)

        nb_pipeline = Pipeline([("features", word_vec_nb), ("clf", best_nb)])
        nb_meta = {
            "selected_hyperparameters": {"alpha": float(best_nb.alpha)},
            "feature_count": int(X_train_nb_sp.shape[1]),
            "tuning_history": nb_tuning_history,
            "label_mapping": {"0": "REAL", "1": "FAKE"},
        }
        nb_dir = save_model_bundle(
            output_dir=out_base,
            model_name="naive_bayes",
            pipeline=nb_pipeline,
            vectorizer=word_vec_nb,
            metadata=nb_meta,
        )

        nb_report = {
            "model_name": "naive_bayes",
            "artifact_dir": str(nb_dir),
            "hyperparameters": nb_meta["selected_hyperparameters"],
            "tuning_history": nb_tuning_history,
            "validation_metrics": nb_metrics,
            "feature_count": nb_meta["feature_count"],
        }
        save_json(nb_report, rep_dir / "naive_bayes_validation.json")
        consolidated_metrics["models"]["naive_bayes"] = nb_report

        memory_timeline["after_nb_training"] = log_memory_stage("After NB Training")
        flush_reports()
        logger.info(
            f"MultinomialNB (Best alpha={best_nb.alpha}) -> "
            f"Accuracy: {nb_metrics['accuracy']:.4f}, Macro-F1: {nb_metrics['macro_f1']:.4f}, "
            f"ROC-AUC: {nb_metrics.get('roc_auc', 0.0):.4f}"
        )

        # Explicitly release Naive Bayes matrices and trigger garbage collection
        logger.info("Releasing Naive Bayes matrices and collecting garbage...")
        del X_train_nb_sp, X_val_nb_sp, best_nb, nb_pipeline, word_vec_nb
        gc.collect()
        memory_timeline["after_nb_cleanup"] = log_memory_stage("After NB Cleanup")

    # -------------------------------------------------------------
    # STAGE 3: Logistic Regression (Word-only TF-IDF, Float32)
    # -------------------------------------------------------------
    if "logistic_word" in models_to_run:
        logger.info("\n--- [STAGE 3] Training Word-Only Logistic Regression ---")
        memory_timeline["lr_word_before_features"] = log_memory_stage("LR Word - Before Features")

        word_vec_lr = build_word_vectorizer(
            ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
            min_df=word_cfg.get("min_df", 2),
            max_df=word_cfg.get("max_df", 0.95),
            max_features=word_max_feat,
            sublinear_tf=word_cfg.get("sublinear_tf", True),
            dtype=np.float32,
        )
        X_train_w_sp = word_vec_lr.fit_transform(X_train_raw)
        X_val_w_sp = word_vec_lr.transform(X_val_raw)
        memory_timeline["lr_word_after_features"] = log_memory_stage("LR Word - After Features")

        lr_cfg = classical_cfg.get("logistic_word", classical_cfg.get("logistic_regression", {}))
        c_candidates = lr_cfg.get("C_candidates", [1.0, 2.0, 4.0])
        solver_candidates = lr_cfg.get("solver_candidates", ["lbfgs", "liblinear"])

        logger.info(
            f"Tuning Word-only Logistic Regression across Solvers {solver_candidates} and C {c_candidates}..."
        )
        best_lr_w, best_lr_w_c, best_lr_w_solver, lr_w_tuning_history = tune_logistic_regression(
            X_train=X_train_w_sp,
            y_train=y_train,
            X_val=X_val_w_sp,
            y_val=y_val,
            C_candidates=c_candidates,
            solver=lr_cfg.get("solver", "lbfgs"),
            solver_candidates=solver_candidates,
            max_iter=lr_cfg.get("max_iter", 2000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            random_state=seed,
        )

        lr_w_preds = best_lr_w.predict(X_val_w_sp)
        lr_w_probs = best_lr_w.predict_proba(X_val_w_sp)
        lr_w_metrics = compute_classification_metrics(y_val, lr_w_preds, lr_w_probs)

        lr_w_pipeline = Pipeline([("features", word_vec_lr), ("clf", best_lr_w)])
        lr_w_meta = {
            "selected_hyperparameters": {"C": float(best_lr_w_c), "solver": best_lr_w_solver},
            "feature_count": int(X_train_w_sp.shape[1]),
            "tuning_history": lr_w_tuning_history,
            "label_mapping": {"0": "REAL", "1": "FAKE"},
        }
        lr_w_dir = save_model_bundle(
            output_dir=out_base,
            model_name="logistic_word",
            pipeline=lr_w_pipeline,
            vectorizer=word_vec_lr,
            metadata=lr_w_meta,
        )

        lr_w_report = {
            "model_name": "logistic_word",
            "artifact_dir": str(lr_w_dir),
            "hyperparameters": lr_w_meta["selected_hyperparameters"],
            "tuning_history": lr_w_tuning_history,
            "validation_metrics": lr_w_metrics,
            "feature_count": lr_w_meta["feature_count"],
        }
        save_json(lr_w_report, rep_dir / "logistic_word_validation.json")
        consolidated_metrics["models"]["logistic_word"] = lr_w_report

        memory_timeline["after_lr_word_training"] = log_memory_stage("After LR Word Training")
        flush_reports()
        logger.info(
            f"Logistic Regression Word (Best C={best_lr_w_c}, Solver={best_lr_w_solver}) -> "
            f"Accuracy: {lr_w_metrics['accuracy']:.4f}, Macro-F1: {lr_w_metrics['macro_f1']:.4f}, "
            f"ROC-AUC: {lr_w_metrics.get('roc_auc', 0.0):.4f}"
        )

        del X_train_w_sp, X_val_w_sp, best_lr_w, lr_w_pipeline, word_vec_lr
        gc.collect()
        memory_timeline["after_lr_word_cleanup"] = log_memory_stage("After LR Word Cleanup")

    # -------------------------------------------------------------
    # STAGE 4: Logistic Regression (Combined Word + Char, Float32)
    # -------------------------------------------------------------
    X_train_comb_sp = None
    X_val_comb_sp = None
    combined_features = None

    if "logistic_word_char" in models_to_run or "linear_svm" in models_to_run:
        logger.info("\n--- [STAGE 4] Building Word + Char TF-IDF Features for Combined Models ---")
        memory_timeline["comb_before_features"] = log_memory_stage("Combined - Before Features")

        # 4a. Word TF-IDF
        logger.info(f"Fitting Word TF-IDF (max_features={word_max_feat:,}, float32)...")
        word_vec_comb = build_word_vectorizer(
            ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
            min_df=word_cfg.get("min_df", 2),
            max_df=word_cfg.get("max_df", 0.95),
            max_features=word_max_feat,
            sublinear_tf=word_cfg.get("sublinear_tf", True),
            dtype=np.float32,
        )
        X_train_w = word_vec_comb.fit_transform(X_train_raw)
        X_val_w = word_vec_comb.transform(X_val_raw)
        logger.info(f"Word features extracted: {X_train_w.shape[1]:,} columns.")

        # 4b. Char TF-IDF
        logger.info(
            f"Fitting Character TF-IDF (max_features={char_max_feat:,}, min_df={char_min_df}, float32)..."
        )
        char_vec_comb = build_char_vectorizer(
            ngram_range=tuple(char_cfg.get("ngram_range", [3, 5])),
            min_df=char_min_df,
            max_df=char_cfg.get("max_df", 0.95),
            max_features=char_max_feat,
            sublinear_tf=char_cfg.get("sublinear_tf", True),
            dtype=np.float32,
        )
        X_train_c = char_vec_comb.fit_transform(X_train_raw)
        X_val_c = char_vec_comb.transform(X_val_raw)
        logger.info(f"Char features extracted: {X_train_c.shape[1]:,} columns.")

        # 4c. Stack without dense conversion
        logger.info("Combining Word + Char sparse matrices using scipy.sparse.hstack...")
        X_train_comb_sp = sparse.hstack([X_train_w, X_train_c], format="csr")
        X_val_comb_sp = sparse.hstack([X_val_w, X_val_c], format="csr")

        del X_train_w, X_train_c, X_val_w, X_val_c
        gc.collect()

        combined_features = FeatureUnion(
            [
                ("word_tfidf", word_vec_comb),
                ("char_tfidf", char_vec_comb),
            ]
        )

        memory_timeline["comb_after_hstack"] = log_memory_stage("Combined - After Feature hstack")
        logger.info(
            f"Total Combined Features: {X_train_comb_sp.shape[1]:,} columns "
            f"(dtype={X_train_comb_sp.dtype}, sparsity={1.0 - (X_train_comb_sp.nnz / (X_train_comb_sp.shape[0] * X_train_comb_sp.shape[1])):.4%})"
        )

    if "logistic_word_char" in models_to_run and X_train_comb_sp is not None:
        logger.info("\n--- Tuning Word + Char Logistic Regression ---")
        lr_cfg = classical_cfg.get(
            "logistic_word_char", classical_cfg.get("logistic_regression", {})
        )
        c_candidates = lr_cfg.get("C_candidates", [1.0, 2.0, 4.0])
        solver_candidates = lr_cfg.get("solver_candidates", ["lbfgs", "liblinear"])

        best_lr_wc, best_lr_wc_c, best_lr_wc_solver, lr_wc_tuning_history = (
            tune_logistic_regression(
                X_train=X_train_comb_sp,
                y_train=y_train,
                X_val=X_val_comb_sp,
                y_val=y_val,
                C_candidates=c_candidates,
                solver=lr_cfg.get("solver", "lbfgs"),
                solver_candidates=solver_candidates,
                max_iter=lr_cfg.get("max_iter", 2000),
                class_weight=lr_cfg.get("class_weight", "balanced"),
                random_state=seed,
            )
        )

        lr_wc_preds = best_lr_wc.predict(X_val_comb_sp)
        lr_wc_probs = best_lr_wc.predict_proba(X_val_comb_sp)
        lr_wc_metrics = compute_classification_metrics(y_val, lr_wc_preds, lr_wc_probs)

        lr_wc_pipeline = Pipeline([("features", combined_features), ("clf", best_lr_wc)])
        lr_wc_meta = {
            "selected_hyperparameters": {"C": float(best_lr_wc_c), "solver": best_lr_wc_solver},
            "feature_count": int(X_train_comb_sp.shape[1]),
            "tuning_history": lr_wc_tuning_history,
            "label_mapping": {"0": "REAL", "1": "FAKE"},
        }
        lr_wc_dir = save_model_bundle(
            output_dir=out_base,
            model_name="logistic_word_char",
            pipeline=lr_wc_pipeline,
            vectorizer=combined_features,
            metadata=lr_wc_meta,
        )

        lr_wc_report = {
            "model_name": "logistic_word_char",
            "artifact_dir": str(lr_wc_dir),
            "hyperparameters": lr_wc_meta["selected_hyperparameters"],
            "tuning_history": lr_wc_tuning_history,
            "validation_metrics": lr_wc_metrics,
            "feature_count": lr_wc_meta["feature_count"],
        }
        save_json(lr_wc_report, rep_dir / "logistic_word_char_validation.json")
        consolidated_metrics["models"]["logistic_word_char"] = lr_wc_report

        memory_timeline["after_lr_word_char_training"] = log_memory_stage(
            "After LR Word+Char Training"
        )
        flush_reports()
        logger.info(
            f"Logistic Regression Word+Char (Best C={best_lr_wc_c}, Solver={best_lr_wc_solver}) -> "
            f"Accuracy: {lr_wc_metrics['accuracy']:.4f}, Macro-F1: {lr_wc_metrics['macro_f1']:.4f}, "
            f"ROC-AUC: {lr_wc_metrics.get('roc_auc', 0.0):.4f}"
        )

        del best_lr_wc, lr_wc_pipeline
        gc.collect()

    # -------------------------------------------------------------
    # STAGE 5: Linear SVM (Model E - Optional Baseline)
    # -------------------------------------------------------------
    if "linear_svm" in models_to_run and X_train_comb_sp is not None:
        logger.info("\n--- [STAGE 5] Tuning Linear SVM Baseline ---")
        svm_cfg = classical_cfg.get("linear_svm", {})
        svm_c_candidates = svm_cfg.get("C_candidates", [0.5, 1.0, 2.0])
        use_calibration = svm_cfg.get("use_calibration", False)

        best_svm, best_svm_c, svm_tuning_history = tune_linear_svm(
            X_train=X_train_comb_sp,
            y_train=y_train,
            X_val=X_val_comb_sp,
            y_val=y_val,
            C_candidates=svm_c_candidates,
            use_calibration=use_calibration,
            max_iter=svm_cfg.get("max_iter", 2000),
            class_weight=svm_cfg.get("class_weight", "balanced"),
            random_state=seed,
        )

        if use_calibration:
            svm_preds = best_svm.predict(X_val_comb_sp)
            svm_probs = best_svm.predict_proba(X_val_comb_sp)
            svm_metrics = compute_classification_metrics(y_val, svm_preds, svm_probs)
        else:
            svm_dec = best_svm.decision_function(X_val_comb_sp)
            svm_metrics = compute_uncalibrated_svm_metrics(y_val, svm_dec)

        svm_pipeline = Pipeline([("features", combined_features), ("clf", best_svm)])
        svm_meta = {
            "selected_hyperparameters": {"C": float(best_svm_c), "calibrated": use_calibration},
            "feature_count": int(X_train_comb_sp.shape[1]),
            "tuning_history": svm_tuning_history,
            "label_mapping": {"0": "REAL", "1": "FAKE"},
        }
        svm_dir = save_model_bundle(
            output_dir=out_base,
            model_name="linear_svm",
            pipeline=svm_pipeline,
            vectorizer=combined_features,
            metadata=svm_meta,
        )

        svm_report = {
            "model_name": "linear_svm",
            "artifact_dir": str(svm_dir),
            "hyperparameters": svm_meta["selected_hyperparameters"],
            "tuning_history": svm_tuning_history,
            "validation_metrics": svm_metrics,
            "feature_count": svm_meta["feature_count"],
        }
        save_json(svm_report, rep_dir / "linear_svm_validation.json")
        consolidated_metrics["models"]["linear_svm"] = svm_report

        memory_timeline["after_svm_training"] = log_memory_stage("After SVM Training")
        flush_reports()
        logger.info(
            f"Linear SVM (Best C={best_svm_c}, Calibrated={use_calibration}) -> "
            f"Accuracy: {svm_metrics['accuracy']:.4f}, Macro-F1: {svm_metrics['macro_f1']:.4f}, "
            f"ROC-AUC: {svm_metrics.get('roc_auc', 0.0):.4f}"
        )

        del best_svm, svm_pipeline
        gc.collect()

    if X_train_comb_sp is not None:
        del X_train_comb_sp, X_val_comb_sp, combined_features
        gc.collect()

    memory_timeline["final_cleanup"] = log_memory_stage("Final Cleanup")
    flush_reports()

    # Generate Model Comparison CSV
    summary_rows = []
    for m_name, m_data in consolidated_metrics["models"].items():
        v = m_data["validation_metrics"]
        summary_rows.append(
            {
                "model": m_name,
                "macro_f1": v.get("macro_f1"),
                "accuracy": v.get("accuracy"),
                "fake_f1": v.get("fake_f1"),
                "real_f1": v.get("real_f1"),
                "macro_precision": v.get("macro_precision"),
                "macro_recall": v.get("macro_recall"),
                "roc_auc": v.get("roc_auc"),
                "pr_auc": v.get("pr_auc"),
                "mcc": v.get("mcc"),
                "brier_score": v.get("brier_score"),
                "expected_calibration_error": v.get("expected_calibration_error"),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = rep_dir / "model_comparison.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Model comparison table saved to: {csv_path}")

    # Summary Table Display in Console
    logger.info("\n==================== VALIDATION BENCHMARK SUMMARY ====================")
    logger.info(
        f"{'Model':<22} | {'Macro-F1':<10} | {'Accuracy':<10} | {'Fake-F1':<10} | {'ROC-AUC':<10} | {'Brier':<10}"
    )
    logger.info("-" * 82)
    for row in summary_rows:
        roc_str = f"{row['roc_auc']:.4f}" if row["roc_auc"] is not None else "N/A"
        brier_str = f"{row['brier_score']:.4f}" if row["brier_score"] is not None else "N/A"
        logger.info(
            f"{row['model']:<22} | {row['macro_f1']:<10.4f} | {row['accuracy']:<10.4f} | "
            f"{row.get('fake_f1', 0.0):<10.4f} | {roc_str:<10} | {brier_str:<10}"
        )
    logger.info("======================================================================")
    logger.info(f"\nAll validation artifacts & metrics safely preserved at: {rep_dir}")


if __name__ == "__main__":
    main()
