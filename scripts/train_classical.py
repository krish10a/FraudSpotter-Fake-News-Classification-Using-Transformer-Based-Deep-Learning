"""
FraudSpotter - Memory-Efficient Classical Baseline Training Script.

Executes sequential training with strict memory management:
  1. Majority Sanity Baseline (saved immediately)
  2. Word TF-IDF + Multinomial Naive Bayes (tuned on val, saved, then matrices deleted & gc.collect)
  3. Word + Char TF-IDF + Logistic Regression (features built separately, solver benchmarked, saved)
  4. (Optional) Linear SVM (disabled by default for initial full-data runs)

Key Memory Safeguards:
  - Float32 sparse matrices
  - Word features capped at 100,000 max; Char features capped at 50,000 max with min_df=3
  - Configurable text length truncation to prevent char n-gram combinatorial explosion
  - Dynamic resource preflight based on actual text length distributions
  - Memory profiling (Process RSS + Available RAM) at every pipeline stage
  - Intermediate model artifact and metrics persistence
  - Zero access to test partition (only train and val/validation)
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure 'src' is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy import sparse
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
    save_model,
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
        # Fallback probe for Linux / Colab
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
            Path("/content/drive/MyDrive/FraudSpotter/data/processed/welfake_v1"),
            Path("/content/drive/MyDrive/FraudSpotter_Data"),
            Path("data/processed/welfake_v1"),
            Path("data/processed/welfake"),
        ]
    )

    for s_dir in search_dirs:
        for alias in aliases:
            for ext in [".parquet", ".pq", ".csv"]:
                candidate = s_dir / f"{alias}{ext}"
                if candidate.is_file():
                    return candidate

    # Default fallback
    return Path(
        explicit_path
        or f"/content/drive/MyDrive/FraudSpotter/data/processed/welfake_v1/{split_name}.parquet"
    )


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

    # Estimate sparse memory:
    # 51k rows * ~800 nonzeros * 8 bytes (float32 data + int32 indices) ≈ 326 MB
    effective_char_len = min(mean_len, max_chars) if max_chars else mean_len
    estimated_nnz_per_doc = min(1200, int(effective_char_len * 0.15))
    projected_csr_mb = (len(X_train) * estimated_nnz_per_doc * 8) / (1024**2)
    logger.info(
        f"Projected CSR Sparse Memory: ~{projected_csr_mb:.1f} MB (float32 precision, word+char)"
    )

    # Preflight safety assessment
    is_safe = True
    warning_msg = None
    if mem["system_available_gb"] > 0 and mem["system_available_gb"] < 3.5:
        is_safe = False
        warning_msg = (
            f"Available RAM ({mem['system_available_gb']:.2f} GB) is under 3.5 GB threshold. "
            "Character TF-IDF intermediate vocabulary construction may risk OOM."
        )
        logger.warning(f"PREFLIGHT ALERT: {warning_msg}")
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
        "projected_csr_mb": projected_csr_mb,
        "system_available_gb_at_preflight": mem["system_available_gb"],
        "is_safe": is_safe,
        "warning_msg": warning_msg,
    }


def save_intermediate_report(report_dict: Dict[str, Any], metrics_file: Path) -> None:
    """Safely persist current metrics JSON report so completed models are preserved."""
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = metrics_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    temp_file.replace(metrics_file)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient classical baseline training for FraudSpotter."
    )
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--train_file", default=None, help="Path to train.parquet or train.csv")
    parser.add_argument(
        "--val_file", default=None, help="Path to validation.parquet or val.parquet"
    )
    parser.add_argument(
        "--drive_dir",
        default=None,
        help="Optional Google Drive directory containing train/val Parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/models/classical",
        help="Directory to save trained models and vectorizers",
    )
    parser.add_argument(
        "--metrics_file",
        default="artifacts/metrics/classical_validation_metrics.json",
        help="JSON file path to save validation metrics",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help="Optional truncation threshold for text characters (e.g. 8000)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional pilot sub-sample size (e.g. 15000) for exploratory benchmarking",
    )
    parser.add_argument(
        "--include_svm",
        action="store_true",
        default=False,
        help="Include optional Linear SVM baseline in training (disabled by default)",
    )
    args = parser.parse_args()

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve and Load Datasets
    train_path = resolve_split_path(args.train_file, args.drive_dir, "train")
    val_path = resolve_split_path(args.val_file, args.drive_dir, "val")

    if not train_path.is_file():
        logger.error(f"Training split not found: {train_path}")
        return
    if not val_path.is_file():
        logger.error(f"Validation split not found: {val_path}")
        return

    log_memory_stage("Init - Before Loading Data")
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
        logger.info(f"--- Running Full-Dataset Final Benchmark ({len(y_train):,} samples) ---")
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

    # 2. Dynamic Preflight Check
    char_cfg = classical_cfg.get("char_tfidf", {})
    char_min_df = char_cfg.get("min_df", 3)
    preflight_info = run_dynamic_memory_preflight(
        X_train_raw, X_val_raw, max_chars=max_chars, char_min_df=char_min_df
    )

    word_cfg = classical_cfg.get("word_tfidf", {})
    word_max_feat = word_cfg.get("max_features", 100000)
    char_max_feat = char_cfg.get("max_features", 50000)

    report: Dict[str, Any] = {
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
            "text_length_config": {
                "max_chars": max_chars,
                "train_truncated_count": train_truncated,
                "train_truncated_pct": round(train_truncated / len(y_train), 4)
                if y_train.size
                else 0.0,
                "val_truncated_count": val_truncated,
                "val_truncated_pct": round(val_truncated / len(y_val), 4) if y_val.size else 0.0,
            },
            "feature_limits": {
                "word_max_features": word_max_feat,
                "char_max_features": char_max_feat,
                "char_min_df": char_min_df,
                "dtype": classical_cfg.get("dtype", "float32"),
            },
            "preflight_audit": preflight_info,
            "train_file": str(train_path),
            "val_file": str(val_path),
            "ram_profiling_timeline": {},
        },
        "models": {},
    }

    report["metadata"]["ram_profiling_timeline"]["after_data_load"] = log_memory_stage(
        "After Data Load"
    )

    # -------------------------------------------------------------
    # STAGE 1: Majority Sanity Baseline
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 1/3] Training Majority Sanity Baseline ---")
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

    maj_path = out_dir / "majority_baseline.joblib"
    save_model(maj_pipeline, str(maj_path))

    report["models"]["majority_baseline"] = {
        "selected_hyperparameters": {
            "majority_class": int(maj_pipeline.named_steps["clf"].majority_class_)
        },
        "validation_metrics": maj_metrics,
        "artifact_path": str(maj_path),
    }
    report["metadata"]["ram_profiling_timeline"]["after_majority"] = log_memory_stage(
        "After Majority Baseline"
    )
    save_intermediate_report(report, metrics_path)
    logger.info(
        f"Majority Baseline -> Accuracy: {maj_metrics['accuracy']:.4f}, Macro-F1: {maj_metrics['macro_f1']:.4f}"
    )

    # -------------------------------------------------------------
    # STAGE 2: Multinomial Naive Bayes (Word TF-IDF, Float32)
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 2/3] Fitting Word TF-IDF for Multinomial Naive Bayes ---")
    word_vec_nb = build_word_vectorizer(
        ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
        min_df=word_cfg.get("min_df", 2),
        max_df=word_cfg.get("max_df", 0.95),
        max_features=word_max_feat,
        sublinear_tf=word_cfg.get("sublinear_tf", True),
        dtype=np.float32,
    )

    report["metadata"]["ram_profiling_timeline"]["nb_before_vectorize"] = log_memory_stage(
        "NB - Before Word Vectorize"
    )
    X_train_word_sp = word_vec_nb.fit_transform(X_train_raw)
    X_val_word_sp = word_vec_nb.transform(X_val_raw)
    report["metadata"]["ram_profiling_timeline"]["nb_after_vectorize"] = log_memory_stage(
        "NB - After Word Vectorize"
    )

    logger.info(
        f"Word TF-IDF vocabulary size: {X_train_word_sp.shape[1]:,} features (dtype={X_train_word_sp.dtype})"
    )

    nb_cfg = classical_cfg.get("multinomial_nb", {})
    alpha_candidates = nb_cfg.get("alpha_candidates", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0])

    logger.info(f"Tuning MultinomialNB across alpha candidates: {alpha_candidates}")
    best_nb, nb_tuning_history = tune_multinomial_nb(
        X_train=X_train_word_sp,
        y_train=y_train,
        X_val=X_val_word_sp,
        y_val=y_val,
        alpha_candidates=alpha_candidates,
        fit_prior=nb_cfg.get("fit_prior", True),
    )

    nb_preds = best_nb.predict(X_val_word_sp)
    nb_probs = best_nb.predict_proba(X_val_word_sp)
    nb_metrics = compute_classification_metrics(y_val, nb_preds, nb_probs)

    nb_pipeline = Pipeline([("features", word_vec_nb), ("clf", best_nb)])
    nb_path = out_dir / "multinomial_nb.joblib"
    nb_vec_path = out_dir / "multinomial_nb_vectorizer.joblib"
    save_model(nb_pipeline, str(nb_path))
    save_model(word_vec_nb, str(nb_vec_path))

    report["models"]["multinomial_nb"] = {
        "selected_hyperparameters": {"alpha": float(best_nb.alpha)},
        "tuning_history": nb_tuning_history,
        "validation_metrics": nb_metrics,
        "feature_count": int(X_train_word_sp.shape[1]),
        "artifact_path": str(nb_path),
        "vectorizer_path": str(nb_vec_path),
    }
    report["metadata"]["ram_profiling_timeline"]["after_nb_training"] = log_memory_stage(
        "After NB Training"
    )
    save_intermediate_report(report, metrics_path)
    logger.info(
        f"MultinomialNB (Best alpha={best_nb.alpha}) -> "
        f"Accuracy: {nb_metrics['accuracy']:.4f}, Macro-F1: {nb_metrics['macro_f1']:.4f}, "
        f"ROC-AUC: {nb_metrics.get('roc_auc', 0.0):.4f}"
    )

    # CRITICAL: Release Naive Bayes matrices, models and vectorizers to restore RAM
    logger.info("Releasing Naive Bayes matrices and executing garbage collection...")
    del X_train_word_sp, X_val_word_sp, best_nb, nb_pipeline, word_vec_nb
    gc.collect()
    report["metadata"]["ram_profiling_timeline"]["after_nb_cleanup"] = log_memory_stage(
        "After NB Cleanup"
    )

    # -------------------------------------------------------------
    # STAGE 3: Logistic Regression (Combined Word + Char, Float32)
    # -------------------------------------------------------------
    logger.info(
        "\n--- [STAGE 3/3] Building Isolated Word + Char TF-IDF Features for Logistic Regression ---"
    )
    report["metadata"]["ram_profiling_timeline"]["lr_before_features"] = log_memory_stage(
        "LR - Before Features"
    )

    # 3a. Fit Word Features (100k max, float32)
    logger.info(f"Fitting Word TF-IDF (max_features={word_max_feat:,}, float32)...")
    word_vec_lr = build_word_vectorizer(
        ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
        min_df=word_cfg.get("min_df", 2),
        max_df=word_cfg.get("max_df", 0.95),
        max_features=word_max_feat,
        sublinear_tf=word_cfg.get("sublinear_tf", True),
        dtype=np.float32,
    )
    X_train_w = word_vec_lr.fit_transform(X_train_raw)
    X_val_w = word_vec_lr.transform(X_val_raw)
    logger.info(f"Word features extracted: {X_train_w.shape[1]:,} columns.")

    # 3b. Fit Char Features (50k max, min_df=3, float32)
    logger.info(
        f"Fitting Character TF-IDF (max_features={char_max_feat:,}, min_df={char_min_df}, float32)..."
    )
    char_vec_lr = build_char_vectorizer(
        ngram_range=tuple(char_cfg.get("ngram_range", [3, 5])),
        min_df=char_min_df,
        max_df=char_cfg.get("max_df", 0.95),
        max_features=char_max_feat,
        sublinear_tf=char_cfg.get("sublinear_tf", True),
        dtype=np.float32,
    )
    X_train_c = char_vec_lr.fit_transform(X_train_raw)
    X_val_c = char_vec_lr.transform(X_val_raw)
    logger.info(f"Char features extracted: {X_train_c.shape[1]:,} columns.")

    # 3c. Combine sparse matrices without dense conversion
    logger.info("Combining Word + Char sparse matrices using scipy.sparse.hstack...")
    X_train_comb_sp = sparse.hstack([X_train_w, X_train_c], format="csr")
    X_val_comb_sp = sparse.hstack([X_val_w, X_val_c], format="csr")

    # Free individual components immediately
    del X_train_w, X_train_c, X_val_w, X_val_c
    gc.collect()

    report["metadata"]["ram_profiling_timeline"]["lr_after_features_hstack"] = log_memory_stage(
        "LR - After Feature hstack"
    )
    logger.info(
        f"Total Combined Features: {X_train_comb_sp.shape[1]:,} columns "
        f"(dtype={X_train_comb_sp.dtype}, sparsity={1.0 - (X_train_comb_sp.nnz / (X_train_comb_sp.shape[0] * X_train_comb_sp.shape[1])):.4%})"
    )

    combined_features = FeatureUnion(
        [
            ("word_tfidf", word_vec_lr),
            ("char_tfidf", char_vec_lr),
        ]
    )

    # 3d. Tune Logistic Regression with solver benchmarking
    lr_cfg = classical_cfg.get("logistic_regression", {})
    c_candidates = lr_cfg.get("C_candidates", [1.0, 2.0, 4.0])
    solver_candidates = lr_cfg.get("solver_candidates", ["lbfgs", "liblinear"])

    logger.info(
        f"Benchmarking Logistic Regression across Solvers {solver_candidates} and C {c_candidates}..."
    )
    best_lr, best_lr_c, best_lr_solver, lr_tuning_history = tune_logistic_regression(
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

    lr_preds = best_lr.predict(X_val_comb_sp)
    lr_probs = best_lr.predict_proba(X_val_comb_sp)
    lr_metrics = compute_classification_metrics(y_val, lr_preds, lr_probs)

    lr_pipeline = Pipeline([("features", combined_features), ("clf", best_lr)])
    lr_path = out_dir / "logistic_regression.joblib"
    lr_feat_path = out_dir / "logistic_regression_features.joblib"
    save_model(lr_pipeline, str(lr_path))
    save_model(combined_features, str(lr_feat_path))

    report["models"]["logistic_regression"] = {
        "selected_hyperparameters": {
            "C": float(best_lr_c),
            "solver": best_lr_solver,
        },
        "tuning_history": lr_tuning_history,
        "validation_metrics": lr_metrics,
        "feature_count": int(X_train_comb_sp.shape[1]),
        "artifact_path": str(lr_path),
        "features_path": str(lr_feat_path),
    }
    report["metadata"]["ram_profiling_timeline"]["after_lr_training"] = log_memory_stage(
        "After LR Training"
    )
    save_intermediate_report(report, metrics_path)
    logger.info(
        f"Logistic Regression (Best C={best_lr_c}, Solver={best_lr_solver}) -> "
        f"Accuracy: {lr_metrics['accuracy']:.4f}, Macro-F1: {lr_metrics['macro_f1']:.4f}, "
        f"ROC-AUC: {lr_metrics.get('roc_auc', 0.0):.4f}"
    )

    # -------------------------------------------------------------
    # STAGE 4: Optional Linear SVM (Only if explicitly requested)
    # -------------------------------------------------------------
    if args.include_svm:
        logger.info("\n--- [STAGE 4/4] Tuning Optional Calibrated Linear SVM ---")
        svm_cfg = classical_cfg.get("linear_svm", {})
        svm_c_candidates = svm_cfg.get("C_candidates", [0.5, 1.0, 2.0])

        best_svm, best_svm_c, svm_tuning_history = tune_linear_svm(
            X_train=X_train_comb_sp,
            y_train=y_train,
            X_val=X_val_comb_sp,
            y_val=y_val,
            C_candidates=svm_c_candidates,
            max_iter=svm_cfg.get("max_iter", 2000),
            class_weight=svm_cfg.get("class_weight", "balanced"),
            random_state=seed,
        )

        svm_preds = best_svm.predict(X_val_comb_sp)
        svm_probs = best_svm.predict_proba(X_val_comb_sp)
        svm_metrics = compute_classification_metrics(y_val, svm_preds, svm_probs)

        svm_pipeline = Pipeline([("features", combined_features), ("clf", best_svm)])
        svm_path = out_dir / "linear_svm.joblib"
        save_model(svm_pipeline, str(svm_path))

        report["models"]["linear_svm"] = {
            "selected_hyperparameters": {"C": float(best_svm_c)},
            "tuning_history": svm_tuning_history,
            "validation_metrics": svm_metrics,
            "artifact_path": str(svm_path),
        }
        report["metadata"]["ram_profiling_timeline"]["after_svm_training"] = log_memory_stage(
            "After SVM Training"
        )
        save_intermediate_report(report, metrics_path)
        logger.info(
            f"Linear SVM -> Accuracy: {svm_metrics['accuracy']:.4f}, Macro-F1: {svm_metrics['macro_f1']:.4f}"
        )

    # Final cleanup of combined matrices
    del X_train_comb_sp, X_val_comb_sp, best_lr, lr_pipeline
    gc.collect()

    report["metadata"]["ram_profiling_timeline"]["final_completed"] = log_memory_stage(
        "Final Completed"
    )
    save_intermediate_report(report, metrics_path)

    # Summary Table Display
    logger.info("\n==================== VALIDATION BENCHMARK SUMMARY ====================")
    logger.info(
        f"{'Model':<25} | {'Macro-F1':<10} | {'Accuracy':<10} | {'Fake-F1':<10} | {'ROC-AUC':<10} | {'Brier':<10}"
    )
    logger.info("-" * 85)
    for m_name, m_data in report["models"].items():
        v = m_data["validation_metrics"]
        roc_str = f"{v.get('roc_auc', 0.0):.4f}" if v.get("roc_auc") is not None else "N/A"
        brier_str = (
            f"{v.get('brier_score', 0.0):.4f}" if v.get("brier_score") is not None else "N/A"
        )
        logger.info(
            f"{m_name:<25} | {v['macro_f1']:<10.4f} | {v['accuracy']:<10.4f} | "
            f"{v.get('fake_f1', 0.0):<10.4f} | {roc_str:<10} | {brier_str:<10}"
        )
    logger.info("======================================================================")
    logger.info(f"\nAll validation artifacts & metrics safely preserved at: {metrics_path}")


if __name__ == "__main__":
    main()
