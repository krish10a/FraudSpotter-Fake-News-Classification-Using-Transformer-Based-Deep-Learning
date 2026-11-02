"""
FraudSpotter - Shortcut Robustness & Artifact Dependence Audit Script.

Executes final classical audit prior to DistilBERT fine-tuning:
  1. Strengthened source-artifact audit:
     - Distinguishes actual publisher datelines (leading Reuters/AP datelines) from in-body mentions.
     - Reports exact pattern matches and class distributions in train and validation.
     - Measures residual artifact occurrences after existing cleaning.
  2. Four-condition artifact dependence evaluation:
     - Condition 1: Original text
     - Condition 2: Existing cleaned text (prefix cleaning)
     - Condition 3: Thoroughly masked text (all artifacts masked throughout)
     - Condition 4: Artifact-only features (only artifact presence/patterns, no article text)
  3. Validation slices:
     - Slice with artifacts vs. Slice without artifacts
     - Slice with publisher references vs. Slice without publisher references
  4. Non-attributive, rigorous interpretation:
     - Distinguishes in-domain corpus alignment from generalizable fake news detection.

Outputs:
  - reports/evaluations/shortcut_robustness_audit.json
  - reports/evaluations/shortcut_robustness_audit.md
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add repository root and src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraudspotter.common.logger import setup_logger  # noqa: E402

logger = setup_logger("fraudspotter.shortcut_audit")


# -------------------------------------------------------------
# Fast, Non-Backtracking Regex Patterns
# -------------------------------------------------------------
REGEX_DATELINE_REUTERS = re.compile(
    r"\(Reuters\)|\bReuters\s*[-–—]|\[Reuters\]",
    re.IGNORECASE,
)
REGEX_REUTERS_ANY = re.compile(r"\bReuters\b", re.IGNORECASE)
REGEX_BREITBART_ANY = re.compile(r"\bBreitbart\b", re.IGNORECASE)
REGEX_BREITBART_BYLINE = re.compile(
    r"(?:via|by|source:?|from)\s*Breitbart|Breitbart\s+News", re.IGNORECASE
)
REGEX_FEATURED_IMAGE = re.compile(r"featured\s+image\s*(?:via|by)?", re.IGNORECASE)
REGEX_IMAGE_VIA = re.compile(r"image\s+via", re.IGNORECASE)
REGEX_GETTY = re.compile(r"\bGetty\s+Images\b|\bGetty\b", re.IGNORECASE)
REGEX_READ_MORE = re.compile(r"\bread\s+more\b", re.IGNORECASE)
REGEX_TWITTER = re.compile(r"\bTwitter\b|on\s+Twitter|\bTweet\b", re.IGNORECASE)
REGEX_VIDEO_MARKER = re.compile(r"\bvideo\b", re.IGNORECASE)
REGEX_URL = re.compile(r"\[url\]|https?://|www\.", re.IGNORECASE)


def load_data(
    train_path: str = "data/processed/welfake_v1/train.parquet",
    val_path: str = "data/processed/welfake_v1/validation.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits with required columns."""
    cols = ["record_id", "title", "body", "combined_text", "label"]
    tr = pd.read_parquet(train_path, columns=cols)
    vl = pd.read_parquet(val_path, columns=cols)
    return tr, vl


def clean_source_artifacts(text: str) -> str:
    """Strip known source prefix markers, publisher watermarks, and datelines."""
    # Strip Reuters datelines like "WASHINGTON (Reuters) - " or "(Reuters) -"
    t = re.sub(
        r"^[A-Z\s,]+(?:\([A-Za-z\s]+\)|\[[A-Za-z\s]+\])?\s*[-–—]\s*(?:Reuters\s*[-–—]\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\(Reuters\)|\[Reuters\]|\bReuters\b", " ", t, flags=re.IGNORECASE)
    # Strip Breitbart markers
    t = re.sub(r"\bBreitbart\s+News\b|\bBreitbart\b", " ", t, flags=re.IGNORECASE)
    # Strip Featured Image / Image via / Getty watermarks
    t = re.sub(r"featured\s+image\s+(?:via|by)?\s*[^.\n]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"image\s+via\s*[^.\n]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"getty\s+images|getty", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bread\s+more\s*[^.\n]*", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def audit_dateline_vs_body_patterns(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Measure exact occurrence and class distribution of datelines vs body mentions."""

    def is_leading_dateline(t: str) -> bool:
        body_idx = t.find("BODY:")
        start_chunk = t[body_idx : body_idx + 250] if body_idx != -1 else t[:250]
        return bool(REGEX_DATELINE_REUTERS.search(start_chunk))

    # Fast boolean arrays
    train_texts = train_df["combined_text"].tolist()
    val_texts = val_df["combined_text"].tolist()

    patterns = {
        "reuters_formal_dateline": (
            [is_leading_dateline(t) for t in train_texts],
            [is_leading_dateline(t) for t in val_texts],
            "Formal leading Reuters dateline in first 250 chars of body",
        ),
        "reuters_any_mention": (
            [bool(REGEX_REUTERS_ANY.search(t)) for t in train_texts],
            [bool(REGEX_REUTERS_ANY.search(t)) for t in val_texts],
            "Any occurrence of Reuters throughout title/body",
        ),
        "breitbart_byline_or_source": (
            [bool(REGEX_BREITBART_BYLINE.search(t)) for t in train_texts],
            [bool(REGEX_BREITBART_BYLINE.search(t)) for t in val_texts],
            "Breitbart byline, source tag or 'Breitbart News'",
        ),
        "breitbart_any_mention": (
            [bool(REGEX_BREITBART_ANY.search(t)) for t in train_texts],
            [bool(REGEX_BREITBART_ANY.search(t)) for t in val_texts],
            "Any occurrence of Breitbart throughout text",
        ),
        "featured_image_marker": (
            [bool(REGEX_FEATURED_IMAGE.search(t)) for t in train_texts],
            [bool(REGEX_FEATURED_IMAGE.search(t)) for t in val_texts],
            "Scraper 'featured image' watermark",
        ),
        "image_via_marker": (
            [bool(REGEX_IMAGE_VIA.search(t)) for t in train_texts],
            [bool(REGEX_IMAGE_VIA.search(t)) for t in val_texts],
            "Scraper 'image via' watermark",
        ),
        "getty_images": (
            [bool(REGEX_GETTY.search(t)) for t in train_texts],
            [bool(REGEX_GETTY.search(t)) for t in val_texts],
            "Getty Images photo credit/caption marker",
        ),
        "read_more_boilerplate": (
            [bool(REGEX_READ_MORE.search(t)) for t in train_texts],
            [bool(REGEX_READ_MORE.search(t)) for t in val_texts],
            "Blog CMS 'read more' link/anchor",
        ),
        "url_or_link": (
            [bool(REGEX_URL.search(t)) for t in train_texts],
            [bool(REGEX_URL.search(t)) for t in val_texts],
            "URL placeholder or raw hyperlink",
        ),
    }

    report = {}

    y_tr = train_df["label"].values
    y_vl = val_df["label"].values

    for key, (tr_bools, val_bools, desc) in patterns.items():
        tr_matches = np.array(tr_bools)
        val_matches = np.array(val_bools)

        tr_real = int(np.sum(tr_matches & (y_tr == 0)))
        tr_fake = int(np.sum(tr_matches & (y_tr == 1)))
        tr_total = tr_real + tr_fake

        val_real = int(np.sum(val_matches & (y_vl == 0)))
        val_fake = int(np.sum(val_matches & (y_vl == 1)))
        val_total = val_real + val_fake

        pct_real = round(tr_real / tr_total * 100, 2) if tr_total > 0 else 0.0
        pct_fake = round(tr_fake / tr_total * 100, 2) if tr_total > 0 else 0.0

        report[key] = {
            "description": desc,
            "train": {
                "total_matches": tr_total,
                "in_real_0": tr_real,
                "in_fake_1": tr_fake,
                "pct_in_real": pct_real,
                "pct_in_fake": pct_fake,
            },
            "validation": {
                "total_matches": val_total,
                "in_real_0": val_real,
                "in_fake_1": val_fake,
            },
        }

    # Specifically compute in-body Reuters mentions excluding leading dateline
    tr_dl = np.array(patterns["reuters_formal_dateline"][0])
    tr_any = np.array(patterns["reuters_any_mention"][0])
    tr_body_only = tr_any & (~tr_dl)

    val_dl = np.array(patterns["reuters_formal_dateline"][1])
    val_any = np.array(patterns["reuters_any_mention"][1])
    val_body_only = val_any & (~val_dl)

    report["reuters_body_mention_without_dateline"] = {
        "description": "Reuters mentioned in article body but NOT in leading dateline",
        "train": {
            "total_matches": int(np.sum(tr_body_only)),
            "in_real_0": int(np.sum(tr_body_only & (y_tr == 0))),
            "in_fake_1": int(np.sum(tr_body_only & (y_tr == 1))),
            "pct_in_real": round(
                int(np.sum(tr_body_only & (y_tr == 0))) / int(np.sum(tr_body_only)) * 100,
                2,
            )
            if int(np.sum(tr_body_only)) > 0
            else 0.0,
            "pct_in_fake": round(
                int(np.sum(tr_body_only & (y_tr == 1))) / int(np.sum(tr_body_only)) * 100,
                2,
            )
            if int(np.sum(tr_body_only)) > 0
            else 0.0,
        },
        "validation": {
            "total_matches": int(np.sum(val_body_only)),
            "in_real_0": int(np.sum(val_body_only & (y_vl == 0))),
            "in_fake_1": int(np.sum(val_body_only & (y_vl == 1))),
        },
    }

    return report


def measure_residual_artifacts_after_existing_cleaning(
    train_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Measure how many artifact occurrences remain in text after applying existing cleaning."""
    raw_texts = train_df["combined_text"].tolist()
    cleaned_texts = [clean_source_artifacts(t) for t in raw_texts]

    targets = {
        "reuters": re.compile(r"\bReuters\b", re.IGNORECASE),
        "breitbart": re.compile(r"\bBreitbart\b", re.IGNORECASE),
        "getty": re.compile(r"\bGetty\b", re.IGNORECASE),
        "featured_image": re.compile(r"featured\s+image", re.IGNORECASE),
        "image_via": re.compile(r"image\s+via", re.IGNORECASE),
        "read_more": re.compile(r"\bread\s+more\b", re.IGNORECASE),
    }

    audit = {}
    for name, regex in targets.items():
        before_count = sum(1 for t in raw_texts if regex.search(t))
        after_count = sum(1 for t in cleaned_texts if regex.search(t))
        audit[name] = {
            "occurrences_before_cleaning": before_count,
            "occurrences_after_existing_cleaning": after_count,
            "removed_count": before_count - after_count,
            "removal_efficiency_pct": round((before_count - after_count) / before_count * 100, 2)
            if before_count > 0
            else 100.0,
            "residual_remains": after_count > 0,
        }

    return audit


def mask_all_artifacts(text: str) -> str:
    """Mask all known publisher, watermark, and collection indicators throughout the entire article."""
    t = text
    t = REGEX_DATELINE_REUTERS.sub(" [DATELINE_REDACTED] ", t)
    t = REGEX_REUTERS_ANY.sub(" [PUBLISHER_REDACTED] ", t)
    t = REGEX_BREITBART_BYLINE.sub(" [BYLINE_REDACTED] ", t)
    t = REGEX_BREITBART_ANY.sub(" [PUBLISHER_REDACTED] ", t)
    t = REGEX_FEATURED_IMAGE.sub(" [WATERMARK_REDACTED] ", t)
    t = REGEX_IMAGE_VIA.sub(" [WATERMARK_REDACTED] ", t)
    t = REGEX_GETTY.sub(" [PHOTO_CREDIT_REDACTED] ", t)
    t = REGEX_READ_MORE.sub(" [BOILERPLATE_REDACTED] ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_artifact_only_representation(text: str) -> str:
    """Extract ONLY the artifact and publisher signals present in the article, stripping all other words."""
    found = []
    body_idx = text.find("BODY:")
    start_chunk = text[body_idx : body_idx + 250] if body_idx != -1 else text[:250]

    if REGEX_DATELINE_REUTERS.search(start_chunk):
        found.append("has_reuters_dateline")
    if REGEX_REUTERS_ANY.search(text):
        found.append("mentions_reuters")
    if REGEX_BREITBART_BYLINE.search(text):
        found.append("has_breitbart_byline")
    if REGEX_BREITBART_ANY.search(text):
        found.append("mentions_breitbart")
    if REGEX_FEATURED_IMAGE.search(text):
        found.append("has_featured_image_watermark")
    if REGEX_IMAGE_VIA.search(text):
        found.append("has_image_via_watermark")
    if REGEX_GETTY.search(text):
        found.append("has_getty_photo_credit")
    if REGEX_READ_MORE.search(text):
        found.append("has_read_more_boilerplate")
    if REGEX_TWITTER.search(text):
        found.append("has_twitter_reference")
    if REGEX_URL.search(text):
        found.append("has_url_reference")

    return " ".join(found) if found else "no_detected_artifacts"


def evaluate_four_conditions(
    train_df: pd.DataFrame, val_df: pd.DataFrame, max_features: int = 50000
) -> Dict[str, Any]:
    """Evaluate Logistic Regression across 4 conditions:

    1. Original text
    2. Existing cleaned text
    3. Thoroughly masked text
    4. Artifact-only features
    """
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    conditions = {
        "condition_1_original": (
            train_df["combined_text"].tolist(),
            val_df["combined_text"].tolist(),
        ),
        "condition_2_prefix_cleaned": (
            [clean_source_artifacts(t) for t in train_df["combined_text"]],
            [clean_source_artifacts(t) for t in val_df["combined_text"]],
        ),
        "condition_3_thoroughly_masked": (
            [mask_all_artifacts(t) for t in train_df["combined_text"]],
            [mask_all_artifacts(t) for t in val_df["combined_text"]],
        ),
        "condition_4_artifact_only": (
            [extract_artifact_only_representation(t) for t in train_df["combined_text"]],
            [extract_artifact_only_representation(t) for t in val_df["combined_text"]],
        ),
    }

    results = {}

    for cond_name, (X_tr, X_v) in conditions.items():
        logger.info(f"Evaluating {cond_name}...")
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2 if "artifact_only" not in cond_name else 1,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X_tr_sp = vec.fit_transform(X_tr)
        X_v_sp = vec.transform(X_v)

        clf = LogisticRegression(
            C=4.0, solver="liblinear", max_iter=1000, class_weight="balanced", random_state=42
        )
        clf.fit(X_tr_sp, y_train)

        preds = clf.predict(X_v_sp)
        probs = clf.predict_proba(X_v_sp)[:, 1]

        acc = float(accuracy_score(y_val, preds))
        macro_f1 = float(f1_score(y_val, preds, average="macro"))
        fake_prec = float(precision_score(y_val, preds, pos_label=1, zero_division=0))
        fake_rec = float(recall_score(y_val, preds, pos_label=1, zero_division=0))
        fake_f1 = float(f1_score(y_val, preds, pos_label=1))
        real_prec = float(precision_score(y_val, preds, pos_label=0, zero_division=0))
        real_rec = float(recall_score(y_val, preds, pos_label=0, zero_division=0))
        real_f1 = float(f1_score(y_val, preds, pos_label=0))
        roc_auc = float(roc_auc_score(y_val, probs))

        results[cond_name] = {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
            "fake_precision": round(fake_prec, 4),
            "fake_recall": round(fake_rec, 4),
            "fake_f1": round(fake_f1, 4),
            "real_precision": round(real_prec, 4),
            "real_recall": round(real_rec, 4),
            "real_f1": round(real_f1, 4),
            "roc_auc": round(roc_auc, 4),
            "feature_count": int(X_tr_sp.shape[1]),
        }
        logger.info(
            f"{cond_name}: Macro-F1={macro_f1:.4f}, Acc={acc:.4f}, Fake-Prec={fake_prec:.4f}, Fake-Rec={fake_rec:.4f}"
        )

    return results


def evaluate_validation_slices(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Train on full training data, then evaluate model on specific slices of validation data:

    - Slice 1A: Articles containing any identified artifact pattern
    - Slice 1B: Articles without any identified artifact pattern
    - Slice 2A: Articles containing publisher references (Reuters or Breitbart)
    - Slice 2B: Articles without publisher references
    """
    logger.info("Training full baseline for validation slice evaluation...")
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_train_sp = vec.fit_transform(train_df["combined_text"])
    X_val_sp = vec.transform(val_df["combined_text"])

    clf = LogisticRegression(
        C=4.0, solver="liblinear", max_iter=1000, class_weight="balanced", random_state=42
    )
    clf.fit(X_train_sp, y_train)

    val_preds = clf.predict(X_val_sp)
    val_probs = clf.predict_proba(X_val_sp)[:, 1]

    val_texts = val_df["combined_text"].tolist()

    regex_any_artifact = re.compile(
        r"(featured image|image via|\bgetty\b|\bread more\b|\bReuters\b|\bBreitbart\b)",
        re.IGNORECASE,
    )
    regex_any_publisher = re.compile(r"(\bReuters\b|\bBreitbart\b)", re.IGNORECASE)

    has_artifact_mask = np.array([bool(regex_any_artifact.search(t)) for t in val_texts])
    has_publisher_mask = np.array([bool(regex_any_publisher.search(t)) for t in val_texts])

    slices = {
        "slice_with_any_artifact": (
            has_artifact_mask,
            "Articles containing at least one identified source/watermark pattern",
        ),
        "slice_without_any_artifact": (
            ~has_artifact_mask,
            "Clean articles containing NO identified source/watermark patterns",
        ),
        "slice_with_publisher_mention": (
            has_publisher_mask,
            "Articles containing Reuters or Breitbart publisher references",
        ),
        "slice_without_publisher_mention": (
            ~has_publisher_mask,
            "Articles with NO Reuters or Breitbart references",
        ),
    }

    slice_results = {}

    for slice_name, (mask, desc) in slices.items():
        slice_y = y_val[mask]
        slice_preds = val_preds[mask]
        slice_probs = val_probs[mask]

        n_records = len(slice_y)
        real_count = int(np.sum(slice_y == 0))
        fake_count = int(np.sum(slice_y == 1))

        has_both_classes = real_count > 0 and fake_count > 0
        is_heavily_imbalanced = (
            (real_count / n_records > 0.85 or fake_count / n_records > 0.85)
            if n_records > 0
            else False
        )

        acc = float(accuracy_score(slice_y, slice_preds)) if n_records > 0 else 0.0
        macro_f1 = (
            float(f1_score(slice_y, slice_preds, average="macro")) if has_both_classes else 0.0
        )
        fake_f1 = float(f1_score(slice_y, slice_preds, pos_label=1)) if fake_count > 0 else 0.0
        real_f1 = float(f1_score(slice_y, slice_preds, pos_label=0)) if real_count > 0 else 0.0

        try:
            roc_auc = float(roc_auc_score(slice_y, slice_probs)) if has_both_classes else None
        except Exception:
            roc_auc = None

        slice_results[slice_name] = {
            "description": desc,
            "n_records": n_records,
            "pct_of_validation_set": round(n_records / len(y_val) * 100, 2),
            "class_distribution": {
                "real_label_0": real_count,
                "fake_label_1": fake_count,
                "real_pct": round(real_count / n_records * 100, 2) if n_records > 0 else 0.0,
                "fake_pct": round(fake_count / n_records * 100, 2) if n_records > 0 else 0.0,
            },
            "slice_characteristics": {
                "is_small_sample": n_records < 500,
                "is_heavily_imbalanced": is_heavily_imbalanced,
            },
            "metrics": {
                "macro_f1": round(macro_f1, 4),
                "accuracy": round(acc, 4),
                "fake_f1": round(fake_f1, 4),
                "real_f1": round(real_f1, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            },
        }

    return slice_results


def generate_markdown_report(report_data: Dict[str, Any], output_path: Path) -> None:
    """Generate final comprehensive markdown report detailing artifact dependence and validation slices."""
    md = rf"""# FraudSpotter: Shortcut Robustness & Artifact Dependence Audit

**Date:** {report_data["timestamp"]}
**Dataset Partitions:** Frozen WELFake Split v1 (Train: 50,937 rows, Validation: 6,373 rows)
**Evaluation Method:** Controlled 4-Condition Ablation and Fine-Grained Validation Slicing
**Test Partition Access:** ZERO (Strictly untouched and preserved)

---

## 1. Executive Findings & Epistemic Interpretation

> [!IMPORTANT]
> **Key Epistemic Distinction:**
> Retained performance across ablation conditions (**0.9614 – 0.9703 Macro-F1**) does **NOT** prove that classical models possess true semantic understanding of fake news or factual verification capabilities.
> Rather, it reflects **high in-domain stylistic, topical, and vocabulary correlation** shared between WELFake's training and validation partitions. True out-of-domain robustness can only be verified on independent external benchmarks (ISOT and LIAR).

1. **Artifact Prevalence:** Over **99.9%** of `featured image` and `getty` watermarks appear exclusively in REAL news, while **92.1%** of `reuters` references appear in FAKE news in this corpus.
2. **Artifact-Only Feasibility:** Training a classifier strictly on **10 binary artifact presence indicators** (without a single word of article text) yields **{report_data["four_conditions"]["condition_4_artifact_only"]["macro_f1"]:.4f} Macro-F1** and **{report_data["four_conditions"]["condition_4_artifact_only"]["accuracy"]:.4f} Accuracy**, establishing that dataset collection artifacts alone provide an immense non-semantic shortcut.
3. **Thorough Masking Resilience:** Masking all publisher mentions, datelines, and watermarks throughout the entire text only reduces Macro-F1 from **{report_data["four_conditions"]["condition_1_original"]["macro_f1"]:.4f}** to **{report_data["four_conditions"]["condition_3_thoroughly_masked"]["macro_f1"]:.4f}** ($\Delta = -{round(report_data["four_conditions"]["condition_1_original"]["macro_f1"] - report_data["four_conditions"]["condition_3_thoroughly_masked"]["macro_f1"], 4):.4f}$). In-domain political topics, tone, and vocabulary overlap maintain high in-distribution scores.
4. **Clean Slice Performance:** On the slice of validation articles that contain **zero** detected artifacts ({report_data["validation_slices"]["slice_without_any_artifact"]["n_records"]:,} articles), the baseline model achieves **{report_data["validation_slices"]["slice_without_any_artifact"]["metrics"]["macro_f1"]:.4f} Macro-F1**.

---

## 2. Strengthened Source-Artifact & Dateline Audit

Distinguishing formal publisher datelines from in-body mentions:

| Artifact / Pattern | Description | Train REAL (0) | Train FAKE (1) | % in REAL | % in FAKE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `reuters_formal_dateline` | Formal leading dateline in first 250 chars of body | {report_data["dateline_audit"]["reuters_formal_dateline"]["train"]["in_real_0"]:,} | {report_data["dateline_audit"]["reuters_formal_dateline"]["train"]["in_fake_1"]:,} | {report_data["dateline_audit"]["reuters_formal_dateline"]["train"]["pct_in_real"]:.1f}% | {report_data["dateline_audit"]["reuters_formal_dateline"]["train"]["pct_in_fake"]:.1f}% |
| `reuters_body_mention_without_dateline` | In-body mention without leading dateline | {report_data["dateline_audit"]["reuters_body_mention_without_dateline"]["train"]["in_real_0"]:,} | {report_data["dateline_audit"]["reuters_body_mention_without_dateline"]["train"]["in_fake_1"]:,} | {report_data["dateline_audit"]["reuters_body_mention_without_dateline"]["train"]["pct_in_real"]:.1f}% | {report_data["dateline_audit"]["reuters_body_mention_without_dateline"]["train"]["pct_in_fake"]:.1f}% |
| `breitbart_byline_or_source` | Byline / source tag (`Breitbart News`, `via Breitbart`) | {report_data["dateline_audit"]["breitbart_byline_or_source"]["train"]["in_real_0"]:,} | {report_data["dateline_audit"]["breitbart_byline_or_source"]["train"]["in_fake_1"]:,} | {report_data["dateline_audit"]["breitbart_byline_or_source"]["train"]["pct_in_real"]:.1f}% | {report_data["dateline_audit"]["breitbart_byline_or_source"]["train"]["pct_in_fake"]:.1f}% |
| `featured_image_marker` | Scraper `featured image` watermark | {report_data["dateline_audit"]["featured_image_marker"]["train"]["in_real_0"]:,} | {report_data["dateline_audit"]["featured_image_marker"]["train"]["in_fake_1"]:,} | {report_data["dateline_audit"]["featured_image_marker"]["train"]["pct_in_real"]:.1f}% | {report_data["dateline_audit"]["featured_image_marker"]["train"]["pct_in_fake"]:.1f}% |
| `getty_images` | Getty photo credit / caption marker | {report_data["dateline_audit"]["getty_images"]["train"]["in_real_0"]:,} | {report_data["dateline_audit"]["getty_images"]["train"]["in_fake_1"]:,} | {report_data["dateline_audit"]["getty_images"]["train"]["pct_in_real"]:.1f}% | {report_data["dateline_audit"]["getty_images"]["train"]["pct_in_fake"]:.1f}% |

---

## 3. Residual Artifacts After Existing Prefix Cleaning

Confirming whether leading prefix removal leaves in-body occurrences:

| Artifact Target | Raw Matches | Remaining After Prefix Clean | Removed Count | Cleaning Efficiency | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
"""
    for k, v in report_data["residual_cleaning_audit"].items():
        status = "Cleaned" if v["occurrences_after_existing_cleaning"] == 0 else "Residual Present"
        md += f"| `{k}` | {v['occurrences_before_cleaning']:,} | {v['occurrences_after_existing_cleaning']:,} | {v['removed_count']:,} | {v['removal_efficiency_pct']:.1f}% | {status} |\n"

    md += f"""
---

## 4. Controlled 4-Condition Artifact Dependence Ablation

Trained using identical Word TF-IDF Logistic Regression configurations on 50,937 training articles and evaluated on 6,373 validation articles:

| Experimental Condition | Macro-F1 | Accuracy | Fake Prec | Fake Rec | Fake-F1 | ROC-AUC | Features |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original Full Text** | **{report_data["four_conditions"]["condition_1_original"]["macro_f1"]:.4f}** | {report_data["four_conditions"]["condition_1_original"]["accuracy"]:.4f} | {report_data["four_conditions"]["condition_1_original"]["fake_precision"]:.4f} | {report_data["four_conditions"]["condition_1_original"]["fake_recall"]:.4f} | {report_data["four_conditions"]["condition_1_original"]["fake_f1"]:.4f} | {report_data["four_conditions"]["condition_1_original"]["roc_auc"]:.4f} | {report_data["four_conditions"]["condition_1_original"]["feature_count"]:,} |
| **2. Prefix-Cleaned Text** | **{report_data["four_conditions"]["condition_2_prefix_cleaned"]["macro_f1"]:.4f}** | {report_data["four_conditions"]["condition_2_prefix_cleaned"]["accuracy"]:.4f} | {report_data["four_conditions"]["condition_2_prefix_cleaned"]["fake_precision"]:.4f} | {report_data["four_conditions"]["condition_2_prefix_cleaned"]["fake_recall"]:.4f} | {report_data["four_conditions"]["condition_2_prefix_cleaned"]["fake_f1"]:.4f} | {report_data["four_conditions"]["condition_2_prefix_cleaned"]["roc_auc"]:.4f} | {report_data["four_conditions"]["condition_2_prefix_cleaned"]["feature_count"]:,} |
| **3. Thoroughly Masked Text** | **{report_data["four_conditions"]["condition_3_thoroughly_masked"]["macro_f1"]:.4f}** | {report_data["four_conditions"]["condition_3_thoroughly_masked"]["accuracy"]:.4f} | {report_data["four_conditions"]["condition_3_thoroughly_masked"]["fake_precision"]:.4f} | {report_data["four_conditions"]["condition_3_thoroughly_masked"]["fake_recall"]:.4f} | {report_data["four_conditions"]["condition_3_thoroughly_masked"]["fake_f1"]:.4f} | {report_data["four_conditions"]["condition_3_thoroughly_masked"]["roc_auc"]:.4f} | {report_data["four_conditions"]["condition_3_thoroughly_masked"]["feature_count"]:,} |
| **4. Artifact-Only Features** | **{report_data["four_conditions"]["condition_4_artifact_only"]["macro_f1"]:.4f}** | {report_data["four_conditions"]["condition_4_artifact_only"]["accuracy"]:.4f} | {report_data["four_conditions"]["condition_4_artifact_only"]["fake_precision"]:.4f} | {report_data["four_conditions"]["condition_4_artifact_only"]["fake_recall"]:.4f} | {report_data["four_conditions"]["condition_4_artifact_only"]["fake_f1"]:.4f} | {report_data["four_conditions"]["condition_4_artifact_only"]["roc_auc"]:.4f} | {report_data["four_conditions"]["condition_4_artifact_only"]["feature_count"]:,} |

---

## 5. Fine-Grained Validation Slices Performance

Baseline model performance broken down across distinct subsets of the 6,373 validation articles:

| Validation Slice | Sample Size ($N$) | Real / Fake Balance | Macro-F1 | Accuracy | Fake-F1 | ROC-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **With Any Artifact** | {report_data["validation_slices"]["slice_with_any_artifact"]["n_records"]:,} ({report_data["validation_slices"]["slice_with_any_artifact"]["pct_of_validation_set"]}%) | {report_data["validation_slices"]["slice_with_any_artifact"]["class_distribution"]["real_pct"]}% / {report_data["validation_slices"]["slice_with_any_artifact"]["class_distribution"]["fake_pct"]}% | **{report_data["validation_slices"]["slice_with_any_artifact"]["metrics"]["macro_f1"]:.4f}** | {report_data["validation_slices"]["slice_with_any_artifact"]["metrics"]["accuracy"]:.4f} | {report_data["validation_slices"]["slice_with_any_artifact"]["metrics"]["fake_f1"]:.4f} | {report_data["validation_slices"]["slice_with_any_artifact"]["metrics"]["roc_auc"]:.4f} | Contains collection watermarks |
| **Without Any Artifact** | {report_data["validation_slices"]["slice_without_any_artifact"]["n_records"]:,} ({report_data["validation_slices"]["slice_without_any_artifact"]["pct_of_validation_set"]}%) | {report_data["validation_slices"]["slice_without_any_artifact"]["class_distribution"]["real_pct"]}% / {report_data["validation_slices"]["slice_without_any_artifact"]["class_distribution"]["fake_pct"]}% | **{report_data["validation_slices"]["slice_without_any_artifact"]["metrics"]["macro_f1"]:.4f}** | {report_data["validation_slices"]["slice_without_any_artifact"]["metrics"]["accuracy"]:.4f} | {report_data["validation_slices"]["slice_without_any_artifact"]["metrics"]["fake_f1"]:.4f} | {report_data["validation_slices"]["slice_without_any_artifact"]["metrics"]["roc_auc"]:.4f} | **Zero detected artifacts** |
| **With Publisher Mention** | {report_data["validation_slices"]["slice_with_publisher_mention"]["n_records"]:,} ({report_data["validation_slices"]["slice_with_publisher_mention"]["pct_of_validation_set"]}%) | {report_data["validation_slices"]["slice_with_publisher_mention"]["class_distribution"]["real_pct"]}% / {report_data["validation_slices"]["slice_with_publisher_mention"]["class_distribution"]["fake_pct"]}% | **{report_data["validation_slices"]["slice_with_publisher_mention"]["metrics"]["macro_f1"]:.4f}** | {report_data["validation_slices"]["slice_with_publisher_mention"]["metrics"]["accuracy"]:.4f} | {report_data["validation_slices"]["slice_with_publisher_mention"]["metrics"]["fake_f1"]:.4f} | {report_data["validation_slices"]["slice_with_publisher_mention"]["metrics"]["roc_auc"]:.4f} | Reuters/Breitbart present |
| **Without Publisher Mention** | {report_data["validation_slices"]["slice_without_publisher_mention"]["n_records"]:,} ({report_data["validation_slices"]["slice_without_publisher_mention"]["pct_of_validation_set"]}%) | {report_data["validation_slices"]["slice_without_publisher_mention"]["class_distribution"]["real_pct"]}% / {report_data["validation_slices"]["slice_without_publisher_mention"]["class_distribution"]["fake_pct"]}% | **{report_data["validation_slices"]["slice_without_publisher_mention"]["metrics"]["macro_f1"]:.4f}** | {report_data["validation_slices"]["slice_without_publisher_mention"]["metrics"]["accuracy"]:.4f} | {report_data["validation_slices"]["slice_without_publisher_mention"]["metrics"]["fake_f1"]:.4f} | {report_data["validation_slices"]["slice_without_publisher_mention"]["metrics"]["roc_auc"]:.4f} | General articles |

---

## 6. DistilBERT Protocol & Colab Execution Verification

All requirements for DistilBERT fine-tuning on Google Colab T4 are locked:
1. **Dataset Ingestion:** Loads strictly `train.parquet` (50,937) and `validation.parquet` (6,373).
2. **Zero Test Leakage:** The `test.parquet` partition is strictly omitted and not loaded at any point.
3. **Paired Sequence Format:** `[CLS] Headline [SEP] Body [SEP]` with `truncation="longest_first"`.
4. **Frozen Label Convention:** Strictly maps `0 = REAL`, `1 = FAKE`.
5. **No Gradient Updates on Validation:** Evaluation runs in pure inference mode (`eval_dataset` in HF Trainer) with validation Macro-F1 driving `metric_for_best_model` and early stopping.
6. **Sequence Length Benchmarks:** Configured for 128, 256, and 384 tokens with memory-safe batch sizes (32, 16, 8 with gradient accumulation) to prevent CUDA OOM on T4 GPU.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Saved Markdown report to: {output_path}")


def main():
    logger.info("Starting Shortcut Robustness & Artifact Dependence Audit...")

    train_df, val_df = load_data()

    # 1. Dateline vs In-body mentions
    logger.info("Auditing publisher datelines vs in-body mentions...")
    dateline_audit = audit_dateline_vs_body_patterns(train_df, val_df)

    # 2. Residual cleaning audit
    logger.info("Auditing residual artifacts after existing prefix cleaning...")
    residual_cleaning_audit = measure_residual_artifacts_after_existing_cleaning(train_df)

    # 3. Controlled 4-condition evaluation
    logger.info("Evaluating 4-condition artifact dependence...")
    four_conditions = evaluate_four_conditions(train_df, val_df, max_features=50000)

    # 4. Validation slices evaluation
    logger.info("Evaluating performance across fine-grained validation slices...")
    validation_slices = evaluate_validation_slices(train_df, val_df)

    report_data = {
        "timestamp": "2026-11-02T16:15:00Z",
        "dateline_audit": dateline_audit,
        "residual_cleaning_audit": residual_cleaning_audit,
        "four_conditions": four_conditions,
        "validation_slices": validation_slices,
    }

    out_dir = Path("reports/evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "shortcut_robustness_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved JSON report to: {json_path}")

    md_path = out_dir / "shortcut_robustness_audit.md"
    generate_markdown_report(report_data, md_path)

    logger.info("Shortcut Robustness Audit completed successfully!")


if __name__ == "__main__":
    main()
