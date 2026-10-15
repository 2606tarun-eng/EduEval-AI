#!/usr/bin/env python3
"""
train_model_v2.py — CPU-only model search for ED-05
===================================================
Improves on ``train_baseline.py`` (TF-IDF + LogisticRegression, val Macro-F1
~0.47) without requiring a GPU.

Why CPU-only:
    The deployed service runs on Render's free tier, which caps instances at
    512 MB RAM. A PyTorch transformer would exhaust that budget the same way
    EasyOCR did. Every configuration tried here is a sparse linear model:
    sub-millisecond CPU inference, a small artifact, and no extra RAM.

Method:
    1. Split the *training* split into train/validation (stratified).
    2. Score candidate configurations by validation Macro-F1.
    3. Refit the winner on the full training split.
    4. Report held-out test Macro-F1 once, at the end.

Tuning happens on validation only -- the test split is never used for model
selection, so the reported number stays honest.

Usage:
    python train_model_v2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from local_evaluation import SemEvalItem, parse_semeval_xml

TRAIN_DIR = Path("data/semeval-2013-task7/extracted_3way/training/3way")
TEST_DIR = Path("data/semeval-2013-task7/extracted_3way/test/3way")
MODEL_OUT = "baseline_model_v2.joblib"

LABELS = ["correct", "contradictory", "incorrect"]


def load_items(path: Path) -> List[SemEvalItem]:
    items = parse_semeval_xml(path)
    print(f"  {path}: {len(items)} items")
    return items


def format_item(item: SemEvalItem) -> str:
    """Must match train_baseline.prepare_features and
    local_evaluation.format_for_model exactly."""
    return (
        f"Question: {item.question} | Reference: {item.reference_answer} "
        f"| Student: {item.student_answer}"
    )


# ---------------------------------------------------------------------------
# Candidate configurations
# ---------------------------------------------------------------------------

def word_tfidf(ngram=(1, 2), max_feat=30000, sublinear=True) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram,
        max_features=max_feat,
        sublinear_tf=sublinear,
        stop_words="english",
        min_df=2,
    )


def char_tfidf(ngram=(3, 5), max_feat=30000) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram,
        max_features=max_feat,
        sublinear_tf=True,
        min_df=3,
    )


def build_configs() -> List[Tuple[str, Pipeline]]:
    """Each entry is (name, pipeline)."""
    configs: List[Tuple[str, Pipeline]] = []

    # 1. Baseline for reference (current train_baseline.py).
    configs.append((
        "baseline: word(1,2)+LogReg",
        Pipeline([
            ("tfidf", word_tfidf((1, 2), 15000)),
            ("clf", LogisticRegression(C=1.5, max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
    ))

    # 2. Wider word features.
    configs.append((
        "word(1,2) 30k + LogReg C=4",
        Pipeline([
            ("tfidf", word_tfidf((1, 2), 30000)),
            ("clf", LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
    ))

    # 3. LinearSVC + calibration -> predict_proba, usually a stronger margin model.
    configs.append((
        "word(1,2) 30k + LinearSVC(cal)",
        Pipeline([
            ("tfidf", word_tfidf((1, 2), 30000)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=0.5, class_weight="balanced", random_state=42), cv=3, method="sigmoid")),
        ]),
    ))

    # 4. Word + char union -- char n-grams help with spelling/morphology.
    configs.append((
        "word+char union + LinearSVC(cal)",
        Pipeline([
            ("features", FeatureUnion([
                ("word", word_tfidf((1, 2), 30000)),
                ("char", char_tfidf((3, 5), 30000)),
            ])),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=0.5, class_weight="balanced", random_state=42), cv=3, method="sigmoid")),
        ]),
    ))

    # 5. Word + char union + LogReg (calibrated probabilities natively).
    configs.append((
        "word+char union + LogReg C=4",
        Pipeline([
            ("features", FeatureUnion([
                ("word", word_tfidf((1, 2), 30000)),
                ("char", char_tfidf((3, 5), 30000)),
            ])),
            ("clf", LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
    ))

    # 6. SGD with modified_huber -- fast, handles sparse well.
    configs.append((
        "word+char union + SGD(mod_huber)",
        Pipeline([
            ("features", FeatureUnion([
                ("word", word_tfidf((1, 2), 30000)),
                ("char", char_tfidf((3, 5), 30000)),
            ])),
            ("clf", SGDClassifier(
                loss="modified_huber", alpha=1e-5, max_iter=3000,
                tol=1e-4, class_weight="balanced", random_state=42)),
        ]),
    ))

    return configs


def evaluate(pipeline, X, y) -> Tuple[float, float]:
    """Return (macro_f1, brier)."""
    preds = pipeline.predict(X)
    f1 = f1_score(y, preds, labels=LABELS, average="macro", zero_division=0.0)

    brier = float("nan")
    if hasattr(pipeline, "predict_proba"):
        classes = [str(c) for c in pipeline.classes_]
        proba = pipeline.predict_proba(X)
        total = 0.0
        for row, label in zip(proba, y):
            for k in LABELS:
                p = row[classes.index(k)] if k in classes else 0.0
                total += (p - (1.0 if k == label else 0.0)) ** 2
        brier = total / (2.0 * len(y))

    return float(f1), brier


def main() -> None:
    print("=" * 70)
    print("  ED-05 CPU-only model search")
    print("=" * 70)

    print("\n[1/4] Loading data ...")
    train_items = load_items(TRAIN_DIR)
    X = [format_item(i) for i in train_items]
    y = [i.true_label for i in train_items]

    print("\n[2/4] Question-grouped validation split ...")
    # SemEval's official test split is built from *unseen questions* and *unseen
    # domains*. A random split leaks questions across the boundary and reports a
    # wildly optimistic score (observed: 0.77 validation vs 0.46 on the real test
    # split). Grouping by question_id reproduces the real difficulty and makes
    # model selection meaningful.
    groups = [i.question_id for i in train_items]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y, groups))
    tr_idx, va_idx = splits[0]

    X_tr = [X[i] for i in tr_idx]
    y_tr = [y[i] for i in tr_idx]
    X_va = [X[i] for i in va_idx]
    y_va = [y[i] for i in va_idx]

    q_tr = {groups[i] for i in tr_idx}
    q_va = {groups[i] for i in va_idx}
    overlap = q_tr & q_va
    print(f"  train: {len(X_tr)} ({len(q_tr)} questions)   "
          f"validation: {len(X_va)} ({len(q_va)} questions)")
    print(f"  question overlap train/validation: {len(overlap)}  (must be 0)")
    assert not overlap, "Grouped split leaked questions"

    print("\n[3/4] Comparing configurations on validation ...\n")
    results = []
    for name, pipeline in build_configs():
        started = time.time()
        try:
            pipeline.fit(X_tr, y_tr)
            f1, brier = evaluate(pipeline, X_va, y_va)
            took = time.time() - started
            results.append((f1, name, brier, took))
            print(f"  {name:<42} valF1={f1:.4f}  brier={brier:.4f}  [{took:.1f}s]")
        except Exception as exc:
            print(f"  {name:<42} FAILED: {type(exc).__name__}: {exc}")

    if not results:
        print("\n[ERROR] No configuration trained successfully.")
        sys.exit(1)

    results.sort(key=lambda r: r[0], reverse=True)
    best_f1, best_name, best_brier, _ = results[0]
    print(f"\n  WINNER: {best_name}   valF1={best_f1:.4f}")

    print("\n[4/4] Refitting winner on the FULL training split ...")
    winner = dict(build_configs())[best_name]
    winner.fit(X, y)

    print(f"\n  Validation classification report ({best_name}):")
    print(classification_report(
        y_va, winner.predict(X_va), labels=LABELS, digits=4, zero_division=0
    ))

    joblib.dump(winner, MODEL_OUT)
    size_mb = Path(MODEL_OUT).stat().st_size / (1024 * 1024)
    print(f"[+] Saved {MODEL_OUT} ({size_mb:.1f} MB)")

    # ---- Held-out test evaluation, reported once ----
    print("\n" + "=" * 70)
    print("  HELD-OUT TEST EVALUATION (never used for tuning)")
    print("=" * 70)
    test_items = load_items(TEST_DIR)
    X_te = [format_item(i) for i in test_items]
    y_te = [i.true_label for i in test_items]
    test_f1, test_brier = evaluate(winner, X_te, y_te)
    print(f"\n  Test Macro-F1 : {test_f1:.4f}   (baseline model: 0.4692)")
    print(f"  Test Brier    : {test_brier:.4f}")
    print()


if __name__ == "__main__":
    main()