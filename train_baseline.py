#!/usr/bin/env python3
"""
train_baseline.py — Supervised Training on SemEval 2013 Task 7 (ED-05)
======================================================================
Trains a baseline supervised classifier on the official SemEval 2013 Task 7
training split (Beetle + SciEntsBank) and outputs evaluation metrics.

Usage
-----
    python train_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from local_evaluation import SemEvalItem, parse_semeval_xml

# Path to the extracted SemEval 3-way training data
DEFAULT_TRAIN_DIR = Path("data/semeval-2013-task7/extracted_3way/training/3way")


def load_corpus(train_dir: Path) -> List[SemEvalItem]:
    """Recursively parse all XML files in the SemEval training directory."""
    print(f"[*] Scanning SemEval XML training files in: {train_dir} ...")
    items = parse_semeval_xml(train_dir)
    print(f"[+] Loaded {len(items)} training instances.")
    return items


def prepare_features(items: List[SemEvalItem]) -> Tuple[List[str], List[str]]:
    """Format input texts combining Question, Reference, and Student Answer."""
    texts: List[str] = []
    labels: List[str] = []

    for item in items:
        # Cross-encoder style formatting for lexical & semantic feature extraction
        combined = f"Question: {item.question} | Reference: {item.reference_answer} | Student: {item.student_answer}"
        texts.append(combined)
        labels.append(item.true_label)

    return texts, labels


def train_and_evaluate(
    data_dir: Path = DEFAULT_TRAIN_DIR,
    model_output_path: str = "baseline_model.joblib",
) -> None:
    """Train pipeline and display validation metrics."""
    if not data_dir.exists():
        print(f"[ERROR] Training directory not found: {data_dir}", file=sys.stderr)
        print("Please ensure data/semeval-2013-task7 is unzipped.", file=sys.stderr)
        sys.exit(1)

    items = load_corpus(data_dir)
    texts, labels = prepare_features(items)

    # Stratified 80/20 train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    print(f"[+] Training set size:   {len(X_train)} instances")
    print(f"[+] Validation set size: {len(X_val)} instances\n")

    print("[*] Training TF-IDF + LogisticRegression baseline model ...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=15000,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.5,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    print("[+] Model training completed successfully.")

    # Validation Evaluation
    val_preds = pipeline.predict(X_val)
    macro_f1 = f1_score(
        y_val,
        val_preds,
        labels=["correct", "contradictory", "incorrect"],
        average="macro",
    )

    print("\n" + "=" * 60)
    print("  SemEval 2013 Task 7 — Baseline Validation Results")
    print("=" * 60)
    print(
        classification_report(
            y_val,
            val_preds,
            labels=["correct", "contradictory", "incorrect"],
            digits=4,
        )
    )
    print(f"  Validation Macro-F1 Score: {macro_f1:.4f}")
    print("=" * 60 + "\n")

    # Save trained artifact
    joblib.dump(pipeline, model_output_path)
    print(f"[+] Saved serialized model artifact to: {model_output_path}\n")


if __name__ == "__main__":
    train_and_evaluate()
