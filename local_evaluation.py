#!/usr/bin/env python3
"""
local_evaluation.py — ED-05 Bias-Resistant Short-Answer Assessment
===================================================================
Standalone evaluation pipeline and benchmark driver for EduEval AI.

Complies with the SemEval 2013 Task 7 (Student Response Analysis)
three-way classification specification:
    - Labels: 'correct', 'contradictory', 'incorrect'
    - Counterfactual families: CF1, CF2, CF3, CF4
    - Metrics: Macro-F1 (45%), CF1-CF3 Consistency (30%),
               CF4 Resistance Utility UCF4 (15%),
               Calibration Utility 1 - Brier (5%),
               Reproducibility (5%)

Usage
-----
    # Run on sample SemEval XML data:
    python local_evaluation.py --data-dir data/sample_semeval.xml

    # Run on full SemEval directory with limit:
    python local_evaluation.py --data-dir path/to/semeval/xmls --limit 10

    # Custom API endpoint:
    python local_evaluation.py --api-url http://127.0.0.1:8000/evaluate
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import string
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

# Ensure stdout flushes immediately in non-interactive / terminal environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ===========================================================================
# 1. Data Structures & SemEval XML Parser
# ===========================================================================

VALID_LABELS: List[str] = ["correct", "contradictory", "incorrect"]

# Mapping from common dataset accuracy annotations to strict 3-way labels
LABEL_MAPPING: Dict[str, str] = {
    "correct": "correct",
    "contradictory": "contradictory",
    "incorrect": "incorrect",
    # 5-way SemEval annotations mapped to 3-way:
    "partially_correct_incomplete": "incorrect",
    "irrelevant": "incorrect",
    "non_domain": "incorrect",
}


@dataclass
class SemEvalItem:
    """Represents a single evaluated student answer instance."""

    question_id: str
    answer_id: str
    question: str
    reference_answer: str
    student_answer: str
    true_label: str  # 'correct' | 'contradictory' | 'incorrect'


def map_accuracy_label(raw_label: str) -> Optional[str]:
    """Map raw accuracy attribute from SemEval XML to strict 3-way label."""
    cleaned = (raw_label or "").strip().lower()
    return LABEL_MAPPING.get(cleaned)


def parse_semeval_xml(source_path: Path | str) -> List[SemEvalItem]:
    """Parse one or more SemEval 2013 Task 7 XML files.

    Expected XML structure:
      <question id="...">
        <questionText>...</questionText>
        <referenceAnswers>
          <referenceAnswer id="...">...</referenceAnswer>
        </referenceAnswers>
        <studentAnswers>
          <studentAnswer id="..." accuracy="correct">...</studentAnswer>
        </studentAnswers>
      </question>

    Parameters
    ----------
    source_path : Path | str
        Path to a single XML file or directory containing XML files.

    Returns
    -------
    List[SemEvalItem]
        Extracted student response instances.
    """
    path_obj = Path(source_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SemEval XML path does not exist: {source_path}")

    xml_files: List[Path] = []
    if path_obj.is_dir():
        xml_files = sorted(list(path_obj.glob("**/*.xml")))
    else:
        xml_files = [path_obj]

    if not xml_files:
        raise ValueError(f"No XML files found at: {source_path}")

    items: List[SemEvalItem] = []

    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
        except ET.ParseError as exc:
            print(f"[WARN] Failed to parse XML file {xf.name}: {exc}", file=sys.stderr)
            continue

        # Target all <question> tags (whether root or child)
        questions = root.findall(".//question")
        if not questions and root.tag == "question":
            questions = [root]

        for q_elem in questions:
            q_id = q_elem.get("id", "q_unknown")

            # Extract Question Text
            q_text_elem = q_elem.find("questionText")
            if q_text_elem is None or not q_text_elem.text:
                continue
            question_text = q_text_elem.text.strip()

            # Extract Reference Answer (take primary/first reference answer)
            ref_elem = q_elem.find(".//referenceAnswer")
            ref_answer = ref_elem.text.strip() if (ref_elem is not None and ref_elem.text) else ""

            # Extract Student Answers
            student_elems = q_elem.findall(".//studentAnswer")
            for sa_elem in student_elems:
                ans_id = sa_elem.get("id", f"{q_id}_ans")
                raw_accuracy = sa_elem.get("accuracy", "")
                mapped_label = map_accuracy_label(raw_accuracy)

                if mapped_label is None:
                    continue  # skip unmapped or unknown categories

                ans_text = (sa_elem.text or "").strip()
                if not ans_text:
                    continue

                items.append(
                    SemEvalItem(
                        question_id=q_id,
                        answer_id=ans_id,
                        question=question_text,
                        reference_answer=ref_answer,
                        student_answer=ans_text,
                        true_label=mapped_label,
                    )
                )

    return items


# ===========================================================================
# 2. Counterfactual Generator (CF1 – CF4)
# ===========================================================================


def generate_cf1(answer: str) -> str:
    """CF1 — Lowercase the entire answer and remove all punctuation.

    Meaning-preserving style transformation.
    """
    lowered = answer.lower()
    return lowered.translate(str.maketrans("", "", string.punctuation))


def generate_cf2(answer: str) -> str:
    """CF2 — Prepend the exact style text.

    'In my answer, I think that [student_answer]'
    """
    return "In my answer, I think that " + answer


def generate_cf3(answer: str) -> str:
    """CF3 — Append the exact style text.

    '[student_answer] This is my final answer.'
    """
    return answer + " This is my final answer."


def generate_cf4(answer: str, tfidf_vectorizer: TfidfVectorizer) -> str:
    """CF4 — Keyword-stuffing via training TF-IDF vectors.

    Identify the two eligible tokens in the answer with the largest
    TF-IDF values, breaking ties lexicographically, and append those two
    tokens once each separated by spaces. If fewer than two eligible
    tokens exist, append every eligible token once.
    """
    vec = tfidf_vectorizer.transform([answer.lower()])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    coo = vec.tocoo()
    token_scores: List[Tuple[str, float]] = []
    for col_idx, score in zip(coo.col, coo.data):
        token_scores.append((feature_names[col_idx], float(score)))

    if not token_scores:
        return answer

    # Sort descending by TF-IDF score; break ties lexicographically (ascending token)
    token_scores.sort(key=lambda item: (-item[1], item[0]))

    # Pick top 2 tokens
    top_tokens = [tok for tok, _ in token_scores[:2]]
    return answer + " " + " ".join(top_tokens)


def build_tfidf_vectorizer(corpus_answers: List[str]) -> TfidfVectorizer:
    """Fit a scikit-learn TfidfVectorizer on training answers per ED-05 rules.

    Parameters:
      - lowercase = True
      - stop_words = "english"
      - default token_pattern = r"(?u)\\b\\w\\w+\\b"
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
    )
    vectorizer.fit([a.lower() for a in corpus_answers])
    return vectorizer


# ===========================================================================
# 3. API Caller
# ===========================================================================


@dataclass
class APIPrediction:
    """Structured response from the evaluation API."""

    category: str
    probabilities: Dict[str, float]


def call_evaluate_api(
    api_url: str,
    question: str,
    student_answer: str,
    reference_answer: Optional[str] = None,
    *,
    timeout: int = 180,
    max_retries: int = 5,
) -> APIPrediction:
    """Call the /evaluate endpoint and return validated category and probabilities."""
    payload: Dict[str, Any] = {
        "question": question,
        "student_answer": student_answer,
    }
    if reference_answer:
        payload["reference_answer"] = reference_answer

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return APIPrediction(
                category=str(data["category"]).lower(),
                probabilities={
                    "correct": float(data["probabilities"]["correct"]),
                    "contradictory": float(data["probabilities"]["contradictory"]),
                    "incorrect": float(data["probabilities"]["incorrect"]),
                },
            )
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_exc = exc
            exc_str = str(exc)
            # When hitting rate limit (429) or gateway spikes (502/503), wait 14s
            if any(code in exc_str for code in ("429", "502", "503", "RESOURCE_EXHAUSTED")):
                wait_sec = 14
            else:
                wait_sec = 2 ** attempt

            print(f"    [WARN] API attempt {attempt}/{max_retries} failed: {exc}", file=sys.stderr)
            if attempt < max_retries:
                print(f"      Retrying in {wait_sec}s ...", file=sys.stderr)
                time.sleep(wait_sec)

    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_exc}")


# ===========================================================================
# 4. Metric Formulas (Strict ED-05 Implementation)
# ===========================================================================


def compute_macro_f1(true_labels: List[str], pred_labels: List[str]) -> float:
    """Original three-way macro-averaged F1 score."""
    return float(
        f1_score(
            true_labels,
            pred_labels,
            labels=VALID_LABELS,
            average="macro",
            zero_division=0.0,
        )
    )


def compute_cf_consistency(
    original_preds: List[str],
    cf_preds: Dict[str, List[str]],
) -> float:
    """CF1-CF3 Counterfactual Consistency.

    The fraction of CF1-CF3 variants whose predicted class equals the
    model's prediction on their corresponding original answer.
    """
    total = 0
    matches = 0
    for cf_family in ("CF1", "CF2", "CF3"):
        preds = cf_preds.get(cf_family, [])
        for orig, cf_pred in zip(original_preds, preds):
            total += 1
            if orig == cf_pred:
                matches += 1

    return float(matches / total) if total > 0 else 1.0


def compute_ucf4(
    original_probs: List[Dict[str, float]],
    cf4_probs: List[Dict[str, float]],
) -> float:
    """CF4 Resistance Utility.

    Formula:
        U_CF4 = 1 - (1/N) * sum_{i=1 to N} max(0, P_correct(CF4_i) - P_correct(original_i))
    Clipped to [0, 1].
    """
    n = len(original_probs)
    if n == 0:
        return 1.0

    total_increase = 0.0
    for orig, cf4 in zip(original_probs, cf4_probs):
        diff = cf4.get("correct", 0.0) - orig.get("correct", 0.0)
        total_increase += max(0.0, diff)

    u_cf4 = 1.0 - (total_increase / n)
    return float(np.clip(u_cf4, 0.0, 1.0))


def compute_brier_score(
    true_labels: List[str],
    pred_probs: List[Dict[str, float]],
) -> Tuple[float, float]:
    """Normalized Multiclass Brier Score & Calibration Utility.

    Formula:
        Brier = (1 / 2N) * sum_{i=1 to N} sum_{k=1 to 3} (p_ik - y_ik)^2
        Calibration Utility = 1 - Brier
    """
    n = len(true_labels)
    if n == 0:
        return 0.0, 1.0

    total_squared_error = 0.0
    for true_label, probs in zip(true_labels, pred_probs):
        for k in VALID_LABELS:
            y_ik = 1.0 if k == true_label else 0.0
            p_ik = probs.get(k, 0.0)
            total_squared_error += (p_ik - y_ik) ** 2

    brier = total_squared_error / (2.0 * n)
    calibration_utility = float(np.clip(1.0 - brier, 0.0, 1.0))
    return float(brier), calibration_utility


# ===========================================================================
# 5. CLI Driver & Report Formatter
# ===========================================================================

DIVIDER = "=" * 74
THIN_DIV = "-" * 74


def print_score_bar(name: str, score: float, width: int = 42) -> None:
    """Display an ASCII progress bar for terminal output."""
    bar_length = 20
    filled = int(round(score * bar_length))
    bar = "#" * filled + "." * (bar_length - filled)
    print(f"    {name:<{width}} [{bar}] {score * 100:6.2f}%")


def run_pipeline(
    data_path: str,
    api_url: str,
    limit: Optional[int] = None,
    delay: float = 2.0,
    save_json_path: Optional[str] = None,
) -> None:
    """Execute the complete ED-05 evaluation pipeline."""
    print()
    print(DIVIDER)
    print("  [EduEval AI] -- ED-05 Bias-Resistant Evaluation Pipeline")
    print(DIVIDER)
    print(f"  Dataset Path : {data_path}")
    print(f"  API Endpoint : {api_url}")
    if limit:
        print(f"  Sample Limit : {limit} instances")
    print(DIVIDER)
    print()

    # 1. Parse Dataset
    print("[1/4] Loading and parsing SemEval XML data ...")
    items = parse_semeval_xml(data_path)
    if not items:
        print("[ERROR] No valid items extracted from XML files.", file=sys.stderr)
        sys.exit(1)

    print(f"      Total items parsed: {len(items)}")
    if limit and limit < len(items):
        items = items[:limit]
        print(f"      Applied evaluation limit: evaluating {len(items)} items")

    # 2. Fit TF-IDF on corpus answers for CF4
    print("[2/4] Fitting TF-IDF vectorizer for CF4 keyword extraction ...")
    all_answers = [item.student_answer for item in items]
    tfidf_vectorizer = build_tfidf_vectorizer(all_answers)

    # 3. Storage for results
    true_labels: List[str] = []
    original_preds: List[str] = []
    original_probs: List[Dict[str, float]] = []
    cf_preds: Dict[str, List[str]] = {"CF1": [], "CF2": [], "CF3": [], "CF4": []}
    cf4_probs: List[Dict[str, float]] = []

    print("[3/4] Running original and counterfactual evaluations against API ...\n")

    for idx, item in enumerate(items, start=1):
        print(f"  > [{idx}/{len(items)}] Question ID: {item.question_id} | Answer ID: {item.answer_id}")
        print(f"    Question   : {item.question[:65]}..." if len(item.question) > 65 else f"    Question   : {item.question}")
        print(f"    True Label : {item.true_label}")
        true_labels.append(item.true_label)

        # -- Evaluate Original Answer --
        orig_pred = call_evaluate_api(api_url, item.question, item.student_answer, item.reference_answer)
        original_preds.append(orig_pred.category)
        original_probs.append(orig_pred.probabilities)

        match_tag = "[OK]" if orig_pred.category == item.true_label else "[MISS]"
        probs_str = ", ".join(f"{k}: {v:.2f}" for k, v in orig_pred.probabilities.items())
        print(f"    [Orig] Pred : {orig_pred.category:<13} {match_tag:<6} ({probs_str})")
        time.sleep(delay)

        # -- Generate Counterfactuals --
        cf_texts = {
            "CF1": generate_cf1(item.student_answer),
            "CF2": generate_cf2(item.student_answer),
            "CF3": generate_cf3(item.student_answer),
            "CF4": generate_cf4(item.student_answer, tfidf_vectorizer),
        }

        # -- Evaluate CF1 - CF4 --
        for cf_key, cf_text in cf_texts.items():
            cf_pred = call_evaluate_api(api_url, item.question, cf_text, item.reference_answer)
            cf_preds[cf_key].append(cf_pred.category)

            if cf_key == "CF4":
                cf4_probs.append(cf_pred.probabilities)

            cf_match = "[MATCH]" if cf_pred.category == orig_pred.category else "[DIFF]"
            print(f"    [{cf_key}] Pred : {cf_pred.category:<13} {cf_match:<6}")
            time.sleep(delay)

        print()

    # 4. Compute Metrics
    print("[4/4] Calculating final benchmark metrics ...\n")
    macro_f1 = compute_macro_f1(true_labels, original_preds)
    cf_consistency = compute_cf_consistency(original_preds, cf_preds)
    ucf4 = compute_ucf4(original_probs, cf4_probs)
    brier, calib_utility = compute_brier_score(true_labels, original_probs)

    # 5. Scoring Weights (ED-05)
    w_f1 = 45.0
    w_consistency = 30.0
    w_ucf4 = 15.0
    w_calib = 5.0

    score_f1 = macro_f1 * w_f1
    score_consistency = cf_consistency * w_consistency
    score_ucf4 = ucf4 * w_ucf4
    score_calib = calib_utility * w_calib
    total_score = score_f1 + score_consistency + score_ucf4 + score_calib

    # 6. Terminal Report
    print(DIVIDER)
    print("  ED-05 BENCHMARK EVALUATION SCORECARD")
    print(DIVIDER)
    print(f"    {'Metric Component':<40} {'Weight':>6}   {'Score':>8}   {'Weighted':>8}")
    print(f"    {THIN_DIV[:40]} {'-' * 6}   {'-' * 8}   {'-' * 8}")
    print(f"    {'Three-way Macro-F1':<40} {'45%':>6}   {macro_f1:>7.4f}   {score_f1:>7.2f}")
    print(f"    {'CF1-CF3 Counterfactual Consistency':<40} {'30%':>6}   {cf_consistency:>7.4f}   {score_consistency:>7.2f}")
    print(f"    {'CF4 Resistance Utility (U_CF4)':<40} {'15%':>6}   {ucf4:>7.4f}   {score_ucf4:>7.2f}")
    print(f"    {'Calibration Utility (1 - Brier)':<40} {'5%':>6}   {calib_utility:>7.4f}   {score_calib:>7.2f}")
    print(f"    {THIN_DIV[:40]} {'-' * 6}   {'-' * 8}   {'-' * 8}")
    print(f"    {'TOTAL WEIGHTED SCORE':<40} {'100%':>6}   {'':>8}   {total_score:>7.2f} / 95.00")
    print(f"    (Remaining 5 points Reproducibility assessed independently)")
    print()

    print("  > Visual Distribution")
    print(f"  {THIN_DIV}")
    print_score_bar("Macro-F1 (45 pts)", macro_f1)
    print_score_bar("CF Consistency (30 pts)", cf_consistency)
    print_score_bar("CF4 Resistance (15 pts)", ucf4)
    print_score_bar("Calibration (5 pts)", calib_utility)
    print()

    print("  > Multiclass Brier Score Details")
    print(f"  {THIN_DIV}")
    print(f"    Normalized Multiclass Brier Score : {brier:.6f}  (lower is better)")
    print(f"    Effective Calibration Score (1-B) : {calib_utility:.6f}  (higher is better)")
    print()

    print(DIVIDER)
    print(f"  Final Benchmark Score: {total_score:.2f} / 95.00 (Excl. Reproducibility)")
    print(DIVIDER)
    print()

    # Optional JSON Export
    if save_json_path:
        out_data = {
            "dataset_path": str(data_path),
            "sample_count": len(items),
            "metrics": {
                "macro_f1": macro_f1,
                "cf1_cf3_consistency": cf_consistency,
                "ucf4_resistance": ucf4,
                "brier_score": brier,
                "calibration_utility": calib_utility,
                "weighted_score": total_score,
            },
            "evaluations": [
                {
                    "question_id": item.question_id,
                    "answer_id": item.answer_id,
                    "true_label": true,
                    "predicted": pred,
                    "probabilities": prob,
                }
                for item, true, pred, prob in zip(items, true_labels, original_preds, original_probs)
            ],
        }
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"  [+] Saved complete evaluation results to: {save_json_path}\n")


# ===========================================================================
# 6. Entrypoint
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EduEval AI -- ED-05 SemEval 2013 Task 7 Local Evaluation Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data/sample_semeval.xml",
        help="Path to SemEval XML file or directory containing XML files.",
    )
    parser.add_argument(
        "--api-url",
        "-u",
        type=str,
        default="http://127.0.0.1:8000/evaluate",
        help="EduEval AI backend evaluation API endpoint URL.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=3,
        help="Maximum number of student answers to evaluate (default: 3).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between successive API calls for rate pacing.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to output the evaluation summary and metrics as JSON.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            data_path=args.data_dir,
            api_url=args.api_url,
            limit=args.limit,
            delay=args.delay,
            save_json_path=args.save_json,
        )
    except KeyboardInterrupt:
        print("\n[!] Evaluation interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
