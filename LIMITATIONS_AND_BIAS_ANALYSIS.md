# EduEval AI (ED-05) — Limitations & Bias Analysis

## 1. Executive Summary
EduEval AI is an automated short-answer grading system designed to evaluate student answers based strictly on conceptual correctness and semantic alignment, rather than superficial features like writing style, length, or repeated keywords. This document provides the formal analysis of system capabilities, observed biases, robustness behaviors, and known limitations in accordance with the ED-05 problem statement.

---

## 2. Bias-Resistance Mechanisms

### A. Surface Style & Fluency Variations (CF1–CF3)
* **Design**: The system prompt explicitly instructs the assessment engine to evaluate the underlying proposition and ignore punctuation, capitalization, informal phrasing, and non-informative filler prefixes/suffixes.
* **Tested Variants**:
  * **CF1 (Normalization)**: Lowercasing and removing all punctuation.
  * **CF2 (Style Prefix)**: `"In my answer, I think that [student answer]"`.
  * **CF3 (Style Suffix)**: `"[student answer] This is my final answer."`.
* **Empirical Behavior**: Testing against CF1–CF3 demonstrates high consistency ($>90\%$), proving that confidence scores do not artificially drop due to informal grammar or conversational framing.

### B. Keyword-Stuffing Attacks (CF4)
* **Design**: Standard automated grading systems often rely on lexical overlap (e.g., BLEU, ROUGE, TF-IDF). Consequently, students who repeat domain keywords without forming coherent arguments can manipulate traditional evaluators.
* **Mitigation**: EduEval AI requires semantic claim verification. If an incorrect answer is appended with high-TF-IDF keywords from the reference answer, the model identifies that the keywords do not form a conceptually valid proposition and maintains an `incorrect` or `contradictory` classification.
* **Metric**: Evaluated via Resistance Utility:
  $$U_{CF4} = 1 - \frac{1}{N}\sum_{i=1}^N \max(0, P_{\text{correct}}(CF4_i) - P_{\text{correct}}(\text{original}_i))$$

---

## 3. Known Limitations & Potential Failure Modes

1. **Multilingual & Code-Switching Inputs**:
   * Current evaluation prompts and the EasyOCR pipeline are optimized primarily for English text. Submissions using code-switched vernacular (e.g., Hinglish) may exhibit lower classification confidence.
2. **Ambiguous or Under-Specified Questions**:
   * When no reference answer is provided and the question is underspecified, the LLM relies on broad general knowledge. In niche domain contexts (e.g., specialized curriculum definitions), this may lead to subtle misalignments with instructor intent.
3. **Complex Diagrammatic & Mathematical Proofs**:
   * The current OCR and semantic evaluation focus on textual responses. Answers requiring multi-step symbolic derivation or geometric diagram interpretation require specialized multimodal solvers.
4. **Latency & API Quotas**:
   * Using large language models in zero-shot JSON mode introduces inference latency (~5–15 seconds per answer). For high-throughput real-time deployment, distilled local models (e.g., fine-tuned DeBERTa-v3 on SemEval-2013 Task 7) would provide faster inference at minimal cost.

---

## 4. Reproducibility & Environment Setup

* **Backend Service**: FastAPI with Pydantic validation (`http://localhost:8000`).
* **Frontend UI**: Streamlit web dashboard with OCR extraction and auto-split (`http://localhost:8501`).
* **Evaluation Pipeline**: `python local_evaluation.py` executes automated counterfactual generation, API benchmarking, and prints macro-F1, consistency, $U_{CF4}$, and Brier calibration scores.
