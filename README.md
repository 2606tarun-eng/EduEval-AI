# EduEval AI — Bias-Resistant Short-Answer Assessment (ED-05)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end AI assessment engine built to solve **ED-05: Bias-Resistant Short-Answer Assessment**. The system evaluates student answers based strictly on semantic and conceptual correctness while remaining invariant to superficial writing styles (CF1–CF3) and resilient against keyword-stuffing attacks (CF4).

---

## 🌟 Key Features

1. **Three-Class Semantic Classifier**:
   * Evaluates answers into `correct`, `contradictory`, or `incorrect`.
   * Produces well-calibrated confidence probabilities summing to 1.0.
   * Generates 2–4 sentences of conceptual evidence explaining the evaluation decision.
2. **Robust Counterfactual Generation (CF1–CF4)**:
   * **CF1**: Punctuation removal and complete lowercasing (style normalization).
   * **CF2**: Style prefix insertion (`"In my answer, I think that "`).
   * **CF3**: Style suffix insertion (`" This is my final answer."`).
   * **CF4**: TF-IDF keyword repetition attack generator (resistant against keyword stuffing).
3. **Supervised Training Pipeline (`train_baseline.py`)**:
   * Direct ingestion of **8,910 official SemEval 2013 Task 7** training examples (Beetle + SciEntsBank).
   * Cross-encoder TF-IDF feature extraction with stratified validation split.
   * Produces `baseline_model.joblib` artifact.
4. **Automated Evaluation Pipeline (`local_evaluation.py`)**:
   * Directly parses public **SemEval 2013 Task 7** XML datasets.
   * Computes **Macro-F1 (45 pts)**, **CF Consistency (30 pts)**, **$U_{CF4}$ Resistance (15 pts)**, and **Multiclass Brier Calibration (5 pts)**.
5. **Interactive Dashboard with Vision OCR & Smart Split**:
   * Upload handwritten or printed exam scripts (JPEG/PNG).
   * Extracts text using **EasyOCR**.
   * Automatically detects and splits combined Question + Answer images or pasted text.

---

## 📁 Repository Structure

```
EduEval Ai/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application routes (/evaluate, /extract-text)
│   │   ├── llm_eval.py      # Bias-resistant LLM evaluation engine
│   │   ├── schemas.py       # Pydantic schemas (requests, responses, probabilities)
│   │   └── ocr_util.py      # EasyOCR image extraction module
│   ├── .env                 # API credentials & model settings
│   └── requirements.txt
├── frontend/
│   └── app.py               # Streamlit web dashboard with OCR and Smart Split
├── data/
│   ├── semeval-2013-task7/  # Official SemEval 2013 Task 7 dataset (8,910+ XML items)
│   └── sample_semeval.xml   # Sample SemEval XML benchmark dataset
├── train_baseline.py        # Supervised training on SemEval dataset (saves baseline_model.joblib)
├── local_evaluation.py      # Standalone CLI evaluation pipeline & metrics driver
├── test_backend.py          # Automated integration test suite
├── LIMITATIONS_AND_BIAS_ANALYSIS.md # Formal bias, robustness & limitations report
├── requirements.txt         # Complete project dependencies
├── .env.example             # Environment variable template
└── README.md                # Documentation and reproduction guide
```

---

## 🚀 Quick Start & Setup

### 1. Prerequisites & Installation
Clone the repository and install required packages in a Python virtual environment:

```bash
git clone https://github.com/your-org/edueval-ai.git
cd "EduEval Ai"

python -m venv .venv
# Activate virtual environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template and configure your API key in `backend/.env`:

```bash
cp .env.example backend/.env
```

Ensure `backend/.env` contains:
```ini
LLM_API_KEY=your_gemini_or_openai_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash
```

---

## 💻 Running the Application

### 1. Launch FastAPI Backend
```bash
cd backend
uvicorn app.main:app --port 8000 --reload
```
* Interactive API documentation: **`http://localhost:8000/docs`**
* Health check endpoint: **`http://localhost:8000/`**

### 2. Launch Streamlit Web Dashboard
In a new terminal window:
```bash
streamlit run frontend/app.py --server.port 8501
```
* Dashboard URL: **`http://localhost:8501`**

---

## 📊 Running Local Evaluation (ED-05 Scorecard)

The evaluation script runs automated testing against the SemEval XML dataset and generates the official ED-05 100-point scorecard:

```bash
# Run on sample SemEval dataset:
python local_evaluation.py --data-dir data/sample_semeval.xml --limit 3

# Run on a full directory of SemEval XML files:
python local_evaluation.py --data-dir path/to/semeval_xmls/ --limit 10

# Export benchmark results to JSON:
python local_evaluation.py --data-dir data/sample_semeval.xml --save-json results.json
```

### Judging Criteria Mapping

| Component | Weight | Metric Formula / Definition |
|-----------|:------:|-----------------------------|
| **Macro-F1** | 45% | $\text{f1\_score}(y_{\text{true}}, y_{\text{pred}}, \text{average}=\text{'macro'})$ |
| **CF1–CF3 Consistency** | 30% | Fraction of style variants whose prediction matches the original answer |
| **$U_{CF4}$ Resistance Utility** | 15% | $1 - \frac{1}{N}\sum_{i=1}^N \max(0, P_{\text{correct}}(CF4_i) - P_{\text{correct}}(\text{original}_i))$ |
| **Calibration Utility** | 5% | $1 - \text{Brier}$, where $\text{Brier} = \frac{1}{2N}\sum_{i=1}^N \sum_{k=1}^3 (p_{ik} - y_{ik})^2$ |
| **Reproducibility** | 5% | Clean code, dependency lockfiles, standalone scripts, and documentation |

---

## 🧪 Running Automated Tests

To run the backend integration test suite:
```bash
python test_backend.py
```

---

## 📄 Documentation
For detailed architectural choices, edge-case evaluations, and failure-mode mitigations, refer to **`LIMITATIONS_AND_BIAS_ANALYSIS.md`**.
