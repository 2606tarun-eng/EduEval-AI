import streamlit as st
import requests
import json
import re

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EduEval AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, modern look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* ── Global ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Header banner ── */
        .edu-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        .edu-header h1 {
            color: #e2e8f0;
            font-size: 2.4rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.5px;
        }
        .edu-header p {
            color: #94a3b8;
            font-size: 1rem;
            margin: 0;
        }
        .edu-header .badge {
            display: inline-block;
            background: #0f3460;
            border: 1px solid #1e4d8c;
            color: #60a5fa;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            margin-bottom: 0.75rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* ── Section cards ── */
        .section-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .section-label {
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        /* ── Result cards ── */
        .result-correct  { background:#052e16; border:1px solid #16a34a; border-radius:12px; padding:1.25rem; }
        .result-contradictory { background:#422006; border:1px solid #d97706; border-radius:12px; padding:1.25rem; }
        .result-incorrect { background:#3f1d1d; border:1px solid #dc2626; border-radius:12px; padding:1.25rem; }
        .result-label { font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }
        .result-value { font-size:2rem; font-weight:700; margin-top:0.2rem; }
        .correct-text { color:#4ade80; }
        .contradictory-text { color:#fbbf24; }
        .incorrect-text { color:#f87171; }

        /* ── Reasoning box ── */
        .reasoning-box {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-left: 4px solid #6366f1;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        /* ── Divider ── */
        .section-divider { border-top: 1px solid #1e293b; margin: 1.5rem 0; }

        /* ── Submit button override ── */
        div.stButton > button {
            background: linear-gradient(135deg, #6366f1, #818cf8);
            color: white;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 2.5rem;
            width: 100%;
            transition: opacity 0.2s;
        }
        div.stButton > button:hover { opacity: 0.88; }

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Constants & OCR API Setup
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000/evaluate"
OCR_URL = "http://localhost:8000/extract-text"

CATEGORY_CONFIG = {
    "correct":       {"card": "result-correct",       "text": "correct-text",       "icon": "✅", "label": "Correct"},
    "contradictory": {"card": "result-contradictory",  "text": "contradictory-text", "icon": "⚠️", "label": "Contradictory"},
    "incorrect":     {"card": "result-incorrect",      "text": "incorrect-text",     "icon": "❌", "label": "Incorrect"},
}

# ---------------------------------------------------------------------------
# Session State Initialization for Inputs & File Tracking
# ---------------------------------------------------------------------------

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""
if "ref_input" not in st.session_state:
    st.session_state["ref_input"] = ""
if "student_input" not in st.session_state:
    st.session_state["student_input"] = ""
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = {}


def extract_ocr_text(uploaded_file, field_name: str) -> str:
    """Send uploaded image to backend /extract-text and return the string."""
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/png",
            )
        }
        res = requests.post(OCR_URL, files=files, timeout=60)
        if res.status_code == 200:
            return res.json().get("extracted_text", "")
        else:
            st.error(f"OCR Error ({res.status_code}): {res.text}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Cannot reach OCR service: {exc}")
    return ""


def split_qa_text(raw_text: str) -> tuple[str | None, str | None]:
    """
    Detects if raw_text contains both a Question and an Answer.
    Returns (question_part, answer_part) or (None, None) if no clear split found.
    Handles markers like:
      - Ans:, Answer:, Ans -, Answer -, Ans., Solution:, Sol:
      - Q: ... A: ...
      - Student Answer:
    """
    if not raw_text or not raw_text.strip():
        return None, None

    patterns = [
        r'(?i)(?:[\r\n]+|[.?!]\s+|\s{2,}|\A)\s*(?:student[\'\s]*s?\s+answer|answer|ans|solution|soln|sol)[\s.:\-]+',
        r'(?i)(?:[\r\n]+|[.?!]\s+)\s*(?:a)[\s.:\-]+',
    ]

    for pat in patterns:
        matches = list(re.finditer(pat, raw_text))
        if matches:
            chosen = None
            for m in matches:
                if m.start() > 3:
                    chosen = m
                    break
            if not chosen and matches and matches[0].start() > 0:
                chosen = matches[0]

            if chosen:
                split_idx = chosen.start()
                if raw_text[split_idx] in '.?!':
                    split_idx += 1
                q_part = raw_text[:split_idx].strip()
                a_part = raw_text[chosen.end():].strip()

                # Clean leading 'Q:' / 'Question:' / 'Q1.' from q_part if present
                q_part = re.sub(r'(?i)^(?:question|q)[\s.:\-0-9]*', '', q_part).strip()
                # Clean leading Answer marker from a_part if still present
                a_part = re.sub(r'(?i)^(?:student[\'\s]*s?\s+answer|answer|ans|solution|soln|sol|a)[\s.:\-]+', '', a_part).strip()

                if len(q_part) >= 3 and len(a_part) >= 1:
                    return q_part, a_part

    return None, None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="edu-header">
        <div class="badge">🏆 Hackathon Build · ED-05</div>
        <h1>🎓 EduEval AI</h1>
        <p>Semantic Answer Assessor — AI-powered evaluation of student responses</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input Section
# ---------------------------------------------------------------------------

col_q, col_sep, col_s = st.columns([5, 0.2, 5])

# ── LEFT: Question + Reference Answer ──────────────────────────────────────
with col_q:

    # · Question ·
    st.markdown('<p class="section-label">📌 Question</p>', unsafe_allow_html=True)
    q_img = st.file_uploader(
        "📷 Upload question image (Auto-OCR)",
        type=["png", "jpg", "jpeg", "webp"],
        key="q_img_file",
        help="Upload an image to automatically extract question text using EasyOCR.",
    )
    if q_img is not None:
        file_sig = f"{q_img.name}_{len(q_img.getvalue())}"
        if st.session_state.processed_files.get("q_img") != file_sig:
            with st.spinner("🔍 Extracting question text from image via OCR..."):
                extracted = extract_ocr_text(q_img, "Question")
                if extracted:
                    q_split, a_split = split_qa_text(extracted)
                    if q_split and a_split:
                        st.session_state["question_input"] = q_split
                        if not st.session_state.get("student_input", "").strip():
                            st.session_state["student_input"] = a_split
                    else:
                        st.session_state["question_input"] = extracted
                    st.session_state.processed_files["q_img"] = file_sig
                    st.rerun()
                else:
                    st.session_state.processed_files["q_img"] = file_sig

        with st.expander("🖼️ View uploaded Question image", expanded=False):
            st.image(q_img, use_container_width=True)

    question = st.text_area(
        label="question_text",
        placeholder="Type, paste, or upload an image above to extract the question…",
        height=130,
        label_visibility="collapsed",
        key="question_input",
    )

    # Smart Auto-Split detector if both Q & A exist in Question box
    detected_q, detected_a = split_qa_text(question)
    if detected_q and detected_a and not st.session_state.get("student_input", "").strip():
        def do_split(q_val, a_val):
            st.session_state["question_input"] = q_val
            st.session_state["student_input"] = a_val

        col_s_msg, col_s_btn = st.columns([6, 4])
        with col_s_msg:
            st.caption("✨ *Detected Question & Answer together!*")
        with col_s_btn:
            st.button(
                "⚡ Split into boxes",
                key="btn_split_qa_box",
                on_click=do_split,
                args=(detected_q, detected_a),
            )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # · Reference Answer ·
    st.markdown('<p class="section-label">📖 Reference Answer <span style="color:#475569;font-weight:400;font-size:0.72rem">(optional)</span></p>', unsafe_allow_html=True)
    ref_img = st.file_uploader(
        "📷 Upload reference answer image (Auto-OCR)",
        type=["png", "jpg", "jpeg", "webp"],
        key="ref_img_file",
        help="Upload an image to automatically extract reference answer text.",
    )
    if ref_img is not None:
        file_sig = f"{ref_img.name}_{len(ref_img.getvalue())}"
        if st.session_state.processed_files.get("ref_img") != file_sig:
            with st.spinner("🔍 Extracting reference answer from image via OCR..."):
                extracted = extract_ocr_text(ref_img, "Reference Answer")
                if extracted:
                    st.session_state["ref_input"] = extracted
                    st.session_state.processed_files["ref_img"] = file_sig
                    st.rerun()
                else:
                    st.session_state.processed_files["ref_img"] = file_sig

        with st.expander("🖼️ View uploaded Reference Answer image", expanded=False):
            st.image(ref_img, use_container_width=True)

    reference_answer = st.text_area(
        label="reference_answer_text",
        placeholder="Paste model answer or upload image above (leave blank to skip)…",
        height=130,
        label_visibility="collapsed",
        key="ref_input",
    )

# ── RIGHT: Student Answer ──────────────────────────────────────────────────
with col_s:
    st.markdown('<p class="section-label">✏️ Student Answer</p>', unsafe_allow_html=True)
    stu_img = st.file_uploader(
        "📷 Upload student answer image (Auto-OCR)",
        type=["png", "jpg", "jpeg", "webp"],
        key="stu_img_file",
        help="Upload handwritten or printed student answer sheet to extract text automatically.",
    )
    if stu_img is not None:
        file_sig = f"{stu_img.name}_{len(stu_img.getvalue())}"
        if st.session_state.processed_files.get("stu_img") != file_sig:
            with st.spinner("🔍 Extracting student answer from image via OCR..."):
                extracted = extract_ocr_text(stu_img, "Student Answer")
                if extracted:
                    st.session_state["student_input"] = extracted
                    st.session_state.processed_files["stu_img"] = file_sig
                    st.rerun()
                else:
                    st.session_state.processed_files["stu_img"] = file_sig

        with st.expander("🖼️ View uploaded Student Answer image", expanded=False):
            st.image(stu_img, use_container_width=True)

    student_answer = st.text_area(
        label="student_answer_text",
        placeholder="Type, paste, or upload student's answer image above…",
        height=330,
        label_visibility="collapsed",
        key="student_input",
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    submitted = st.button("🚀 Evaluate Answer", use_container_width=True)

# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------

if submitted:
    # ── Auto-Split Fallback ──────────────────────────────────────────────────
    # If student_answer is blank, but question contains both Q & A, auto-split them for evaluation!
    if not student_answer.strip() and question.strip():
        auto_q, auto_a = split_qa_text(question)
        if auto_q and auto_a:
            question = auto_q
            student_answer = auto_a
            st.toast("⚡ Auto-split Question & Student Answer for evaluation!", icon="🎯")

    # ── Validation ──────────────────────────────────────────────────────────
    if not question.strip():
        st.error("⚠️  Please enter a **Question** before submitting.")
        st.stop()
    if not student_answer.strip():
        st.error("⚠️  Please enter the **Student Answer** before submitting.")
        st.stop()

    # ── Payload — STRICTLY matches EvaluationRequest schema ─────────────────
    payload: dict = {
        "question": question.strip(),
        "student_answer": student_answer.strip(),
    }
    if reference_answer.strip():
        payload["reference_answer"] = reference_answer.strip()
    # reference_answer is omitted entirely when blank (backend treats None)

    # ── API Call ─────────────────────────────────────────────────────────────
    with st.spinner("🔍 Evaluating with AI…"):
        try:
            response = requests.post(
                BACKEND_URL,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data: dict = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                "🔌 **Cannot reach backend.** Make sure FastAPI is running on `http://localhost:8000`."
            )
            st.stop()
        except requests.exceptions.Timeout:
            st.error("⏱️ **Request timed out.** The backend took too long to respond.")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"🚨 **Backend error {exc.response.status_code}:** {exc.response.text}")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            st.stop()

    # ── Parse Response ────────────────────────────────────────────────────────
    category: str        = data.get("category", "incorrect")
    probabilities: dict  = data.get("probabilities", {})
    reasoning: str       = data.get("reasoning", "No reasoning provided.")
    cfg                  = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["incorrect"])

    # Confidence = probability of the winning category
    confidence: float = probabilities.get(category, 0.0)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Evaluation Results")

    # ── Row 1: Category card + Confidence metrics ─────────────────────────
    res_col1, res_col2 = st.columns([3, 5])

    with res_col1:
        st.markdown(
            f"""
            <div class="{cfg['card']}">
                <div class="result-label {cfg['text']}">{cfg['icon']} Verdict</div>
                <div class="result-value {cfg['text']}">{cfg['label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown('<p class="section-label">📈 Category Probabilities</p>', unsafe_allow_html=True)

        prob_correct       = probabilities.get("correct", 0.0)
        prob_contradictory = probabilities.get("contradictory", 0.0)
        prob_incorrect     = probabilities.get("incorrect", 0.0)

        col_c, col_ct, col_i = st.columns(3)
        col_c.metric("✅ Correct",        f"{prob_correct * 100:.1f}%")
        col_ct.metric("⚠️ Contradictory", f"{prob_contradictory * 100:.1f}%")
        col_i.metric("❌ Incorrect",      f"{prob_incorrect * 100:.1f}%")

        st.markdown('<p class="section-label" style="margin-top:0.75rem">🎯 Confidence Score</p>', unsafe_allow_html=True)
        st.progress(confidence, text=f"{confidence * 100:.1f}% confidence in **{cfg['label']}**")

    # ── Row 2: Reasoning ──────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">💬 Semantic Reasoning</p>', unsafe_allow_html=True)

    with st.expander("View detailed AI reasoning", expanded=True):
        st.markdown(
            f'<div class="reasoning-box">{reasoning}</div>',
            unsafe_allow_html=True,
        )

    # ── Debug (hidden by default) ─────────────────────────────────────────
    with st.expander("🛠️ Raw API Response (debug)", expanded=False):
        st.json(data)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="text-align:center;color:#334155;font-size:0.8rem;margin-top:3rem;padding-top:1rem;border-top:1px solid #1e293b">
        EduEval AI · ED-05 Hackathon · Powered by FastAPI + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
