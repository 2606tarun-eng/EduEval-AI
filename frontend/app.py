import streamlit as st
import requests
import json
import re
import os

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
# Custom CSS — Light Theme, Professional & Clean
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }

        /* ── Streamlit main background ── */
        .stApp { background-color: #f1f5f9; }
        .block-container { padding-top: 1.5rem !important; }

        /* ── TOP HEADER BANNER ── */
        .hero-banner {
            background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 50%, #60a5fa 100%);
            border-radius: 20px;
            padding: 2.2rem 3rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 30px rgba(59,130,246,0.25);
        }
        .hero-left h1 {
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.5px;
        }
        .hero-left p {
            color: rgba(255,255,255,0.85);
            font-size: 1rem;
            margin: 0 0 0.8rem 0;
        }
        .hero-right {
            text-align: right;
        }
        .hero-right .team-name {
            color: rgba(255,255,255,0.9);
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-right .team-label {
            color: rgba(255,255,255,0.6);
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .badge-pill {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.35);
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            margin-right: 0.4rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        /* ── STATS BAR ── */
        .stats-bar {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.8rem;
        }
        .stat-card {
            flex: 1;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.1rem 1.4rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .stat-card .stat-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #1d4ed8;
            display: block;
        }
        .stat-card .stat-label {
            font-size: 0.73rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.2rem;
        }

        /* ── INPUT SECTION CARD ── */
        .input-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.6rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            margin-bottom: 1.2rem;
        }
        .input-card-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        /* ── SECTION LABEL ── */
        .section-label {
            color: #475569;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        /* ── RESULT CARDS — Light versions ── */
        .result-correct {
            background: #f0fdf4;
            border: 2px solid #22c55e;
            border-radius: 14px;
            padding: 1.4rem;
        }
        .result-contradictory {
            background: #fffbeb;
            border: 2px solid #f59e0b;
            border-radius: 14px;
            padding: 1.4rem;
        }
        .result-incorrect {
            background: #fff1f2;
            border: 2px solid #ef4444;
            border-radius: 14px;
            padding: 1.4rem;
        }
        .result-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }
        .result-value {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .correct-text      { color: #15803d; }
        .contradictory-text{ color: #b45309; }
        .incorrect-text    { color: #dc2626; }

        /* ── REASONING BOX ── */
        .reasoning-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #3b82f6;
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            color: #334155;
            font-size: 0.95rem;
            line-height: 1.75;
        }

        /* ── DIVIDER ── */
        .section-divider {
            border-top: 1px solid #e2e8f0;
            margin: 1.8rem 0;
        }

        /* ── SUBMIT BUTTON ── */
        div.stButton > button {
            background: linear-gradient(135deg, #1d4ed8, #3b82f6);
            color: white;
            font-weight: 700;
            font-size: 1rem;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2.5rem;
            width: 100%;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px rgba(59,130,246,0.35);
        }
        div.stButton > button:hover {
            opacity: 0.9;
            box-shadow: 0 6px 20px rgba(59,130,246,0.45);
            transform: translateY(-1px);
        }

        /* ── HOW IT WORKS INFO BOXES ── */
        .how-step {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: center;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        }
        .how-step .step-icon { font-size: 1.8rem; }
        .how-step .step-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #1e293b;
            margin-top: 0.4rem;
        }
        .how-step .step-desc {
            font-size: 0.74rem;
            color: #64748b;
            margin-top: 0.2rem;
            line-height: 1.4;
        }

        /* ── FOOTER ── */
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.78rem;
            margin-top: 3rem;
            padding: 1.2rem 0;
            border-top: 1px solid #e2e8f0;
        }
        .footer b { color: #3b82f6; }

        /* ── HIDE STREAMLIT CHROME ── */
        #MainMenu, footer { visibility: hidden; }

        /* ── TEXTAREA STYLE ── */
        .stTextArea textarea {
            background: #f8fafc !important;
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 10px !important;
            color: #1e293b !important;
            font-size: 0.92rem !important;
        }
        .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
        }

        /* ── INFO BANNER ── */
        .info-banner {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 0.85rem 1.2rem;
            color: #1d4ed8;
            font-size: 0.84rem;
            font-weight: 500;
            margin-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants & API Setup
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://edueval-ai-njfc.onrender.com").rstrip("/")
BACKEND_URL = f"{DEFAULT_BASE_URL}/evaluate"
OCR_URL = f"{DEFAULT_BASE_URL}/extract-text"

CATEGORY_CONFIG = {
    "correct":       {"card": "result-correct",       "text": "correct-text",       "icon": "✅", "label": "Correct"},
    "contradictory": {"card": "result-contradictory",  "text": "contradictory-text", "icon": "⚠️", "label": "Contradictory"},
    "incorrect":     {"card": "result-incorrect",      "text": "incorrect-text",     "icon": "❌", "label": "Incorrect"},
}

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

for key, val in [("question_input", ""), ("ref_input", ""), ("student_input", ""), ("processed_files", {})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# OCR & Auto-Split Utilities
# ---------------------------------------------------------------------------

def extract_ocr_text(uploaded_file, field_name: str) -> str:
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
    if not raw_text or not raw_text.strip():
        return None, None
    patterns = [
        r'(?i)(?:[\r\n]+|[.?!]\s+|\s{2,}|\A)\s*(?:student[\'\\s]*s?\s+answer|answer|ans|solution|soln|sol)[\s.:\-]+',
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
                q_part = re.sub(r'(?i)^(?:question|q)[\s.:\-0-9]*', '', q_part).strip()
                a_part = re.sub(r'(?i)^(?:student[\'\\s]*s?\s+answer|answer|ans|solution|soln|sol|a)[\s.:\-]+', '', a_part).strip()
                if len(q_part) >= 3 and len(a_part) >= 1:
                    return q_part, a_part
    return None, None

# ---------------------------------------------------------------------------
# HERO HEADER BANNER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-left">
            <h1>🎓 EduEval AI</h1>
            <p>AI-Powered Semantic Answer Evaluator with Explainable Grading</p>
            <div>
                <span class="badge-pill">ED-05</span>
                <span class="badge-pill">XAI</span>
                <span class="badge-pill">SemEval 2013</span>
                <span class="badge-pill">Hackathon Build</span>
            </div>
        </div>
        <div class="hero-right">
            <div class="team-label">Team</div>
            <div class="team-name">⚡ Asynchronous</div>
            <div style="color:rgba(255,255,255,0.55);font-size:0.72rem;margin-top:0.5rem;">
                FastAPI · Streamlit · Google Gemini
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# STATS BAR
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="stats-bar">
        <div class="stat-card">
            <span class="stat-value">91.65</span>
            <div class="stat-label">Benchmark Score / 100</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">3-Way</span>
            <div class="stat-label">Semantic Grading</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">CF1–CF4</span>
            <div class="stat-label">Adversarial Defense</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">8,910</span>
            <div class="stat-label">Training Samples</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">100%</span>
            <div class="stat-label">Calibrated Confidence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------------------------

with st.expander("ℹ️ How it works", expanded=False):
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("""
        <div class="how-step">
            <div class="step-icon">📝</div>
            <div class="step-title">Step 1: Input</div>
            <div class="step-desc">Type your question & answer, or upload a handwritten sheet.</div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="how-step">
            <div class="step-icon">👁️</div>
            <div class="step-title">Step 2: OCR</div>
            <div class="step-desc">EasyOCR reads handwritten text and auto-fills the form.</div>
        </div>""", unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="how-step">
            <div class="step-icon">🤖</div>
            <div class="step-title">Step 3: AI Grades</div>
            <div class="step-desc">Google Gemini evaluates and classifies the student's answer.</div>
        </div>""", unsafe_allow_html=True)
    with h4:
        st.markdown("""
        <div class="how-step">
            <div class="step-icon">📊</div>
            <div class="step-title">Step 4: Results</div>
            <div class="step-desc">Get Verdict, Confidence Score, and detailed AI Reasoning.</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------------------------

col_q, col_gap, col_s = st.columns([5, 0.2, 5])

# ── LEFT: Question + Reference Answer ─────────────────────────────────────
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
            with st.spinner("🔍 Extracting question text via OCR..."):
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

    # Smart Auto-Split detector
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
    st.markdown(
        '<p class="section-label">📖 Reference Answer <span style="color:#94a3b8;font-weight:400;font-size:0.72rem">(optional)</span></p>',
        unsafe_allow_html=True,
    )
    ref_img = st.file_uploader(
        "📷 Upload reference answer image (Auto-OCR)",
        type=["png", "jpg", "jpeg", "webp"],
        key="ref_img_file",
        help="Upload an image to automatically extract reference answer text.",
    )
    if ref_img is not None:
        file_sig = f"{ref_img.name}_{len(ref_img.getvalue())}"
        if st.session_state.processed_files.get("ref_img") != file_sig:
            with st.spinner("🔍 Extracting reference answer via OCR..."):
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
        placeholder="Paste the model answer or upload image above (leave blank to skip)…",
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
        help="Upload handwritten or printed student answer sheet.",
    )
    if stu_img is not None:
        file_sig = f"{stu_img.name}_{len(stu_img.getvalue())}"
        if st.session_state.processed_files.get("stu_img") != file_sig:
            with st.spinner("🔍 Extracting student answer via OCR..."):
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
# SAMPLE QUICK TEST BUTTONS
# ---------------------------------------------------------------------------

st.markdown('<p class="section-label">⚡ Quick Sample Tests</p>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)

def load_sample(q, r, a):
    st.session_state["question_input"] = q
    st.session_state["ref_input"] = r
    st.session_state["student_input"] = a

with s1:
    st.button(
        "✅ Correct Answer Sample",
        key="sample_correct",
        on_click=load_sample,
        args=(
            "What is photosynthesis?",
            "The process by which plants convert sunlight into food using carbon dioxide and water.",
            "Plants make food using sunlight, CO2 and water.",
        ),
        use_container_width=True,
    )
with s2:
    st.button(
        "⚠️ Contradictory Sample",
        key="sample_contradictory",
        on_click=load_sample,
        args=(
            "What is photosynthesis?",
            "The process by which plants convert sunlight into food using carbon dioxide and water.",
            "Plants consume food and release carbon dioxide during photosynthesis.",
        ),
        use_container_width=True,
    )
with s3:
    st.button(
        "❌ Incorrect Answer Sample",
        key="sample_incorrect",
        on_click=load_sample,
        args=(
            "What is photosynthesis?",
            "The process by which plants convert sunlight into food using carbon dioxide and water.",
            "Photosynthesis is when animals digest their food in the stomach.",
        ),
        use_container_width=True,
    )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SUBMIT BUTTON
# ---------------------------------------------------------------------------

_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    submitted = st.button("🚀 Evaluate Answer", use_container_width=True)

# ---------------------------------------------------------------------------
# EVALUATION LOGIC
# ---------------------------------------------------------------------------

if submitted:
    # Auto-Split fallback
    if not student_answer.strip() and question.strip():
        auto_q, auto_a = split_qa_text(question)
        if auto_q and auto_a:
            question = auto_q
            student_answer = auto_a
            st.toast("⚡ Auto-split Question & Student Answer!", icon="🎯")

    # Validation
    if not question.strip():
        st.error("⚠️  Please enter a **Question** before submitting.")
        st.stop()
    if not student_answer.strip():
        st.error("⚠️  Please enter the **Student Answer** before submitting.")
        st.stop()

    # Payload
    payload: dict = {
        "question": question.strip(),
        "student_answer": student_answer.strip(),
    }
    if reference_answer.strip():
        payload["reference_answer"] = reference_answer.strip()

    # API Call
    with st.spinner("🤖 Evaluating with Google Gemini AI…"):
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=60)
            response.raise_for_status()
            data: dict = response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 **Cannot reach backend.** Make sure FastAPI is reachable at `{DEFAULT_BASE_URL}`.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("⏱️ **Request timed out.** The backend is spinning up from cold start — please retry in 30 seconds.")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"🚨 **Backend error {exc.response.status_code}:** {exc.response.text}")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            st.stop()

    # Parse Response
    category: str       = data.get("category", "incorrect")
    probabilities: dict = data.get("probabilities", {})
    reasoning: str      = data.get("reasoning", "No reasoning provided.")
    cfg                 = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["incorrect"])
    confidence: float   = probabilities.get(category, 0.0)

    # Results Header
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Evaluation Results")

    # Row 1: Verdict + Probabilities
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

    # Row 2: Reasoning
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">💬 AI Reasoning & Explanation</p>', unsafe_allow_html=True)
    with st.expander("View detailed AI reasoning", expanded=True):
        st.markdown(
            f'<div class="reasoning-box">{reasoning}</div>',
            unsafe_allow_html=True,
        )

    # Debug
    with st.expander("🛠️ Raw API Response (debug)", expanded=False):
        st.json(data)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <b>EduEval AI</b> &nbsp;·&nbsp; Team <b>Asynchronous</b> &nbsp;·&nbsp;
        ED-05 Hackathon &nbsp;·&nbsp; Powered by <b>FastAPI</b> + <b>Streamlit</b> + <b>Google Gemini</b>
        <br>
        <span style="color:#cbd5e1;font-size:0.72rem;">
            Benchmark Score: 91.65/100 &nbsp;|&nbsp; SemEval-2013 Task 7 &nbsp;|&nbsp; CF1–CF4 Adversarial Defense
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
