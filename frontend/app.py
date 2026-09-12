import streamlit as st
import requests
import re
import os

st.set_page_config(
    page_title="EduEval AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0a0f; color: #e2e8f0; }
    .stApp { background-color: #0a0a0f; }
    .block-container { padding-top: 1.5rem !important; }
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px; padding: 2.2rem 3rem; margin-bottom: 2rem;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 8px 30px rgba(59,130,246,0.15); border: 1px solid #1e3a5f;
    }
    .hero-left h1 { color: #e2e8f0; font-size: 2.5rem; font-weight: 800; margin: 0 0 0.3rem 0; }
    .hero-left p  { color: #94a3b8; font-size: 1rem; margin: 0 0 0.8rem 0; }
    .hero-right   { text-align: right; }
    .hero-right .team-name { color: #93c5fd; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .hero-right .team-label { color: #475569; font-size: 0.72rem; text-transform: uppercase; }
    .badge-pill {
        display: inline-block; background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.35); color: #60a5fa;
        font-size: 0.72rem; font-weight: 600; padding: 0.25rem 0.8rem;
        border-radius: 999px; margin-right: 0.4rem; text-transform: uppercase;
    }
    .stats-bar { display: flex; gap: 1rem; margin-bottom: 1.8rem; }
    .stat-card { flex: 1; background: #111827; border: 1px solid #1e293b; border-radius: 14px; padding: 1.1rem 1.4rem; text-align: center; }
    .stat-card .stat-value { font-size: 1.6rem; font-weight: 800; color: #60a5fa; display: block; }
    .stat-card .stat-label { font-size: 0.73rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }
    .section-label { color: #64748b; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.4rem; }
    .how-step { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
    .how-step .step-icon  { font-size: 1.8rem; }
    .how-step .step-title { font-size: 0.82rem; font-weight: 700; color: #e2e8f0; margin-top: 0.4rem; }
    .how-step .step-desc  { font-size: 0.74rem; color: #64748b; margin-top: 0.2rem; line-height: 1.4; }
    .reasoning-box {
        background: #0f172a; border: 1px solid #1e293b; border-left: 4px solid #6366f1;
        border-radius: 10px; padding: 1.2rem 1.5rem; color: #cbd5e1; font-size: 0.95rem; line-height: 1.75;
    }
    .section-divider { border-top: 1px solid #1e293b; margin: 1.8rem 0; }
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1, #818cf8); color: white;
        font-weight: 700; font-size: 1rem; border: none; border-radius: 12px;
        padding: 0.75rem 2.5rem; width: 100%;
        box-shadow: 0 4px 15px rgba(99,102,241,0.35);
    }
    div.stButton > button:hover { opacity: 0.88; }
    .footer { text-align: center; color: #475569; font-size: 0.78rem; margin-top: 3rem; padding: 1.2rem 0; border-top: 1px solid #1e293b; }
    .footer b { color: #60a5fa; }
    #MainMenu, footer { visibility: hidden; }
    .stTextArea textarea { background: #0f172a !important; border: 1.5px solid #1e293b !important; border-radius: 10px !important; color: #e2e8f0 !important; font-size: 0.92rem !important; }
    .stTextArea textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important; }
</style>
""", unsafe_allow_html=True)

DEFAULT_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://edueval-ai-njfc.onrender.com").rstrip("/")
BACKEND_URL = f"{DEFAULT_BASE_URL}/evaluate"
OCR_URL     = f"{DEFAULT_BASE_URL}/extract-text"

CATEGORY_CONFIG = {
    "correct":       {"icon": "✅", "label": "Correct"},
    "contradictory": {"icon": "⚠️", "label": "Contradictory"},
    "incorrect":     {"icon": "❌", "label": "Incorrect"},
}

for key, val in [("question_input", ""), ("ref_input", ""), ("student_input", ""), ("processed_files", {})]:
    if key not in st.session_state:
        st.session_state[key] = val

def extract_ocr_text(uploaded_file, field_name):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/png")}
        res = requests.post(OCR_URL, files=files, timeout=60)
        if res.status_code == 200:
            return res.json().get("extracted_text", "")
        else:
            st.error(f"OCR Error ({res.status_code}): {res.text}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Cannot reach OCR service: {exc}")
    return ""

def split_qa_text(raw_text):
    """
    Detects if raw_text contains both a Question and an Answer.
    Returns (question_part, answer_part) or (None, None).
    """
    if not raw_text or not raw_text.strip():
        return None, None
    patterns = [
        r'(?i)(?:[\r\n]+|[.?!]\s+|\s{2,}|\A)\s*(?:student[\'\s]*s?\s+answer|model\s+answer|reference\s+answer|answer|ans\b|solution|soln|sol\b)[\s.:\-]+',
        r'(?i)(?:[\r\n]+|[.?!]\s+)\s*(?:a|ans)\s*[\.:\-]+',
        r'(?i)(?:[\r\n]+)\s*(?:ans|a)\s+',
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, raw_text))
        if matches:
            chosen = None
            for m in matches:
                if m.start() > 2:
                    chosen = m
                    break
            if not chosen and matches and matches[0].start() > 0:
                chosen = matches[0]
            if chosen:
                split_idx = chosen.start()
                if raw_text[split_idx] in '.?!':
                    split_idx += 1
                q_part = re.sub(r'(?i)^(?:question|q|\d+[\.)])[\s.:\-0-9]*', '', raw_text[:split_idx].strip()).strip()
                a_part = re.sub(r'(?i)^(?:student[\'\s]*s?\s+answer|answer|ans|solution|soln|sol|a)[\s.:\-]+', '', raw_text[chosen.end():].strip()).strip()
                if len(q_part) >= 2 and len(a_part) >= 1:
                    return q_part, a_part
    return None, None

# HERO
st.markdown("""
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
        <div style="color:#334155;font-size:0.72rem;margin-top:0.5rem;">FastAPI · Streamlit · Google Gemini</div>
    </div>
</div>
""", unsafe_allow_html=True)

# STATS
st.markdown("""
<div class="stats-bar">
    <div class="stat-card"><span class="stat-value">91.65</span><div class="stat-label">Benchmark Score / 100</div></div>
    <div class="stat-card"><span class="stat-value">3-Way</span><div class="stat-label">Semantic Grading</div></div>
    <div class="stat-card"><span class="stat-value">CF1–CF4</span><div class="stat-label">Adversarial Defense</div></div>
    <div class="stat-card"><span class="stat-value">8,910</span><div class="stat-label">Training Samples</div></div>
    <div class="stat-card"><span class="stat-value">100%</span><div class="stat-label">Calibrated Confidence</div></div>
</div>
""", unsafe_allow_html=True)

# HOW IT WORKS
with st.expander("ℹ️ How it works", expanded=False):
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown('<div class="how-step"><div class="step-icon">📝</div><div class="step-title">Step 1: Input</div><div class="step-desc">Type your question & answer, or upload a handwritten sheet.</div></div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="how-step"><div class="step-icon">👁️</div><div class="step-title">Step 2: OCR</div><div class="step-desc">EasyOCR reads handwritten text and auto-fills the form.</div></div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="how-step"><div class="step-icon">🤖</div><div class="step-title">Step 3: AI Grades</div><div class="step-desc">Google Gemini evaluates and classifies the student answer.</div></div>', unsafe_allow_html=True)
    with h4:
        st.markdown('<div class="how-step"><div class="step-icon">📊</div><div class="step-title">Step 4: Results</div><div class="step-desc">Get Verdict, Confidence Score, and detailed AI Reasoning.</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# INPUTS
col_q, col_gap, col_s = st.columns([5, 0.2, 5])

with col_q:
    st.markdown('<p class="section-label">📌 Question</p>', unsafe_allow_html=True)
    q_img = st.file_uploader("📷 Upload question image (Auto-OCR)", type=["png","jpg","jpeg","webp"], key="q_img_file")
    if q_img is not None:
        file_sig = f"{q_img.name}_{len(q_img.getvalue())}"
        if st.session_state.processed_files.get("q_img") != file_sig:
            with st.spinner("🔍 Extracting via OCR..."):
                extracted = extract_ocr_text(q_img, "Question")
                if extracted:
                    q_split, a_split = split_qa_text(extracted)
                    st.session_state["question_input"] = q_split if q_split else extracted
                    if a_split and not st.session_state.get("student_input", "").strip():
                        st.session_state["student_input"] = a_split
                st.session_state.processed_files["q_img"] = file_sig
                st.rerun()
        with st.expander("🖼️ View image", expanded=False):
            st.image(q_img, use_container_width=True)

    question = st.text_area(label="q", placeholder="Type or upload question image above…", height=130, label_visibility="collapsed", key="question_input")

    detected_q, detected_a = split_qa_text(question)
    if detected_q and detected_a and not st.session_state.get("student_input", "").strip():
        def do_split(q_val, a_val):
            st.session_state["question_input"] = q_val
            st.session_state["student_input"]  = a_val
        c1, c2 = st.columns([6, 4])
        with c1: st.caption("✨ *Detected Question & Answer together!*")
        with c2: st.button("⚡ Split into boxes", key="btn_split", on_click=do_split, args=(detected_q, detected_a))

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">📖 Reference Answer <span style="color:#334155;font-weight:400;font-size:0.72rem">(optional)</span></p>', unsafe_allow_html=True)
    ref_img = st.file_uploader("📷 Upload reference answer image (Auto-OCR)", type=["png","jpg","jpeg","webp"], key="ref_img_file")
    if ref_img is not None:
        file_sig = f"{ref_img.name}_{len(ref_img.getvalue())}"
        if st.session_state.processed_files.get("ref_img") != file_sig:
            with st.spinner("🔍 Extracting via OCR..."):
                extracted = extract_ocr_text(ref_img, "Reference Answer")
                if extracted:
                    st.session_state["ref_input"] = extracted
                st.session_state.processed_files["ref_img"] = file_sig
                st.rerun()
        with st.expander("🖼️ View image", expanded=False):
            st.image(ref_img, use_container_width=True)

    reference_answer = st.text_area(label="ref", placeholder="Paste model answer or leave blank…", height=130, label_visibility="collapsed", key="ref_input")

with col_s:
    st.markdown('<p class="section-label">✏️ Student Answer</p>', unsafe_allow_html=True)
    stu_img = st.file_uploader("📷 Upload student answer image (Auto-OCR)", type=["png","jpg","jpeg","webp"], key="stu_img_file")
    if stu_img is not None:
        file_sig = f"{stu_img.name}_{len(stu_img.getvalue())}"
        if st.session_state.processed_files.get("stu_img") != file_sig:
            with st.spinner("🔍 Extracting via OCR..."):
                extracted = extract_ocr_text(stu_img, "Student Answer")
                if extracted:
                    st.session_state["student_input"] = extracted
                st.session_state.processed_files["stu_img"] = file_sig
                st.rerun()
        with st.expander("🖼️ View image", expanded=False):
            st.image(stu_img, use_container_width=True)

    student_answer = st.text_area(label="stu", placeholder="Type or upload student answer image…", height=330, label_visibility="collapsed", key="student_input")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# SUBMIT
_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    submitted = st.button("🚀 Evaluate Answer", use_container_width=True)

# EVALUATION
if submitted:
    if not student_answer.strip() and question.strip():
        auto_q, auto_a = split_qa_text(question)
        if auto_q and auto_a:
            question = auto_q
            student_answer = auto_a
            st.toast("Auto-split applied!", icon="🎯")

    if not question.strip():
        st.error("Please enter a Question.")
        st.stop()
    if not student_answer.strip():
        st.error("Please enter the Student Answer.")
        st.stop()

    payload = {"question": question.strip(), "student_answer": student_answer.strip()}
    if reference_answer.strip():
        payload["reference_answer"] = reference_answer.strip()

    with st.spinner("🤖 Evaluating with Google Gemini AI…"):
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach backend at `{DEFAULT_BASE_URL}`.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. Backend spinning up — retry in 30 seconds.")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"Backend error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    category      = data.get("category", "incorrect")
    probabilities = data.get("probabilities", {})
    reasoning     = data.get("reasoning", "No reasoning provided.")
    cfg           = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["incorrect"])
    confidence    = probabilities.get(category, 0.0)
    prob_correct       = probabilities.get("correct", 0.0)
    prob_contradictory = probabilities.get("contradictory", 0.0)
    prob_incorrect     = probabilities.get("incorrect", 0.0)

    BANNER = {
        "correct":       "linear-gradient(135deg,#16a34a,#22c55e)",
        "contradictory": "linear-gradient(135deg,#d97706,#f59e0b)",
        "incorrect":     "linear-gradient(135deg,#dc2626,#ef4444)",
    }
    grad = BANNER.get(category, BANNER["incorrect"])

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{grad};border-radius:20px;padding:2rem 2.5rem;text-align:center;
                margin-bottom:1.5rem;box-shadow:0 8px 30px rgba(0,0,0,0.3);">
        <div style="color:rgba(255,255,255,0.7);font-size:0.75rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem;">
            Evaluation Verdict
        </div>
        <div style="color:#fff;font-size:3.5rem;font-weight:900;line-height:1;margin-bottom:0.4rem;">
            {cfg['icon']} {cfg['label'].upper()}
        </div>
        <div style="color:rgba(255,255,255,0.85);font-size:1.1rem;font-weight:600;margin-bottom:1rem;">
            Confidence: {confidence * 100:.1f}%
        </div>
        <div style="background:rgba(0,0,0,0.2);border-radius:999px;height:12px;
                    width:70%;margin:0 auto 1.5rem auto;overflow:hidden;">
            <div style="background:#fff;height:100%;width:{confidence * 100:.1f}%;border-radius:999px;"></div>
        </div>
        <div style="display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;">
            <div style="background:rgba(0,0,0,0.2);border-radius:14px;padding:0.9rem 1.8rem;min-width:100px;">
                <div style="color:rgba(255,255,255,0.6);font-size:0.68rem;font-weight:700;text-transform:uppercase;">Correct</div>
                <div style="color:#fff;font-size:2rem;font-weight:900;">{prob_correct * 100:.1f}%</div>
            </div>
            <div style="background:rgba(0,0,0,0.2);border-radius:14px;padding:0.9rem 1.8rem;min-width:100px;">
                <div style="color:rgba(255,255,255,0.6);font-size:0.68rem;font-weight:700;text-transform:uppercase;">Contradictory</div>
                <div style="color:#fff;font-size:2rem;font-weight:900;">{prob_contradictory * 100:.1f}%</div>
            </div>
            <div style="background:rgba(0,0,0,0.2);border-radius:14px;padding:0.9rem 1.8rem;min-width:100px;">
                <div style="color:rgba(255,255,255,0.6);font-size:0.68rem;font-weight:700;text-transform:uppercase;">Incorrect</div>
                <div style="color:#fff;font-size:2rem;font-weight:900;">{prob_incorrect * 100:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">💬 AI Reasoning & Explanation</p>', unsafe_allow_html=True)
    with st.expander("View detailed AI reasoning", expanded=True):
        st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

    with st.expander("🛠️ Raw API Response (debug)", expanded=False):
        st.json(data)

# FOOTER
st.markdown("""
<div class="footer">
    <b>EduEval AI</b> &nbsp;·&nbsp; Team <b>Asynchronous</b> &nbsp;·&nbsp;
    ED-05 Hackathon &nbsp;·&nbsp; Powered by <b>FastAPI</b> + <b>Streamlit</b> + <b>Google Gemini</b><br>
    <span style="color:#334155;font-size:0.72rem;">
        Benchmark Score: 91.65/100 &nbsp;|&nbsp; SemEval-2013 Task 7 &nbsp;|&nbsp; CF1–CF4 Adversarial Defense
    </span>
</div>
""", unsafe_allow_html=True)
