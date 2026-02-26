import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import PyPDF2
import re
import requests
import base64
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. CORE CONFIG & NLP SETUP ---
@st.cache_resource
def setup_nlp():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except:
        pass

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

setup_nlp()
lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_prev8pbt.json")

# ✅ NEW ANIMATIONS ADDED (No changes to existing)
lottie_robot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_kyu7xb1v.json")
lottie_scan = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_gjmecwii.json")

# --- 2. LOGIC FUNCTIONS ---
def get_pdf_text(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    try:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in stop_words and len(w) > 2]
    except:
        return text.split()

def get_pdf_display(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    return f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'

# --- 3. UI DESIGN ---
st.set_page_config(page_title="Nexus AI Pro", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    @keyframes moveText {
        0% { transform: translateY(0px); text-shadow: 0 0 5px #f59e0b; }
        50% { transform: translateY(-5px); text-shadow: 0 0 20px #fbbf24; }
        100% { transform: translateY(0px); text-shadow: 0 0 5px #f59e0b; }
    }
    .stApp { background-color: #f1f5f9; }
    .executive-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 40px; border-radius: 0 0 30px 30px;
        text-align: center; border-bottom: 5px solid #f59e0b;
        margin: -60px -100px 30px -100px;
    }
    .main-title { 
        color: #f59e0b !important; 
        font-weight: 900; 
        font-size: 3rem !important;
        animation: moveText 3s ease-in-out infinite;
    }
    .moving-login-title {
        text-align: center; 
        color: #0f172a;
        animation: moveText 4s ease-in-out infinite;
    }
    .metric-card {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #f59e0b, #d97706) !important;
        color: white !important; border-radius: 12px !important;
        height: 50px; font-weight: bold; width: 100%; border: none;
    }
    .login-container {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-top: 5px solid #f59e0b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- LOGIN PAGE ---
def show_login():
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("<h2 class='moving-login-title'>System Login</h2>", unsafe_allow_html=True)
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        if st.button("LOGIN TO NEXUS"):
            if email and password:
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Please enter credentials")
        st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APP ---
def show_main_app():
    st.markdown("""
        <div class="executive-header">
            <h1 class="main-title">NEXUS AI PRO</h1>
            <p style="color: #94a3b8; font-size: 1.2rem;">Advanced Semantic Recruitment Engine</p>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        if lottie_ai:
            st_lottie(lottie_ai, height=150)

        # ✅ Added animations
        if lottie_robot:
            st_lottie(lottie_robot, height=120)
        if lottie_scan:
            st_lottie(lottie_scan, height=120)

        st.markdown("### ⚙️ Engine Settings")
        st.info("Algorithm: TF-IDF + Cosine Similarity")
        st.divider()
        threshold = st.slider("Success Threshold %", 0, 100, 75)
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📂 Resume Upload")
        resume_file = st.file_uploader("Upload PDF", type=['pdf'])
    with col2:
        st.markdown("### 📝 Job Description")
        job_desc = st.text_area("Paste JD here", height=150)

    if st.button("🔍 START DEEP NEURAL SCAN"):
        if resume_file and job_desc:
            raw_resume = get_pdf_text(resume_file)
            res_words = clean_text(raw_resume)
            job_words = clean_text(job_desc)

            if not raw_resume.strip():
                st.error("Could not read PDF text.")
            else:
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([" ".join(res_words), " ".join(job_words)])
                score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

                # ✅ ATS SCORE ADDED
                common = set(res_words) & set(job_words)
                ats_score = min(100, score + (len(common) * 0.5))

                missing = [w for w, c in Counter(job_words).most_common(15) if w not in set(res_words)]

                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='metric-card'><h4>MATCH</h4><h1 style='color:#f59e0b'>{score:.1f}%</h1></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><h4>KEYWORDS</h4><h1 style='color:#10b981'>{len(common)}</h1></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><h4>MISSING</h4><h1 style='color:#ef4444'>{len(missing)}</h1></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><h4>ATS SCORE</h4><h1 style='color:#3b82f6'>{ats_score:.1f}%</h1></div>", unsafe_allow_html=True)

                tab_chart, tab_gaps, tab_pdf, tab_break = st.tabs(
                    ["📊 Skill Map", "💡 Recommendations", "📄 Document Viewer", "🧠 Skill Breakdown"]
                )

                with tab_chart:
                    top_job_counts = Counter(job_words).most_common(8)
                    labels = [x[0] for x in top_job_counts]
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=[x[1] for x in top_job_counts],
                        theta=labels,
                        fill='toself',
                        name='Job Required'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=[Counter(res_words).get(x, 0) for x in labels],
                        theta=labels,
                        fill='toself',
                        name='Your Profile'
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                with tab_gaps:
                    for m in missing[:10]:
                        st.warning(f"Add keyword: **{m.upper()}**")
                    if score >= threshold:
                        st.success("✅ Ready for Submission!")
                    else:
                        st.error("❌ Profile Optimization Required.")

                with tab_pdf:
                    resume_file.seek(0)
                    st.markdown(get_pdf_display(resume_file), unsafe_allow_html=True)

                # ✅ NEW SKILL BREAKDOWN TAB
                with tab_break:
                    tech_keywords = ["python","java","sql","machine","learning","ai","data","analysis","cloud","aws"]
                    soft_keywords = ["communication","leadership","team","management","problem","solving"]

                    tech_count = len([w for w in res_words if w in tech_keywords])
                    soft_count = len([w for w in res_words if w in soft_keywords])

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=["Technical Skills"], y=[tech_count]))
                    fig2.add_trace(go.Bar(x=["Soft Skills"], y=[soft_count]))
                    st.plotly_chart(fig2, use_container_width=True)

                # ✅ DOWNLOAD REPORT FEATURE
                report = f"""
NEXUS AI PRO REPORT

Match Score: {score:.2f}%
ATS Score: {ats_score:.2f}%
Common Keywords: {len(common)}
Missing Keywords: {len(missing)}

Top Missing Skills:
{', '.join(missing[:10])}
"""
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report,
                    file_name="nexus_ai_report.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please upload a file and paste a job description.")

# --- ROUTING ---
if not st.session_state['logged_in']:
    show_login()
else:
    show_main_app()