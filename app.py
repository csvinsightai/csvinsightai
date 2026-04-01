import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Data Analysis & ML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #F0F4FA;
    color: #1A2340;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1C3F 0%, #1A2F5E 50%, #0D1B38 100%);
    border-right: 1px solid rgba(99,130,255,0.15);
    box-shadow: 4px 0 24px rgba(0,0,0,0.25);
}
[data-testid="stSidebar"] * { color: #C8D6F0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #8FA8D8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    text-align: left !important;
    padding: 0.65rem 1rem !important;
    border-radius: 10px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,130,255,0.15) !important;
    color: #FFFFFF !important;
    transform: translateX(3px) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Main area ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

/* ── Top header ── */
.top-header {
    background: linear-gradient(135deg, #1A3A8F 0%, #2D5BE3 50%, #4A7CFF 100%);
    border-radius: 16px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(45,91,227,0.3);
}
.top-header h1 {
    color: white !important;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.top-header .badge {
    background: rgba(255,255,255,0.18);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    border: 1px solid rgba(255,255,255,0.25);
}
.avatar {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #A78BFA, #60A5FA);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; color: white;
    border: 2px solid rgba(255,255,255,0.4);
}

/* ── Section heading ── */
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0F1C3F;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.section-subtitle {
    font-size: 0.875rem;
    color: #64748B;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* ── Cards ── */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(226,232,240,0.8);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(45,91,227,0.12);
}
.metric-card .label {
    font-size: 0.75rem;
    color: #94A3B8;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0F1C3F;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* ── Upload area ── */
.upload-zone {
    background: white;
    border: 2.5px dashed #CBD5E1;
    border-radius: 20px;
    padding: 3.5rem 2rem;
    text-align: center;
    transition: all 0.25s ease;
    cursor: pointer;
}
.upload-zone:hover {
    border-color: #3B82F6;
    background: #EFF6FF;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.08);
}
.upload-icon { font-size: 3rem; margin-bottom: 1rem; }
.upload-title { font-size: 1.1rem; font-weight: 700; color: #1E293B; }
.upload-sub { font-size: 0.82rem; color: #94A3B8; margin-top: 0.4rem; }

/* ── Status badges ── */
.badge-success {
    background: #DCFCE7; color: #15803D;
    padding: 0.25rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
    border: 1px solid #86EFAC; display: inline-block;
}
.badge-warning {
    background: #FEF3C7; color: #B45309;
    padding: 0.25rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
    border: 1px solid #FCD34D; display: inline-block;
}
.badge-error {
    background: #FEE2E2; color: #B91C1C;
    padding: 0.25rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
    border: 1px solid #FCA5A5; display: inline-block;
}

/* ── Chart containers ── */
.chart-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(226,232,240,0.8);
    margin-bottom: 1.2rem;
}
.chart-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #F1F5F9;
}

/* ── Best model highlight ── */
.best-model-row {
    background: linear-gradient(90deg, #EFF6FF, #DBEAFE) !important;
    border-left: 4px solid #3B82F6 !important;
    font-weight: 600 !important;
}

/* ── Primary button override ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2D5BE3, #4A7CFF) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(45,91,227,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(45,91,227,0.45) !important;
}

/* ── DataFrame styling ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #E2E8F0 !important;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #CBD5E1, transparent);
    margin: 2rem 0;
}

/* ── Info box ── */
.info-box {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 1px solid #BFDBFE;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #1E40AF;
}

/* ── Logo area ── */
.logo-area {
    padding: 1.5rem 1.2rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.logo-text {
    font-size: 1rem;
    font-weight: 800;
    color: white !important;
    letter-spacing: -0.01em;
}
.logo-sub {
    font-size: 0.7rem;
    color: #6B85B8 !important;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Spinner ── */
@keyframes spin { to { transform: rotate(360deg); } }
.loader {
    width: 36px; height: 36px;
    border: 3px solid #E2E8F0;
    border-top-color: #3B82F6;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 1rem auto;
}

/* ── Insight panel ── */
.insight-card {
    background: linear-gradient(135deg, #1E3A8A, #1D4ED8);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    color: white;
    box-shadow: 0 8px 24px rgba(30,58,138,0.35);
}
.insight-card h3 { font-size: 0.95rem; font-weight: 700; margin-bottom: 0.8rem; opacity: 0.9; }
.insight-card p { font-size: 0.82rem; opacity: 0.8; line-height: 1.6; }

/* ── Step indicator (guide) ── */
.step-box {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #3B82F6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.step-num {
    font-size: 0.7rem; font-weight: 700;
    color: #3B82F6; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 0.2rem;
}
.step-text { font-size: 0.85rem; color: #334155; line-height: 1.5; }
.step-code {
    font-family: 'JetBrains Mono', monospace;
    background: #0F172A; color: #7DD3FC;
    border-radius: 8px; padding: 0.6rem 1rem;
    font-size: 0.8rem; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state init ──────────────────────────────────────────────────────
defaults = {
    "page": "Upload Data",
    "df": None,
    "df_clean": None,
    "trained_models": {},
    "model_metrics": {},
    "target_col": None,
    "le_map": {},
    "feature_cols": [],
    "best_model_name": None,
    "X_test": None,
    "y_test": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div style="font-size:1.8rem; margin-bottom:0.4rem">🤖</div>
        <div class="logo-text">DataML Studio</div>
        <div class="logo-sub">AI Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("📁", "Upload Data"),
        ("📊", "Data Overview"),
        ("🧹", "Data Cleaning"),
        ("📈", "EDA Visualization"),
        ("🧠", "Model Training"),
        ("🏆", "Results Dashboard"),
    ]

    st.markdown("<div style='padding: 0.5rem 0.8rem; font-size:0.65rem; font-weight:700; color:#475569; letter-spacing:0.1em; text-transform:uppercase;'>NAVIGATION</div>", unsafe_allow_html=True)

    for icon, label in nav_items:
        active = st.session_state.page == label
        btn_style = ""
        if active:
            st.markdown(f"""
            <div style='background:rgba(99,130,255,0.2); border-radius:10px; padding:0.65rem 1rem;
                        margin:0.15rem 0; border-left:3px solid #6382FF; cursor:default;'>
                <span style='font-size:0.88rem; font-weight:600; color:#FFFFFF;'>{icon} &nbsp; {label}</span>
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{label}"):
                st.session_state.page = label
                st.rerun()

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Data status
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"""
        <div style='padding:0.8rem 1rem; background:rgba(34,197,94,0.1);
                    border-radius:10px; border:1px solid rgba(34,197,94,0.2);'>
            <div style='font-size:0.7rem; color:#4ADE80; font-weight:700; letter-spacing:0.05em;'>✓ DATA LOADED</div>
            <div style='font-size:0.8rem; color:#CBD5E1; margin-top:0.2rem;'>{df.shape[0]:,} rows × {df.shape[1]} cols</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding:0.8rem 1rem; background:rgba(148,163,184,0.1);
                    border-radius:10px; border:1px solid rgba(148,163,184,0.2);'>
            <div style='font-size:0.7rem; color:#94A3B8; font-weight:700;'>○ NO DATA LOADED</div>
            <div style='font-size:0.8rem; color:#64748B; margin-top:0.2rem;'>Upload a CSV to begin</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem; color:#334155; text-align:center; padding:0 1rem;'>DataML Studio v2.0 · Built with Streamlit</div>", unsafe_allow_html=True)


# ─── Top Header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
    <div>
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.65); font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.2rem;">
            AI-Powered Data Analysis & Machine Learning System
        </div>
        <div style="font-size:1.25rem; font-weight:800; color:white; letter-spacing:-0.01em;">
            {st.session_state.page}
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:1rem;">
        <div class="badge">PRO PLAN</div>
        <div class="avatar">DS</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATA
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Upload Data":
    st.markdown('<div class="section-title">Upload Your Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Drag & drop or browse to upload a CSV file to get started with your analysis.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">☁️</div>
            <div class="upload-title">Drop your CSV file here</div>
            <div class="upload-sub">Supported format: .csv · Max size: 200 MB</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

        if uploaded_file:
            with st.spinner("🔄 Parsing your dataset..."):
                time.sleep(0.5)
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.df_clean = df.copy()
                    st.markdown(f"""
                    <div style="background:#DCFCE7; border:1px solid #86EFAC; border-radius:14px;
                                padding:1rem 1.4rem; margin-top:1rem; display:flex; align-items:center; gap:0.8rem;">
                        <span style="font-size:1.5rem;">✅</span>
                        <div>
                            <div style="font-weight:700; color:#15803D; font-size:0.9rem;">Upload Successful!</div>
                            <div style="color:#166534; font-size:0.8rem; margin-top:0.1rem;">
                                Loaded <strong>{df.shape[0]:,} rows</strong> and <strong>{df.shape[1]} columns</strong> from <em>{uploaded_file.name}</em>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                except Exception as e:
                    st.markdown(f"""
                    <div style="background:#FEE2E2; border:1px solid #FCA5A5; border-radius:14px;
                                padding:1rem 1.4rem; margin-top:1rem;">
                        <div style="font-weight:700; color:#B91C1C;">❌ Upload Failed</div>
                        <div style="color:#991B1B; font-size:0.82rem;">{str(e)}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="height:100%;">
            <div class="label">QUICK TIPS</div>
            <div style="margin-top:0.8rem;">
        """, unsafe_allow_html=True)

        tips = [
            ("📋", "CSV must have a header row"),
            ("🔢", "Mix of numeric & categorical columns works best"),
            ("🎯", "Ensure your target column is included"),
            ("🚫", "Avoid special characters in column names"),
            ("📏", "Datasets up to 1M rows supported"),
        ]
        for icon, tip in tips:
            st.markdown(f"""
            <div style="display:flex; gap:0.6rem; align-items:flex-start; margin-bottom:0.8rem;">
                <span style="font-size:1rem;">{icon}</span>
                <span style="font-size:0.8rem; color:#475569; line-height:1.4;">{tip}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Demo dataset button
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>🧪 No dataset?</strong><br>
            Click below to load the built-in Titanic demo dataset and explore all features instantly.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Load Demo Dataset (Titanic)", use_container_width=True):
            from sklearn.datasets import fetch_openml
            with st.spinner("Loading demo data..."):
                try:
                    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                    import urllib.request
                    demo_df = pd.read_csv(url)
                    st.session_state.df = demo_df
                    st.session_state.df_clean = demo_df.copy()
                    st.success(f"✅ Titanic dataset loaded: {demo_df.shape[0]} rows × {demo_df.shape[1]} cols")
                    st.rerun()
                except:
                    # fallback synthetic dataset
                    np.random.seed(42)
                    n = 300
                    demo_df = pd.DataFrame({
                        "age": np.random.randint(18, 80, n),
                        "income": np.random.randint(20000, 150000, n),
                        "education_years": np.random.randint(8, 22, n),
                        "gender": np.random.choice(["Male","Female"], n),
                        "department": np.random.choice(["HR","Tech","Sales","Marketing"], n),
                        "experience": np.random.randint(0, 40, n),
                        "satisfaction": np.random.uniform(1, 5, n).round(2),
                        "promoted": np.random.choice([0, 1], n, p=[0.7, 0.3]),
                    })
                    demo_df.loc[np.random.choice(n, 20, replace=False), "income"] = np.nan
                    demo_df.loc[np.random.choice(n, 10, replace=False), "age"] = np.nan
                    st.session_state.df = demo_df
                    st.session_state.df_clean = demo_df.copy()
                    st.success(f"✅ Demo dataset loaded: {demo_df.shape[0]} rows × {demo_df.shape[1]} cols")
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DATA OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Data Overview":
    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a CSV file first.")
        st.stop()

    df = st.session_state.df
    st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">A comprehensive snapshot of your dataset structure and contents.</div>', unsafe_allow_html=True)

    # Metric cards
    total_missing = df.isnull().sum().sum()
    pct_missing = (total_missing / (df.shape[0] * df.shape[1]) * 100)
    num_cols = df.select_dtypes(include=np.number).shape[1]
    cat_cols = df.select_dtypes(include="object").shape[1]

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "TOTAL ROWS", f"{df.shape[0]:,}", "Records", "📋"),
        (c2, "TOTAL COLUMNS", f"{df.shape[1]}", "Features", "📐"),
        (c3, "NUMERIC COLS", f"{num_cols}", "Quantitative", "🔢"),
        (c4, "CATEGORICAL", f"{cat_cols}", "Qualitative", "🏷️"),
        (c5, "MISSING VALUES", f"{total_missing:,}", f"{pct_missing:.1f}% of total", "⚠️"),
    ]
    for col, label, val, sub, icon in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.3rem;">
                <span style="font-size:1.6rem;">{icon}</span>
                <span class="value">{val}</span>
            </div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Table preview
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📋 Dataset Preview — First 10 Rows</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🗂️ Column Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📊 Statistical Summary</div>', unsafe_allow_html=True)
        num_df = df.describe().round(3)
        st.dataframe(num_df, use_container_width=True, height=280)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Data Cleaning":
    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a CSV file first.")
        st.stop()

    df = st.session_state.df.copy()
    st.markdown('<div class="section-title">Data Cleaning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Identify and handle missing values, duplicates, and data quality issues.</div>', unsafe_allow_html=True)

    # Missing values analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
    }).sort_values("Missing %", ascending=False)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🔍 Missing Value Analysis</div>', unsafe_allow_html=True)

    if missing.sum() == 0:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#15803D; font-weight:600; font-size:1rem;">
            ✅ No missing values found! Your dataset is clean.
        </div>""", unsafe_allow_html=True)
    else:
        for _, row in missing_df.iterrows():
            if row["Missing %"] == 0:
                badge = f'<span class="badge-success">✓ Clean</span>'
            elif row["Missing %"] < 10:
                badge = f'<span class="badge-warning">⚠ {row["Missing %"]}% missing</span>'
            else:
                badge = f'<span class="badge-error">✗ {row["Missing %"]}% missing</span>'

            bar_color = "#22C55E" if row["Missing %"] == 0 else ("#F59E0B" if row["Missing %"] < 10 else "#EF4444")
            bar_w = min(row["Missing %"], 100)

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; padding:0.6rem 0;
                        border-bottom:1px solid #F1F5F9;">
                <div style="width:160px; font-size:0.83rem; font-weight:600; color:#334155; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    {row["Column"]}
                </div>
                <div style="flex:1; background:#F1F5F9; border-radius:4px; height:8px; overflow:hidden;">
                    <div style="width:{bar_w}%; background:{bar_color}; height:100%; border-radius:4px;"></div>
                </div>
                <div style="width:80px; font-size:0.8rem; color:#64748B; text-align:right;">
                    {int(row["Missing Count"])} rows
                </div>
                <div style="width:160px; text-align:right;">{badge}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Cleaning actions
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🛠️ Cleaning Actions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        num_strategy = st.selectbox("Numeric missing values strategy",
            ["Fill with Median", "Fill with Mean", "Fill with 0", "Drop rows"])
    with col2:
        cat_strategy = st.selectbox("Categorical missing values strategy",
            ["Fill with Mode", "Fill with 'Unknown'", "Drop rows"])

    drop_dups = st.checkbox("🗑️ Drop duplicate rows", value=True)

    if st.button("⚡ Apply Cleaning", type="primary"):
        df_c = st.session_state.df.copy()
        num_cols_list = df_c.select_dtypes(include=np.number).columns.tolist()
        cat_cols_list = df_c.select_dtypes(include="object").columns.tolist()

        if num_strategy == "Fill with Median":
            df_c[num_cols_list] = df_c[num_cols_list].fillna(df_c[num_cols_list].median())
        elif num_strategy == "Fill with Mean":
            df_c[num_cols_list] = df_c[num_cols_list].fillna(df_c[num_cols_list].mean())
        elif num_strategy == "Fill with 0":
            df_c[num_cols_list] = df_c[num_cols_list].fillna(0)
        elif num_strategy == "Drop rows":
            df_c = df_c.dropna(subset=num_cols_list)

        if cat_strategy == "Fill with Mode":
            for c in cat_cols_list:
                df_c[c] = df_c[c].fillna(df_c[c].mode()[0] if not df_c[c].mode().empty else "Unknown")
        elif cat_strategy == "Fill with 'Unknown'":
            df_c[cat_cols_list] = df_c[cat_cols_list].fillna("Unknown")
        elif cat_strategy == "Drop rows":
            df_c = df_c.dropna(subset=cat_cols_list)

        if drop_dups:
            before = len(df_c)
            df_c = df_c.drop_duplicates()
            removed = before - len(df_c)

        st.session_state.df_clean = df_c

        st.markdown(f"""
        <div style="background:#DCFCE7; border:1px solid #86EFAC; border-radius:12px; padding:1rem 1.4rem; margin-top:1rem;">
            <div style="font-weight:700; color:#15803D; font-size:0.95rem;">✅ Cleaning Complete!</div>
            <div style="color:#166534; font-size:0.82rem; margin-top:0.3rem;">
                Dataset cleaned: <strong>{df_c.shape[0]:,} rows × {df_c.shape[1]} cols</strong> remaining.
                Missing values: <strong>{df_c.isnull().sum().sum()}</strong>
                {f"· Duplicates removed: <strong>{removed}</strong>" if drop_dups else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Duplicate count
    dups = df.duplicated().sum()
    st.markdown(f"""
    <div class="info-box" style="margin-top:1rem;">
        🔁 <strong>Duplicate rows detected:</strong> {dups} rows ({dups/len(df)*100:.2f}% of total)
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EDA VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "EDA Visualization":
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Visualize distributions, relationships, and patterns in your data.</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if not num_cols:
        st.warning("No numeric columns found for visualization.")
        st.stop()

    # Column selector row
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        hist_col = st.selectbox("📊 Select column for Histogram", num_cols)
    with col_sel2:
        box_col = st.selectbox("📦 Select column for Boxplot", num_cols, index=min(1, len(num_cols)-1))

    # Row 1: Histogram + Boxplot
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-title">📊 Distribution — {hist_col}</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x=hist_col, nbins=30,
            color_discrete_sequence=["#3B82F6"],
            template="plotly_white"
        )
        fig.update_layout(
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans"),
            bargap=0.05,
        )
        fig.update_traces(marker_line_color="#2D5BE3", marker_line_width=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-title">📦 Box Plot — {box_col}</div>', unsafe_allow_html=True)
        if cat_cols:
            group_col = st.selectbox("Group by", ["None"] + cat_cols, key="box_group")
            fig = px.box(
                df, y=box_col,
                x=group_col if group_col != "None" else None,
                color=group_col if group_col != "None" else None,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
        else:
            fig = px.box(df, y=box_col, template="plotly_white",
                         color_discrete_sequence=["#6366F1"])
        fig.update_layout(
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans"), showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Correlation Heatmap + Scatter
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🌡️ Correlation Heatmap</div>', unsafe_allow_html=True)
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, template="plotly_white"
            )
            fig.update_layout(
                margin=dict(l=0,r=0,t=10,b=0), height=300,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", size=10),
                coloraxis_colorbar=dict(thickness=10, len=0.8)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation.")
        st.markdown("</div>", unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🔵 Scatter Plot</div>', unsafe_allow_html=True)
        if len(num_cols) >= 2:
            sc_x = st.selectbox("X axis", num_cols, key="sc_x")
            sc_y = st.selectbox("Y axis", num_cols, index=1, key="sc_y")
            color_col = st.selectbox("Color by", ["None"] + cat_cols, key="sc_color")
            fig = px.scatter(
                df, x=sc_x, y=sc_y,
                color=color_col if color_col != "None" else None,
                template="plotly_white", opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(
                margin=dict(l=0,r=0,t=10,b=0), height=260,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Category distribution
    if cat_cols:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🏷️ Categorical Column Distribution</div>', unsafe_allow_html=True)
        cat_sel = st.selectbox("Select categorical column", cat_cols, key="cat_dist")
        vc = df[cat_sel].value_counts().reset_index()
        vc.columns = [cat_sel, "Count"]
        fig = px.bar(vc, x=cat_sel, y="Count", template="plotly_white",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(
            margin=dict(l=0,r=0,t=10,b=0), height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans"), coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Model Training":
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    st.markdown('<div class="section-title">Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Configure and train multiple machine learning models to find the best fit.</div>', unsafe_allow_html=True)

    MODEL_MAP = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">⚙️ Training Configuration</div>', unsafe_allow_html=True)

        target_col = st.selectbox("🎯 Select Target Column", df.columns.tolist())
        st.session_state.target_col = target_col

        selected_models = st.multiselect(
            "🤖 Select Models to Train",
            list(MODEL_MAP.keys()),
            default=["Logistic Regression", "Random Forest", "Decision Tree"]
        )

        test_size = st.slider("📐 Test Set Size", 0.1, 0.4, 0.2, 0.05,
                              help="Fraction of data used for testing")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📋 Data Preparation Preview</div>', unsafe_allow_html=True)

        feature_cols = [c for c in df.columns if c != target_col]
        st.markdown(f"""
        <div style="margin-bottom:0.8rem;">
            <span class="badge-success">✓ Features</span>
            <span style="font-size:0.82rem; color:#475569; margin-left:0.5rem;">{len(feature_cols)} columns will be used</span>
        </div>
        <div style="margin-bottom:0.8rem;">
            <span class="badge-warning">⊕ Target</span>
            <span style="font-size:0.82rem; color:#475569; margin-left:0.5rem;"><strong>{target_col}</strong> · {df[target_col].nunique()} unique classes</span>
        </div>
        <div>
            <span class="badge-error">⊗ Test split</span>
            <span style="font-size:0.82rem; color:#475569; margin-left:0.5rem;">{int(test_size*100)}% of {len(df):,} rows</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        classes = df[target_col].value_counts()
        fig = px.pie(values=classes.values, names=classes.index,
                     hole=0.55, template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=150,
                          paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=True, legend=dict(font=dict(size=10)))
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model.")
        else:
            with st.spinner(""):
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Prepare data
                df_model = df.copy().dropna(subset=[target_col])
                X = df_model[feature_cols].copy()
                y = df_model[target_col].copy()

                # Encode categoricals
                le_map = {}
                for c in X.select_dtypes(include="object").columns:
                    le = LabelEncoder()
                    X[c] = le.fit_transform(X[c].astype(str))
                    le_map[c] = le

                if y.dtype == "object":
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y.astype(str))

                # Fill remaining NaN
                X = X.fillna(X.median(numeric_only=True))

                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.feature_cols = feature_cols
                st.session_state.le_map = le_map

                metrics = {}
                trained = {}

                for i, name in enumerate(selected_models):
                    status_text.markdown(f"""
                    <div style="text-align:center; padding:1rem;">
                        <div class="loader"></div>
                        <div style="font-size:0.9rem; font-weight:600; color:#1E293B; margin-top:0.5rem;">
                            Training <span style="color:#3B82F6;">{name}</span>...
                        </div>
                        <div style="font-size:0.78rem; color:#64748B; margin-top:0.3rem;">
                            Model {i+1} of {len(selected_models)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress((i) / len(selected_models))

                    clf = MODEL_MAP[name]
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") and len(np.unique(y)) == 2 else None

                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    roc_auc = None
                    if y_prob is not None:
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        metrics[name] = {
                            "Accuracy": round(acc, 4),
                            "Precision": round(prec, 4),
                            "Recall": round(rec, 4),
                            "F1-Score": round(f1, 4),
                            "ROC-AUC": round(roc_auc, 4),
                            "_fpr": fpr, "_tpr": tpr,
                        }
                    else:
                        metrics[name] = {
                            "Accuracy": round(acc, 4),
                            "Precision": round(prec, 4),
                            "Recall": round(rec, 4),
                            "F1-Score": round(f1, 4),
                            "ROC-AUC": "N/A",
                        }

                    metrics[name]["_cm"] = confusion_matrix(y_test, y_pred)
                    metrics[name]["_y_pred"] = y_pred

                    trained[name] = clf
                    time.sleep(0.3)

                progress_bar.progress(1.0)
                status_text.empty()
                progress_placeholder.empty()

                st.session_state.trained_models = trained
                st.session_state.model_metrics = metrics

                best = max(metrics, key=lambda x: metrics[x]["Accuracy"])
                st.session_state.best_model_name = best

                st.success(f"✅ Training complete! **{len(selected_models)} models** trained successfully. Best model: **{best}** ({metrics[best]['Accuracy']*100:.2f}% accuracy)")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Results Dashboard":
    if not st.session_state.model_metrics:
        st.warning("⚠️ No trained models yet. Please train models first.")
        st.stop()

    metrics = st.session_state.model_metrics
    best = st.session_state.best_model_name

    st.markdown('<div class="section-title">Results Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Compare model performance, analyze results, and extract key insights.</div>', unsafe_allow_html=True)

    # Model performance table
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)

    table_data = []
    for name, m in metrics.items():
        table_data.append({
            "Model": ("⭐ " if name == best else "") + name,
            "Accuracy": f"{m['Accuracy']*100:.2f}%",
            "Precision": f"{m['Precision']*100:.2f}%",
            "Recall": f"{m['Recall']*100:.2f}%",
            "F1-Score": f"{m['F1-Score']*100:.2f}%",
            "ROC-AUC": f"{m['ROC-AUC']*100:.2f}%" if m['ROC-AUC'] != "N/A" else "N/A",
        })

    table_df = pd.DataFrame(table_data)
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 Accuracy Comparison</div>', unsafe_allow_html=True)
    model_names = list(metrics.keys())
    accuracies = [metrics[n]["Accuracy"] * 100 for n in model_names]
    colors = ["#3B82F6" if n == best else "#CBD5E1" for n in model_names]

    fig = go.Figure(go.Bar(
        x=model_names, y=accuracies,
        marker_color=colors,
        text=[f"{a:.1f}%" for a in accuracies],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white", height=280,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans"),
        yaxis=dict(range=[0, max(accuracies) * 1.15], showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Confusion Matrix + ROC Curve
    best_cm = metrics[best]["_cm"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sel_model = st.selectbox("Select model", list(metrics.keys()), key="cm_sel")
        st.markdown(f'<div class="chart-title">🔢 Confusion Matrix — {sel_model}</div>', unsafe_allow_html=True)
        cm = metrics[sel_model]["_cm"]
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            color_continuous_scale="Blues", template="plotly_white",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig.update_layout(
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 ROC Curves</div>', unsafe_allow_html=True)
        fig = go.Figure()
        colors_roc = ["#3B82F6","#22C55E","#F59E0B","#EF4444","#8B5CF6","#EC4899"]
        has_roc = False
        for i, (name, m) in enumerate(metrics.items()):
            if "_fpr" in m:
                fig.add_trace(go.Scatter(
                    x=m["_fpr"], y=m["_tpr"],
                    name=f"{name} (AUC={m['ROC-AUC']:.3f})",
                    line=dict(color=colors_roc[i % len(colors_roc)], width=2.5),
                    mode="lines"
                ))
                has_roc = True
        if has_roc:
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="#CBD5E1", width=1),
                                     showlegend=False))
            fig.update_layout(
                template="plotly_white", height=280,
                margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor="#F1F5F9"),
                legend=dict(font=dict(size=10)),
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curves available for binary classification only.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature importance
    if best in st.session_state.trained_models:
        model_obj = st.session_state.trained_models[best]
        if hasattr(model_obj, "feature_importances_"):
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-title">🌟 Feature Importance — {best}</div>', unsafe_allow_html=True)
            importances = model_obj.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": st.session_state.feature_cols,
                "Importance": importances
            }).sort_values("Importance", ascending=True).tail(15)

            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues",
                         template="plotly_white")
            fig.update_layout(
                margin=dict(l=0,r=0,t=10,b=0), height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans"),
                yaxis=dict(showgrid=False), xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Insight Panel
    best_acc = metrics[best]["Accuracy"] * 100
    all_accs = [metrics[n]["Accuracy"] * 100 for n in metrics]
    avg_acc = np.mean(all_accs)

    st.markdown(f"""
    <div class="insight-card">
        <h3>🧠 Summary Insights</h3>
        <p>
            <strong>{best}</strong> achieved the highest accuracy of <strong>{best_acc:.2f}%</strong>
            across {len(metrics)} trained models (avg: {avg_acc:.2f}%).
            The model demonstrates {("strong" if best_acc > 85 else "moderate" if best_acc > 70 else "baseline")} predictive performance
            on the test set ({int(st.session_state.y_test.shape[0])} samples).
            {'Consider feature engineering or hyperparameter tuning to further improve results.' if best_acc < 85 else 'Results look strong — consider deploying this model.'}
        </p>
        <br>
        <p>
            📌 Target: <strong>{st.session_state.target_col}</strong> &nbsp;·&nbsp;
            🔢 Features used: <strong>{len(st.session_state.feature_cols)}</strong> &nbsp;·&nbsp;
            🏅 Best model: <strong>{best}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
