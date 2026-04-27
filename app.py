import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="CLV Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_API_BASE = "https://clv-pre.onrender.com"

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #060A12 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 40% at 10% 0%, rgba(29,78,216,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 30% at 90% 100%, rgba(99,57,255,0.13) 0%, transparent 60%),
        #060A12 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080E1A 0%, #060A12 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}

[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

section.main > div { padding-top: 0 !important; }

.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 1600px !important;
}

/* ── HERO HEADER ── */
.hero-wrapper {
    position: relative;
    padding: 2.8rem 2.4rem 2rem 2.4rem;
    margin: 0 0 1.8rem 0;
    border-radius: 24px;
    background: linear-gradient(135deg, #0D1525 0%, #0A1020 60%, #0D1030 100%);
    border: 1px solid rgba(61,126,255,0.18);
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 50% 80% at 0% 50%, rgba(29,78,216,0.22) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 100% 20%, rgba(99,57,255,0.16) 0%, transparent 60%);
    pointer-events: none;
}
.hero-wrapper::after {
    content: 'CLV';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-family: 'Syne', sans-serif;
    font-size: 9rem; font-weight: 800;
    color: rgba(61,126,255,0.06);
    letter-spacing: -0.05em;
    pointer-events: none; line-height: 1;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(61,126,255,0.12);
    border: 1px solid rgba(61,126,255,0.3);
    border-radius: 999px; padding: 4px 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem; font-weight: 600;
    color: #6EA8FF; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 0.85rem;
}
.hero-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3D7EFF;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800; line-height: 1.05;
    letter-spacing: -0.03em; color: #F1F5FF;
    margin: 0 0 0.6rem 0;
}
.hero-title span {
    background: linear-gradient(90deg, #3D7EFF 0%, #7B5EFF 50%, #2ECC8A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.97rem; color: #6B7A99;
    font-weight: 400; line-height: 1.6;
    max-width: 560px; margin: 0;
}
.hero-stats { display: flex; gap: 2rem; margin-top: 1.6rem; }
.hero-stat  { display: flex; flex-direction: column; gap: 2px; }
.hero-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 700; color: #E2E8F0;
}
.hero-stat-lbl {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem; color: #4A5568;
    text-transform: uppercase; letter-spacing: 0.07em;
}

/* ── METRIC CARDS ── */
.kpi-card {
    position: relative;
    background: linear-gradient(145deg, #0D1525 0%, #0A1020 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px; padding: 18px 20px;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.kpi-card:hover {
    border-color: rgba(61,126,255,0.3);
    transform: translateY(-2px);
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #3D7EFF);
    border-radius: 18px 18px 0 0; opacity: 0.8;
}
.kpi-icon   { font-size: 1.3rem; margin-bottom: 10px; display: block; }
.kpi-label  {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem; font-weight: 500; color: #4A5568;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
}
.kpi-value  {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: #F1F5FF; line-height: 1; margin-bottom: 6px;
}
.kpi-sub    { font-family: 'DM Sans', sans-serif; font-size: 0.75rem; color: #2D3748; }

/* ── MODEL ROW CARDS (Leaderboard) ── */
.model-row-card {
    position: relative;
    background: linear-gradient(145deg, #0D1525 0%, #0A1020 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 16px 20px;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 16px;
    transition: border-color 0.2s;
    overflow: hidden;
}
.model-row-card:hover { border-color: rgba(61,126,255,0.25); }
.model-row-card.champion {
    border-color: rgba(46,204,138,0.3);
    background: linear-gradient(145deg, #0A1A16 0%, #0A1020 100%);
}
.model-row-card.champion::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #2ECC8A, #3D7EFF);
}
.model-rank {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    min-width: 36px; text-align: center;
}
.model-name-block { flex: 1; min-width: 120px; }
.model-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem; font-weight: 700; color: #E2E8F0;
    display: flex; align-items: center; gap: 8px;
}
.champion-badge {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.62rem; font-weight: 600;
    background: rgba(46,204,138,0.15);
    border: 1px solid rgba(46,204,138,0.3);
    color: #2ECC8A; padding: 2px 8px; border-radius: 999px;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.model-type-tag {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem; color: #4A5568; margin-top: 3px;
}
.model-metrics {
    display: flex; gap: 20px; flex-wrap: wrap;
}
.model-metric { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.model-metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem; font-weight: 700; color: #CBD5E1;
}
.model-metric-lbl {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem; color: #4A5568;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.r2-bar-wrap {
    width: 80px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px; height: 6px; overflow: hidden;
}
.r2-bar {
    height: 6px; border-radius: 4px;
    background: linear-gradient(90deg, #1D4ED8, #2ECC8A);
    transition: width 0.5s ease;
}

/* ── SECTION HEADERS ── */
.section-header { display: flex; align-items: center; gap: 10px; margin: 1.4rem 0 0.9rem 0; }
.section-bar    {
    width: 3px; height: 18px; border-radius: 2px;
    background: linear-gradient(180deg, #3D7EFF, #7B5EFF);
}
.section-title  {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem; font-weight: 700;
    color: #CBD5E1; letter-spacing: -0.01em;
}

/* ── SIDEBAR ── */
.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
}
.sidebar-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem; font-weight: 600; color: #4A5568;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
}
.status-badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 12px; border-radius: 999px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem; font-weight: 600; width: 100%;
}
.status-ok  { background: rgba(46,204,138,0.1);  border: 1px solid rgba(46,204,138,0.25); color: #2ECC8A; }
.status-bad { background: rgba(224,85,85,0.1);   border: 1px solid rgba(224,85,85,0.25);  color: #E05555; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-ok     { background: #2ECC8A; box-shadow: 0 0 6px #2ECC8A; }
.dot-bad    { background: #E05555; box-shadow: 0 0 6px #E05555; }
.model-chip {
    display: inline-block; padding: 3px 10px;
    background: rgba(61,126,255,0.1);
    border: 1px solid rgba(61,126,255,0.2);
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif; font-size: 0.75rem; color: #6EA8FF;
    margin-top: 7px; width: 100%; text-align: center;
}

/* ── FILE UPLOAD SECTION ── */
.upload-section {
    background: rgba(61,126,255,0.03);
    border: 1px dashed rgba(61,126,255,0.2);
    border-radius: 14px; padding: 12px 14px; margin-bottom: 10px;
}
.upload-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem; font-weight: 600; color: #4A5568;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;
}
.file-ready-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(46,204,138,0.1);
    border: 1px solid rgba(46,204,138,0.2);
    border-radius: 8px; padding: 5px 10px; margin-top: 6px;
    font-family: 'DM Sans', sans-serif; font-size: 0.75rem; color: #2ECC8A;
    width: 100%;
}

/* ── TABS ── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    color: #4A5568 !important; letter-spacing: 0.01em !important;
    border-radius: 10px 10px 0 0 !important; padding: 8px 18px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #F1F5FF !important;
    background: rgba(61,126,255,0.08) !important;
    border-bottom: 2px solid #3D7EFF !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important; overflow: hidden !important;
    background: #0A1020 !important;
}

/* ── BUTTONS ── */
div.stButton > button,
div.stDownloadButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(61,126,255,0.35) !important;
    background: linear-gradient(135deg, #1D4ED8 0%, #1a3abf 100%) !important;
    color: #F1F5FF !important; padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important; letter-spacing: 0.01em !important;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: #3D7EFF !important;
    box-shadow: 0 0 0 3px rgba(61,126,255,0.15), 0 4px 16px rgba(29,78,216,0.3) !important;
    transform: translateY(-1px) !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border-radius: 14px !important;
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed rgba(61,126,255,0.3) !important;
    border-radius: 12px !important;
    background: rgba(61,126,255,0.03) !important;
}

/* ── TEXT INPUTS ── */
[data-testid="stTextInput"] input {
    font-family: 'DM Sans', sans-serif !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; color: #CBD5E1 !important;
    font-size: 0.85rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(61,126,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(61,126,255,0.1) !important;
}

hr { border-color: rgba(255,255,255,0.05) !important; }
[data-testid="stSpinner"] { color: #3D7EFF !important; }
[data-testid="stAlert"] {
    border-radius: 12px !important; border: none !important;
    font-family: 'DM Sans', sans-serif !important;
}
p, li, label, span, div { font-family: 'DM Sans', sans-serif !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 700 !important;
    color: #4A5568 !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stPlotlyChart"] {
    border-radius: 16px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stJson"] {
    background: #0A1020 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOT DEFAULTS
# ─────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=16, b=16),
    font=dict(color="#6B7A99", family="DM Sans, sans-serif", size=11),
)
TIER_COLORS = {
    "Premium":      "#7B5EFF",
    "High Value":   "#2ECC8A",
    "Medium Value": "#F59E0B",
    "Low Value":    "#E05555",
}
MODEL_COLORS = ["#3D7EFF", "#2ECC8A", "#7B5EFF", "#F59E0B", "#E05555", "#38BDF8", "#F97316"]

# Model type tags for known models
MODEL_TYPE_MAP = {
    "Ridge":      "Linear · Regularized",
    "Lasso":      "Linear · Sparse",
    "ElasticNet": "Linear · L1+L2",
    "XGBoost":    "Gradient Boosting · Tree",
    "LightGBM":   "Gradient Boosting · Leaf-wise",
}
RANK_COLORS = ["#F59E0B", "#CBD5E1", "#E05555", "#2ECC8A", "#7B5EFF"]


# ─────────────────────────────────────────────
#  HELPER COMPONENTS
# ─────────────────────────────────────────────
def kpi_card(label, value, subtitle="", accent="#3D7EFF", icon=""):
    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""
    return (
        f'<div class="kpi-card" style="--accent:{accent};">'
        f'{icon_html}'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{subtitle}</div>'
        f'</div>'
    )


def section_header(title, icon=""):
    prefix = icon + " " if icon else ""
    return (
        f'<div class="section-header">'
        f'<div class="section-bar"></div>'
        f'<div class="section-title">{prefix}{title}</div>'
        f'</div>'
    )


def api_get(base_url, path, params=None):
    try:
        r = requests.get(f"{base_url}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def api_post_file(base_url, path, uploaded_file):
    try:
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        r = requests.post(f"{base_url}{path}", files=files, timeout=180)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def classify_tier_order(df, col="tier"):
    order_map = {"Premium": 4, "High Value": 3, "Medium Value": 2, "Low Value": 1}
    if col in df.columns:
        df = df.copy()
        df["_tier_order"] = df[col].map(order_map).fillna(0)
        df = df.sort_values(
            ["_tier_order", col], ascending=[False, True]
        ).drop(columns=["_tier_order"])
    return df


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-tag">
        <span class="hero-dot"></span>AI-Powered Analytics
    </div>
    <h1 class="hero-title">CLV Prediction<br><span>Dashboard</span></h1>
    <p class="hero-subtitle">
        Upload customer data, score cohorts through the FastAPI backend, and explore
        model performance with rich interactive visualisations.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-val">5+</span>
            <span class="hero-stat-lbl">ML Models</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">4</span>
            <span class="hero-stat-lbl">CLV Tiers</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">Real-time</span>
            <span class="hero-stat-lbl">Predictions</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
#  FIX: st.file_uploader is called EXACTLY ONCE,
#  never wrapped in st.markdown("<div>...</div>") open/close
#  pairs — that caused the widget to render twice in the DOM.
#  The file is stored in st.session_state so it persists
#  across reruns without re-uploading.
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # ── Backend status ──────────────────────────
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Backend URL</div>', unsafe_allow_html=True)
    api_base = st.text_input(
        "api_endpoint", value=DEFAULT_API_BASE, label_visibility="collapsed"
    ).strip()
    st.markdown("</div>", unsafe_allow_html=True)

    health_data, health_err = api_get(api_base, "/")
    if health_err:
        st.markdown(
            '<div class="status-badge status-bad">'
            '<span class="status-dot dot-bad"></span>Backend Unreachable'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        loaded     = health_data.get("loaded", False)
        model_name = health_data.get("model", "Unknown")
        if loaded:
            st.markdown(
                '<div class="status-badge status-ok">'
                '<span class="status-dot dot-ok"></span>Backend Connected'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="model-chip">🤖 {model_name}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge status-bad">'
                '<span class="status-dot dot-bad"></span>Model Not Loaded'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── File upload ─────────────────────────────
    st.markdown("### 📂 Batch Input")
    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        key="csv_uploader",
    )

    # Persist the file reference in session state so it survives reruns
    if uploaded_file is not None:
        st.session_state["last_uploaded_file"] = uploaded_file
        st.markdown(
            f'<div class="file-ready-badge">✔ {uploaded_file.name}</div>',
            unsafe_allow_html=True,
        )
    elif "last_uploaded_file" in st.session_state:
        # File was cleared — remove from session state
        del st.session_state["last_uploaded_file"]

    run_batch = st.button("▶  Run Batch Prediction", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
    <div style="font-size:0.78rem;color:#4A5568;line-height:1.8;">
        <b style="color:#CBD5E1;">1.</b> Train models via
        <code style="color:#6EA8FF;">ml_training.py</code><br>
        <b style="color:#CBD5E1;">2.</b> Start FastAPI backend<br>
        <b style="color:#CBD5E1;">3.</b> Upload a customer CSV<br>
        <b style="color:#CBD5E1;">4.</b> Click Run Batch Prediction
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Batch Prediction  ",
    "  🏅  Leaderboard  ",
    "  🎯  Best Model  ",
    "  🔬  Feature Importance  ",
])


# ══════════════════════════════════════════════
#  TAB 1 — BATCH PREDICTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown(section_header("Batch Prediction", "📊"), unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            preview_df = pd.read_csv(uploaded_file)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(
                    kpi_card("Total Records", f"{len(preview_df):,}", "Uploaded rows", "#3D7EFF", "📁"),
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    kpi_card("Feature Columns", f"{preview_df.shape[1]:,}", "Available signals", "#2ECC8A", "🧩"),
                    unsafe_allow_html=True,
                )
            with col_c:
                missing = int(preview_df.isna().sum().sum())
                accent  = "#F59E0B" if missing > 0 else "#2ECC8A"
                icon    = "⚠️" if missing > 0 else "✅"
                st.markdown(
                    kpi_card("Missing Values", f"{missing:,}", "Will default to 0", accent, icon),
                    unsafe_allow_html=True,
                )

            st.markdown(section_header("CSV Preview — First 10 Rows", "👁"), unsafe_allow_html=True)
            st.dataframe(preview_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"CSV preview failed: {e}")

    if run_batch:
        if uploaded_file is None:
            st.warning("Please upload a CSV file first.")
        else:
            with st.spinner("Scoring customers — please wait..."):
                start   = time.time()
                result, err = api_post_file(api_base, "/predict/batch", uploaded_file)
                elapsed = time.time() - start

            if err:
                st.error(f"Batch prediction failed: {err}")
            else:
                rows       = result.get("rows", 0)
                model_name = result.get("model", "Unknown")
                results    = result.get("results", [])

                if not results:
                    st.warning("No results returned from the backend.")
                else:
                    results_df = pd.DataFrame(results)
                    prem_pct   = (results_df["tier"] == "Premium").mean() * 100
                    low_pct    = (results_df["tier"] == "Low Value").mean() * 100

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.markdown(
                            kpi_card("Rows Scored", f"{rows:,}", f"In {elapsed:.1f}s", "#3D7EFF", "⚡"),
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            kpi_card("Avg CLV", f"${results_df['clv'].mean():,.0f}", model_name, "#2ECC8A", "💰"),
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            kpi_card("Max CLV", f"${results_df['clv'].max():,.0f}", "Top customer", "#7B5EFF", "🚀"),
                            unsafe_allow_html=True,
                        )
                    with c4:
                        st.markdown(
                            kpi_card("Premium", f"{prem_pct:.1f}%", "Premium tier", "#F59E0B", "💎"),
                            unsafe_allow_html=True,
                        )
                    with c5:
                        st.markdown(
                            kpi_card("Low Value", f"{low_pct:.1f}%", "At-risk segment", "#E05555", "📉"),
                            unsafe_allow_html=True,
                        )

                    tc = results_df["tier"].value_counts().reset_index()
                    tc.columns = ["Tier", "Count"]
                    col_l, col_r = st.columns(2)

                    with col_l:
                        st.markdown(section_header("Tier Distribution", "🍩"), unsafe_allow_html=True)
                        fig_pie = go.Figure(go.Pie(
                            labels=tc["Tier"], values=tc["Count"], hole=0.62,
                            marker=dict(
                                colors=[TIER_COLORS.get(t, "#3D7EFF") for t in tc["Tier"]],
                                line=dict(color="#060A12", width=3),
                            ),
                            textinfo="none",
                            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
                        ))
                        fig_pie.add_annotation(
                            text=f"<b>{rows:,}</b><br><span style='font-size:10px'>customers</span>",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(color="#F1F5FF", size=16),
                            align="center",
                        )
                        fig_pie.update_layout(
                            **PLOT_BASE, height=320,
                            legend=dict(
                                font=dict(size=11, color="#6B7A99"),
                                x=0.72, y=0.5, bgcolor="rgba(0,0,0,0)",
                            ),
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_r:
                        st.markdown(section_header("CLV Distribution", "📈"), unsafe_allow_html=True)
                        fig_hist = go.Figure(go.Histogram(
                            x=results_df["clv"], nbinsx=30,
                            marker=dict(
                                color="#3D7EFF", opacity=0.85,
                                line=dict(color="#060A12", width=0.5),
                            ),
                            hovertemplate="CLV: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
                        ))
                        fig_hist.update_layout(
                            **PLOT_BASE, height=320,
                            xaxis=dict(
                                title="Predicted CLV ($)", tickformat="$,.0f",
                                gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10),
                            ),
                            yaxis=dict(
                                title="Customers",
                                gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10),
                            ),
                            bargap=0.05,
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    results_df = classify_tier_order(results_df, "tier")
                    st.markdown(section_header("Prediction Results", "📋"), unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)

                    st.download_button(
                        "⬇  Download Prediction Results",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="clv_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# ══════════════════════════════════════════════
#  TAB 2 — LEADERBOARD
#  Shows every model trained in ml_training.py
#  (Ridge, Lasso, ElasticNet, XGBoost, LightGBM)
#  as individual ranked cards + comparison charts.
# ══════════════════════════════════════════════
with tab2:
    st.markdown(section_header("Model Leaderboard", "🏅"), unsafe_allow_html=True)

    leaderboard_data, err = api_get(api_base, "/leaderboard")
    if err:
        st.error(f"Could not load leaderboard: {err}")
    else:
        lb_df = pd.DataFrame(leaderboard_data)
        if lb_df.empty:
            st.warning("Leaderboard is empty. Run `python ml_training.py` first.")
        else:
            # Sort by R2 descending
            if "R2" in lb_df.columns:
                lb_df = lb_df.sort_values("R2", ascending=False).reset_index(drop=True)

            top       = lb_df.iloc[0]
            top_model = str(top.get("Model", "N/A"))
            top_r2    = float(top.get("R2",    0))
            top_cv    = float(top.get("CV_R2", 0))
            top_rmse  = float(top.get("RMSE",  0))

            # ── Summary KPIs ─────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    kpi_card("Champion Model", top_model, "Best overall R²", "#2ECC8A", "🥇"),
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    kpi_card("Top R²", f"{top_r2:.4f}", "Holdout test score", "#3D7EFF", "📐"),
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    kpi_card("Top CV R²", f"{top_cv:.4f}", "Cross-validation score", "#7B5EFF", "🔄"),
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    kpi_card("Models Trained", f"{len(lb_df)}", "Total evaluated", "#F59E0B", "🧪"),
                    unsafe_allow_html=True,
                )

            # ── Per-model ranked cards ───────────────────────
            st.markdown(section_header("All Models — Ranked by R²", "📊"), unsafe_allow_html=True)

            max_r2 = lb_df["R2"].max() if lb_df["R2"].max() > 0 else 1.0

            for idx, row in lb_df.iterrows():
                rank        = idx + 1
                model_nm    = str(row.get("Model", "Unknown"))
                r2          = float(row.get("R2",    0))
                cv_r2       = float(row.get("CV_R2", 0))
                rmse        = float(row.get("RMSE",  0))
                mae         = float(row.get("MAE",   0))
                mape        = float(row.get("MAPE",  0)) if "MAPE" in row else 0.0
                overfit     = float(row.get("Overfit_Gap", 0)) if "Overfit_Gap" in row else 0.0
                train_time  = float(row.get("Train_Time_s", 0)) if "Train_Time_s" in row else 0.0
                bar_pct     = (r2 / max_r2) * 100 if max_r2 > 0 else 0
                is_champion = rank == 1
                card_class  = "model-row-card champion" if is_champion else "model-row-card"
                rank_color  = RANK_COLORS[min(rank - 1, len(RANK_COLORS) - 1)]
                type_tag    = MODEL_TYPE_MAP.get(model_nm, "ML Model")
                overfit_color = "#E05555" if abs(overfit) > 0.05 else "#2ECC8A"
                champion_html = '<span class="champion-badge">Champion</span>' if is_champion else ""

                card_html = (
                    f'<div class="{card_class}">'
                    f'<div class="model-rank" style="color:{rank_color};">#{rank}</div>'
                    f'<div class="model-name-block">'
                    f'<div class="model-name">{model_nm} {champion_html}</div>'
                    f'<div class="model-type-tag">{type_tag}</div>'
                    f'<div class="r2-bar-wrap" style="margin-top:8px;">'
                    f'<div class="r2-bar" style="width:{bar_pct:.1f}%;"></div>'
                    f'</div></div>'
                    f'<div class="model-metrics">'
                    f'<div class="model-metric"><div class="model-metric-val" style="color:#3D7EFF;">{r2:.4f}</div><div class="model-metric-lbl">R²</div></div>'
                    f'<div class="model-metric"><div class="model-metric-val" style="color:#7B5EFF;">{cv_r2:.4f}</div><div class="model-metric-lbl">CV R²</div></div>'
                    f'<div class="model-metric"><div class="model-metric-val">&#36;{rmse:,.0f}</div><div class="model-metric-lbl">RMSE</div></div>'
                    f'<div class="model-metric"><div class="model-metric-val">&#36;{mae:,.0f}</div><div class="model-metric-lbl">MAE</div></div>'
                    f'<div class="model-metric"><div class="model-metric-val" style="color:{overfit_color};">{overfit:+.4f}</div><div class="model-metric-lbl">Overfit Gap</div></div>'
                    f'<div class="model-metric"><div class="model-metric-val">{train_time:.1f}s</div><div class="model-metric-lbl">Train Time</div></div>'
                    f'</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

            # ── Comparison charts ────────────────────────────
            chart_l, chart_r = st.columns(2)

            with chart_l:
                st.markdown(section_header("R² Score Comparison", "📊"), unsafe_allow_html=True)
                bar_colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(lb_df))]
                fig_r2 = go.Figure(go.Bar(
                    x=lb_df["R2"],
                    y=lb_df["Model"],
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        opacity=0.85,
                        line=dict(color="rgba(0,0,0,0.3)", width=1),
                    ),
                    text=[f"  {v:.4f}" for v in lb_df["R2"]],
                    textposition="outside",
                    textfont=dict(color="#CBD5E1", size=11),
                    hovertemplate="<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>",
                ))
                fig_r2.update_layout(
                    **PLOT_BASE, height=max(300, len(lb_df) * 60),
                    xaxis=dict(
                        title="R²",
                        gridcolor="rgba(255,255,255,0.04)",
                        tickfont=dict(size=10),
                        range=[0, lb_df["R2"].max() * 1.15],
                    ),
                    yaxis=dict(title="", autorange="reversed", tickfont=dict(size=11)),
                )
                st.plotly_chart(fig_r2, use_container_width=True)

            with chart_r:
                st.markdown(section_header("RMSE & MAE by Model", "📉"), unsafe_allow_html=True)
                fig_err = go.Figure()
                fig_err.add_trace(go.Bar(
                    x=lb_df["Model"], y=lb_df["RMSE"], name="RMSE",
                    marker=dict(color="#3D7EFF", opacity=0.85),
                    hovertemplate="<b>%{x}</b><br>RMSE: $%{y:,.0f}<extra></extra>",
                ))
                if "MAE" in lb_df.columns:
                    fig_err.add_trace(go.Bar(
                        x=lb_df["Model"], y=lb_df["MAE"], name="MAE",
                        marker=dict(color="#7B5EFF", opacity=0.7),
                        hovertemplate="<b>%{x}</b><br>MAE: $%{y:,.0f}<extra></extra>",
                    ))
                fig_err.update_layout(
                    **PLOT_BASE, height=max(300, len(lb_df) * 60),
                    barmode="group",
                    xaxis=dict(tickfont=dict(size=11), gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(
                        title="Error ($)",
                        gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10),
                    ),
                    legend=dict(
                        orientation="h", y=1.08, x=0,
                        font=dict(size=11, color="#CBD5E1"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    bargap=0.2, bargroupgap=0.06,
                )
                st.plotly_chart(fig_err, use_container_width=True)

            # ── CV R² scatter ─────────────────────────────────
            st.markdown(section_header("CV R² vs RMSE — Bubble Chart", "🫧"), unsafe_allow_html=True)
            if "MAE" in lb_df.columns:
                sv = pd.to_numeric(lb_df["MAE"], errors="coerce").fillna(lb_df["MAE"].mean())
                mn, mx = sv.min(), sv.max()
                bsizes = ((sv - mn) / (mx - mn + 1e-9) * 28 + 16).tolist()
            else:
                bsizes = [22] * len(lb_df)

            fig_sc = go.Figure(go.Scatter(
                x=lb_df["CV_R2"],
                y=lb_df["RMSE"],
                mode="markers+text",
                text=lb_df["Model"],
                textposition="top center",
                textfont=dict(size=10, color="#CBD5E1"),
                marker=dict(
                    size=bsizes,
                    color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(lb_df))],
                    line=dict(color="rgba(255,255,255,0.15)", width=1.5),
                    opacity=0.9,
                ),
                hovertemplate="<b>%{text}</b><br>CV R²: %{x:.4f}<br>RMSE: $%{y:,.0f}<extra></extra>",
            ))
            fig_sc.update_layout(
                **PLOT_BASE, height=380,
                xaxis=dict(title="CV R² (higher = better)", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
                yaxis=dict(title="RMSE $ (lower = better)", gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            # ── Overfit gap chart ─────────────────────────────
            if "Overfit_Gap" in lb_df.columns:
                st.markdown(section_header("Overfit Gap (Train R² − Test R²)", "⚖️"), unsafe_allow_html=True)
                gap_colors = ["#E05555" if abs(v) > 0.05 else "#2ECC8A" for v in lb_df["Overfit_Gap"]]
                fig_gap = go.Figure(go.Bar(
                    x=lb_df["Model"],
                    y=lb_df["Overfit_Gap"],
                    marker=dict(color=gap_colors, opacity=0.85),
                    text=[f"{v:+.4f}" for v in lb_df["Overfit_Gap"]],
                    textposition="outside",
                    textfont=dict(color="#CBD5E1", size=11),
                    hovertemplate="<b>%{x}</b><br>Overfit Gap: %{y:+.4f}<extra></extra>",
                ))
                fig_gap.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dot")
                fig_gap.update_layout(
                    **PLOT_BASE, height=280,
                    xaxis=dict(tickfont=dict(size=11)),
                    yaxis=dict(
                        title="Gap (lower = more generalised)",
                        gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10),
                    ),
                )
                st.plotly_chart(fig_gap, use_container_width=True)

            # ── Full table ────────────────────────────────────
            st.markdown(section_header("Full Leaderboard Table", "📋"), unsafe_allow_html=True)
            display_cols = [c for c in ["Model","R2","CV_R2","CV_R2_std","RMSE","MAE","MAPE","Overfit_Gap","Train_Time_s"] if c in lb_df.columns]
            st.dataframe(lb_df[display_cols], use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — BEST MODEL
# ══════════════════════════════════════════════
with tab3:
    st.markdown(section_header("Best Model Summary", "🎯"), unsafe_allow_html=True)

    best_data, err           = api_get(api_base, "/best-model")
    leaderboard_data, lb_err = api_get(api_base, "/leaderboard")

    if err:
        st.error(f"Could not load best model info: {err}")
    else:
        best_model_name = best_data.get("best_model", "N/A")
        metrics         = best_data.get("metrics", {})
        bm_r2           = float(metrics.get("R2",    0))
        bm_cv           = float(metrics.get("CV_R2", 0))
        bm_rmse         = float(metrics.get("RMSE",  0))
        bm_mae          = float(metrics.get("MAE",   0))

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(
                kpi_card("Best Model", best_model_name, "Selected winner", "#3D7EFF", "🏆"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                kpi_card("R²", f"{bm_r2:.4f}", "Higher is better", "#2ECC8A", "📐"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                kpi_card("CV R²", f"{bm_cv:.4f}", "Generalisation", "#7B5EFF", "🔄"),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                kpi_card("RMSE", f"${bm_rmse:,.0f}", "Error spread", "#F59E0B", "📏"),
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                kpi_card("MAE", f"${bm_mae:,.0f}", "Avg absolute error", "#E05555", "🎯"),
                unsafe_allow_html=True,
            )

        chart_l, chart_r = st.columns(2)

        with chart_l:
            st.markdown(section_header("Performance Radar", "🕸"), unsafe_allow_html=True)
            r2    = float(metrics.get("R2",    0))
            cv_r2 = float(metrics.get("CV_R2", 0))
            rmse  = float(metrics.get("RMSE",  0))
            mae   = float(metrics.get("MAE",   0))
            mape  = float(metrics.get("MAPE",  0)) if "MAPE" in metrics else 0.0
            rv    = [r2, cv_r2, 1 / (1 + rmse), 1 / (1 + mae), 1 / (1 + mape)]
            rl    = ["R²", "CV R²", "RMSE Score", "MAE Score", "MAPE Score"]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=rv + [rv[0]], theta=rl + [rl[0]],
                fill="toself",
                line=dict(color="#3D7EFF", width=2.5),
                fillcolor="rgba(61,126,255,0.15)",
                hovertemplate="%{theta}: %{r:.4f}<extra></extra>",
            ))
            fig_radar.update_layout(
                **PLOT_BASE, height=360,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(1.0, max(rv) if rv else 1.0)],
                        gridcolor="rgba(255,255,255,0.06)",
                        tickfont=dict(size=9, color="#4A5568"),
                        linecolor="rgba(255,255,255,0.06)",
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255,255,255,0.06)",
                        tickfont=dict(size=10, color="#CBD5E1"),
                    ),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with chart_r:
            st.markdown(section_header("Best Model vs Leaderboard Average", "⚖️"), unsafe_allow_html=True)
            if lb_err:
                st.info("Leaderboard comparison unavailable.")
            else:
                lb_df2 = pd.DataFrame(leaderboard_data)
                if lb_df2.empty:
                    st.info("Leaderboard comparison unavailable.")
                else:
                    avg_r2   = float(lb_df2["R2"].mean())    if "R2"    in lb_df2.columns else 0
                    avg_cv   = float(lb_df2["CV_R2"].mean()) if "CV_R2" in lb_df2.columns else 0
                    avg_rmse = float(lb_df2["RMSE"].mean())  if "RMSE"  in lb_df2.columns else 0
                    avg_mae  = float(lb_df2["MAE"].mean())   if "MAE"   in lb_df2.columns else 0

                    cmp_metrics = ["R²", "CV R²", "RMSE", "MAE"]
                    best_vals   = [
                        float(metrics.get("R2",    0)),
                        float(metrics.get("CV_R2", 0)),
                        float(metrics.get("RMSE",  0)),
                        float(metrics.get("MAE",   0)),
                    ]
                    avg_vals = [avg_r2, avg_cv, avg_rmse, avg_mae]

                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Bar(
                        x=cmp_metrics, y=best_vals, name=best_model_name,
                        marker=dict(color="#3D7EFF", opacity=0.9),
                        hovertemplate="%{x}: %{y:,.4f}<extra></extra>",
                    ))
                    fig_cmp.add_trace(go.Bar(
                        x=cmp_metrics, y=avg_vals, name="Leaderboard Avg",
                        marker=dict(color="#2ECC8A", opacity=0.6),
                        hovertemplate="%{x}: %{y:,.4f}<extra></extra>",
                    ))
                    fig_cmp.update_layout(
                        **PLOT_BASE, height=360, barmode="group",
                        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=11)),
                        yaxis=dict(
                            title="Metric Value",
                            gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10),
                        ),
                        legend=dict(
                            orientation="h", y=1.08, x=0,
                            font=dict(size=11, color="#CBD5E1"),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        bargap=0.25, bargroupgap=0.08,
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown(section_header("Raw Metrics JSON", "🔧"), unsafe_allow_html=True)
        st.json(best_data)


# ══════════════════════════════════════════════
#  TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════
with tab4:
    st.markdown(section_header("Feature Importance", "🔬"), unsafe_allow_html=True)

    top_n = st.slider("Top N Features to Display", min_value=5, max_value=30, value=15, step=1)

    fi_data, err = api_get(api_base, "/feature-importance", params={"top_n": top_n})
    if err:
        st.error(f"Could not load feature importance: {err}")
    else:
        fi_df = pd.DataFrame(fi_data)
        if fi_df.empty:
            st.info("This model does not expose feature importance (linear models only).")
        else:
            fi_df   = fi_df.sort_values("importance", ascending=True)
            max_imp = fi_df["importance"].max()

            colors = []
            for v in fi_df["importance"]:
                ratio = v / max_imp if max_imp > 0 else 0
                if ratio > 0.75:
                    colors.append("#3D7EFF")
                elif ratio > 0.45:
                    colors.append("#2ECC8A")
                elif ratio > 0.2:
                    colors.append("#F59E0B")
                else:
                    colors.append("#4A5568")

            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                marker=dict(
                    color=colors, opacity=0.88,
                    line=dict(color="rgba(0,0,0,0.3)", width=0.8),
                ),
                text=[f"  {v:.5f}" for v in fi_df["importance"]],
                textposition="outside",
                textfont=dict(color="#6B7A99", size=10),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>",
            ))
            fig_fi.update_layout(
                **PLOT_BASE,
                height=max(400, top_n * 28),
                xaxis=dict(
                    title="Feature Importance Score",
                    gridcolor="rgba(255,255,255,0.04)",
                    tickfont=dict(size=10),
                    range=[0, max_imp * 1.18],
                ),
                yaxis=dict(
                    title="",
                    gridcolor="rgba(255,255,255,0.04)",
                    tickfont=dict(size=11, color="#CBD5E1"),
                ),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(section_header("Importance Table", "📋"), unsafe_allow_html=True)
                st.dataframe(
                    fi_df.sort_values("importance", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                )

            with col_b:
                st.markdown(section_header("Top 5 Highlights", "⭐"), unsafe_allow_html=True)
                top5        = fi_df.sort_values("importance", ascending=False).head(5)
                rank_colors = ["#F59E0B", "#CBD5E1", "#E05555", "#2ECC8A", "#7B5EFF"]
                for i, (_, row) in enumerate(top5.iterrows()):
                    ratio = row["importance"] / max_imp if max_imp > 0 else 0
                    c     = rank_colors[i]
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:12px;'
                        f'padding:10px 14px;margin-bottom:8px;'
                        f'background:rgba(255,255,255,0.03);'
                        f'border:1px solid rgba(255,255,255,0.05);'
                        f'border-radius:12px;">'
                        f'<div style="font-family:\'Syne\',sans-serif;font-size:1.1rem;'
                        f'font-weight:800;color:{c};min-width:24px;">#{i + 1}</div>'
                        f'<div style="flex:1;">'
                        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.85rem;'
                        f'color:#CBD5E1;font-weight:500;">{row["feature"]}</div>'
                        f'<div style="height:4px;background:rgba(255,255,255,0.05);'
                        f'border-radius:2px;margin-top:5px;">'
                        f'<div style="height:4px;width:{ratio * 100:.1f}%;'
                        f'background:{c};border-radius:2px;"></div>'
                        f'</div>'
                        f'</div>'
                        f'<div style="font-family:\'Syne\',sans-serif;font-size:0.82rem;'
                        f'font-weight:700;color:{c};">{row["importance"]:.5f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
