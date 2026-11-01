import html
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# 1. PROJECT PATHS
# ==========================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Works whether app.py is in root or dashboard folder
if os.path.basename(APP_DIR).lower() == "dashboard":
    BASE_DIR = os.path.dirname(APP_DIR)
else:
    BASE_DIR = APP_DIR

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_complaints.csv")
COMPARISON_PATH = os.path.join(BASE_DIR, "results", "model_comparison.csv")
CONFUSION_PATH = os.path.join(BASE_DIR, "results", "confusion_matrix.png")


# ==========================================
# 2. PAGE CONFIGURATION + VISUAL SYSTEM
# ==========================================

st.set_page_config(
    page_title="AI Complaint Intelligence",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #162033;
        --muted: #667085;
        --line: #e7ebf2;
        --surface: #ffffff;
        --canvas: #f5f7fb;
        --navy: #182a4a;
        --blue: #3678f6;
        --mint: #ddf8ed;
        --mint-ink: #087443;
        --amber: #fff1d6;
        --amber-ink: #a45b00;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--canvas); }
    .block-container { max-width: 1440px; padding: 2.2rem 3.5rem 2rem; }

    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.035em; color: var(--ink); }
    h1 { font-size: 2.5rem !important; margin-bottom: .25rem !important; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.15rem !important; }
    p, li, label, .stCaption { color: var(--muted); }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1.6rem 1.2rem; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #f7f9fc !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: .55rem .7rem;
        transition: background .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.08); }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.12);
    }

    .brand { display: flex; gap: .75rem; align-items: center; margin-bottom: .3rem; }
    .brand-mark {
        width: 2.25rem; height: 2.25rem; display: grid; place-items: center;
        border-radius: 10px; background: linear-gradient(135deg, #5f9bff, #7b61ff);
        color: white; font-size: 1.2rem; box-shadow: 0 8px 22px rgba(0,0,0,.18);
    }
    .brand-title { font: 700 1.03rem 'Space Grotesk', sans-serif; color: white; }
    .brand-subtitle { font-size: .73rem; color: #aab7cd; margin-top: .1rem; }

    .eyebrow { color: var(--blue); text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 700; margin-bottom: .65rem; }
    .page-intro { color: var(--muted); font-size: 1.03rem; max-width: 680px; margin-bottom: 1.6rem; }
    .hero {
        background: linear-gradient(120deg, #182a4a 0%, #274a86 62%, #3f79dc 100%);
        border-radius: 22px; padding: 2.3rem 2.5rem; color: white; margin-bottom: 1.4rem;
        box-shadow: 0 18px 42px rgba(25, 51, 96, .18);
    }
    .hero h1 { color: white; font-size: 2.75rem !important; margin: 0 0 .55rem !important; }
    .hero p { color: #d8e4fb; max-width: 630px; margin: 0; font-size: 1.05rem; }
    .hero-tag { display: inline-flex; gap: .4rem; align-items: center; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.18); border-radius: 999px; padding: .35rem .7rem; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 1rem; }

    .card {
        background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
        padding: 1.25rem 1.35rem; box-shadow: 0 5px 20px rgba(24,42,74,.035);
    }
    .card-title { font: 700 1.02rem 'Space Grotesk', sans-serif; color: var(--ink); margin-bottom: .45rem; }
    .card-copy { color: var(--muted); font-size: .92rem; line-height: 1.55; }
    .metric-card { min-height: 112px; }
    .metric-label { color: var(--muted); font-size: .8rem; font-weight: 600; }
    .metric-value { font: 700 1.75rem 'Space Grotesk', sans-serif; color: var(--ink); margin-top: .45rem; }
    .metric-accent { width: 28px; height: 4px; border-radius: 99px; background: var(--blue); margin-top: .8rem; }

    .result-card { background: #eff6ff; border: 1px solid #cfe1ff; border-radius: 16px; padding: 1.4rem; }
    .result-label { color: #55709d; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }
    .result-value { font: 700 1.65rem 'Space Grotesk', sans-serif; color: #1f4d9a; margin-top: .35rem; }
    .status-pill { display: inline-block; padding: .3rem .6rem; border-radius: 999px; background: var(--mint); color: var(--mint-ink); font-size: .75rem; font-weight: 700; }
    .muted-note { color: var(--muted); font-size: .84rem; }

    .stButton > button { border-radius: 10px; font-weight: 700; min-height: 2.65rem; border: 1px solid #d9e1ee; }
    .stButton > button[kind="primary"] { background: var(--blue); border-color: var(--blue); }
    /* Explicit text colors keep typed input readable across Streamlit themes. */
    .stTextArea textarea,
    .stTextInput input {
        border-radius: 12px;
        border-color: #d9e1ee;
        background: #ffffff !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        caret-color: var(--blue) !important;
    }
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #98a2b3 !important;
        -webkit-text-fill-color: #98a2b3 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 5px 20px rgba(24,42,74,.035); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 12px; }
    .section-rule { height: 1px; background: var(--line); margin: 1.8rem 0; }
    .footer { text-align: center; color: #98a2b3; font-size: .78rem; padding: 1.2rem 0 .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 3. DATA + MODEL LOADING
# ==========================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_dataset(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_comparison(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def card(title, copy, icon):
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:1.45rem; margin-bottom:.55rem">{icon}</div>
            <div class="card-title">{html.escape(title)}</div>
            <div class="card-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="card metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-accent"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if not os.path.exists(MODEL_PATH):
    st.error("The trained model could not be found.")
    st.markdown("The app expected the model at:")
    st.code(MODEL_PATH)
    st.info("Run `python train_model.py` from the project root to generate it.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

df = load_dataset(DATA_PATH)
comparison_df = load_comparison(COMPARISON_PATH)


# ==========================================
# 4. SESSION STATE + SIDEBAR
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-title">Complaint Intelligence</div>
                <div class="brand-subtitle">AI-powered service insights</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("WORKSPACE")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Complaint Analyzer",
            "Model Comparison",
            "Confusion Matrix",
            "Category Analysis",
            "Prediction History",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Model stack**")
    st.caption("Linear SVM  ·  TF-IDF  ·  Multiclass")
    st.markdown("**Session predictions**")
    st.caption(f"{len(st.session_state.history)} complaint(s) classified")
    st.divider()
    st.caption("MCA Project · AI & Machine Learning")


# ==========================================
# 5. OVERVIEW
# ==========================================

if page == "Overview":
    total_records = len(df) if not df.empty else 0
    accuracy = 0.0
    macro_f1 = 0.0
    if not comparison_df.empty and "Model" in comparison_df.columns:
        svm_row = comparison_df[comparison_df["Model"] == "Linear SVM"]
        if not svm_row.empty:
            accuracy = float(svm_row.iloc[0].get("Accuracy", 0)) * 100
            macro_f1 = float(svm_row.iloc[0].get("Macro F1", 0)) * 100
    total_categories = int(df["category"].nunique()) if "category" in df.columns else 0

    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">● Model online</div>
            <h1>Turn complaints into<br>clear next steps.</h1>
            <p>Classify customer complaints automatically, understand category patterns, and monitor model quality from one focused workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="eyebrow">At a glance</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4, gap="medium")
    with metric_cols[0]:
        metric_card("Total complaints", f"{total_records:,}")
    with metric_cols[1]:
        metric_card("Model accuracy", f"{accuracy:.2f}%")
    with metric_cols[2]:
        metric_card("Macro F1 score", f"{macro_f1:.2f}%")
    with metric_cols[3]:
        metric_card("Active categories", total_categories)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">What you can do</div>', unsafe_allow_html=True)
    feature_cols = st.columns(3, gap="medium")
    with feature_cols[0]:
        card("Classify instantly", "Paste a complaint and get a category prediction in seconds.", "⌁")
    with feature_cols[1]:
        card("Compare models", "See how accuracy, precision, recall, and F1 score compare.", "◒")
    with feature_cols[2]:
        card("Find patterns", "Explore category volume and review predictions from this session.", "◌")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("### How the classifier works")
        st.markdown(
            """
            <div class="card">
                <div class="card-title">From text to category</div>
                <div class="card-copy">TF-IDF converts the complaint into weighted text features. The Linear SVM then matches those patterns to one of the trained complaint categories.</div>
                <div style="display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1rem">
                    <span class="status-pill">1 · Read text</span>
                    <span class="status-pill">2 · Extract signals</span>
                    <span class="status-pill">3 · Predict category</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### System status")
        st.success("Trained model loaded and ready")
        if df.empty:
            st.warning("Dataset unavailable — category metrics may be incomplete.")
        else:
            st.info(f"Dataset connected · {len(df):,} records available")


# ==========================================
# 6. COMPLAINT ANALYZER
# ==========================================

elif page == "Complaint Analyzer":
    st.markdown('<div class="eyebrow">Live prediction</div>', unsafe_allow_html=True)
    st.title("Complaint Analyzer")
    st.markdown('<div class="page-intro">Paste a customer message below. The classifier will identify the complaint category using the trained Linear SVM model.</div>', unsafe_allow_html=True)

    input_col, guide_col = st.columns([1.55, .75], gap="large")
    with input_col:
        with st.form("complaint_form", clear_on_submit=False):
            complaint = st.text_area(
                "Customer complaint",
                placeholder="Example: I was charged twice for my order and need one of the payments refunded.",
                height=210,
                label_visibility="visible",
            )
            submitted = st.form_submit_button("✦  Classify complaint", type="primary", use_container_width=True)

        if submitted:
            if complaint.strip():
                prediction = model.predict([complaint])[0]
                st.session_state.history.append(
                    {
                        "Complaint": complaint.strip(),
                        "Predicted Category": prediction,
                        "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    }
                )
                st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted category</div>
                        <div class="result-value">{html.escape(str(prediction))}</div>
                        <div class="muted-note" style="margin-top:.7rem">Prediction generated from TF-IDF text features and the Linear SVM classifier.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.toast("Complaint classified successfully", icon="✅")
            else:
                st.warning("Add a complaint before running the classifier.")

    with guide_col:
        st.markdown("### Write a useful complaint")
        st.markdown(
            """
            <div class="card">
                <div class="card-copy">
                    <b>Include the important signal:</b><br><br>
                    • What happened<br>
                    • Which product or service is involved<br>
                    • What outcome the customer wants
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        st.info("The model predicts one of the categories it was trained on. Unknown categories are not currently supported.")


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":
    st.markdown('<div class="eyebrow">Model quality</div>', unsafe_allow_html=True)
    st.title("Model Performance")
    st.markdown('<div class="page-intro">Compare the algorithms evaluated during training and see which one currently performs best.</div>', unsafe_allow_html=True)

    if not comparison_df.empty:
        available_metrics = [col for col in ["Accuracy", "Precision", "Recall", "Macro F1"] if col in comparison_df.columns]
        display_df = comparison_df.copy()
        st.dataframe(
            display_df.style.format({col: "{:.4f}" for col in available_metrics}),
            use_container_width=True,
            hide_index=True,
        )

        chart_cols = st.columns(2, gap="large")
        with chart_cols[0]:
            if "Accuracy" in comparison_df.columns:
                st.markdown("### Accuracy")
                st.bar_chart(comparison_df.set_index("Model")["Accuracy"], color="#3678f6")
        with chart_cols[1]:
            if "Macro F1" in comparison_df.columns:
                st.markdown("### Macro F1")
                st.bar_chart(comparison_df.set_index("Model")["Macro F1"], color="#6f59d9")

        if "Macro F1" in comparison_df.columns:
            best_model_name = comparison_df.sort_values("Macro F1", ascending=False).iloc[0]["Model"]
            st.success(f"Best performing model by Macro F1: {best_model_name}")
    else:
        st.info("Model comparison data is not available yet. Add `results/model_comparison.csv` to populate this page.")


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

elif page == "Confusion Matrix":
    st.markdown('<div class="eyebrow">Error analysis</div>', unsafe_allow_html=True)
    st.title("Confusion Matrix")
    st.markdown('<div class="page-intro">Inspect where the classifier is accurate and which complaint categories are most often confused.</div>', unsafe_allow_html=True)

    if os.path.exists(CONFUSION_PATH):
        st.image(CONFUSION_PATH, caption="Linear SVM · Confusion Matrix", use_container_width=True)
        st.info("Diagonal cells represent correct predictions. Off-diagonal cells represent misclassifications.")
    else:
        st.info("The confusion matrix image is not available yet. Add `results/confusion_matrix.png` to display it here.")


# ==========================================
# 9. CATEGORY ANALYSIS
# ==========================================

elif page == "Category Analysis":
    st.markdown('<div class="eyebrow">Dataset insights</div>', unsafe_allow_html=True)
    st.title("Complaint Categories")
    st.markdown('<div class="page-intro">Understand which complaint types make up the current dataset and where volume is concentrated.</div>', unsafe_allow_html=True)

    if not df.empty and "category" in df.columns:
        category_counts = df["category"].value_counts().rename_axis("Category").reset_index(name="Complaint Count")
        top_category = category_counts.iloc[0]
        insight_cols = st.columns(3, gap="medium")
        with insight_cols[0]:
            metric_card("Categories", category_counts.shape[0])
        with insight_cols[1]:
            metric_card("Largest category", top_category["Category"])
        with insight_cols[2]:
            metric_card("Largest category volume", f"{int(top_category['Complaint Count']):,}")

        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
        chart_col, table_col = st.columns([1.2, .8], gap="large")
        with chart_col:
            st.markdown("### Complaint distribution")
            st.bar_chart(category_counts.set_index("Category"), color="#3678f6")
        with table_col:
            st.markdown("### Category statistics")
            st.dataframe(category_counts, use_container_width=True, hide_index=True)
    else:
        st.info("The cleaned dataset is not available yet. Add `data/processed/cleaned_complaints.csv` to populate this page.")


# ==========================================
# 10. PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":
    st.markdown('<div class="eyebrow">Current session</div>', unsafe_allow_html=True)
    st.title("Prediction History")
    st.markdown('<div class="page-intro">Review the complaints classified since this app session began.</div>', unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.markdown(f'<span class="status-pill">{len(history_df)} prediction(s) this session</span>', unsafe_allow_html=True)
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No complaints have been classified in this session yet. Use Complaint Analyzer to create your first prediction.")


st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="footer">AI Customer Complaint Classifier · Python · Scikit-learn · Streamlit</div>', unsafe_allow_html=True)
