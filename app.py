"""
===================================================================
APP.PY — Streamlit Web Application
===================================================================
Personalized Study Plan Generator using Linear Regression
Modern Educational UI inspired by TextScribe design
Dark navbar, light gradient hero, yellow badges, purple/cyan accents
===================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys
import random

sys.path.insert(0, os.path.dirname(__file__))

from model import (
    load_data, clean_data, encode_features, analyze_features,
    select_features, train_model, evaluate_model, save_artifacts,
    load_artifacts, predict_score, plot_correlation_heatmap,
    plot_actual_vs_predicted, plot_feature_importance
)
from study_plan import generate_study_plan
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StudyGenie — AI Study Plan Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  THEME STATE
# ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# ─────────────────────────────────────────────
#  THEME PALETTE
# ─────────────────────────────────────────────
if is_dark:
    # Dark navy theme
    BG_MAIN = "#0B0E18"
    BG_CARD = "#141929"
    BG_CARD2 = "#1A2035"
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#94A3B8"
    TEXT_MUTED = "#64748B"
    BORDER_COLOR = "rgba(148, 163, 184, 0.12)"
    NAV_BG = "#0F1221"
    SIDEBAR_BG = "linear-gradient(180deg, #0F1221 0%, #141929 100%)"
    HERO_BG = "linear-gradient(135deg, #1a1040 0%, #231555 40%, #1B1145 70%, #0F1221 100%)"
    FORM_BG = "#141929"
    CODE_BG = "rgba(0,0,0,0.4)"
    HOVER_SHADOW = "rgba(124, 58, 237, 0.15)"
else:
    # Light cream/lavender theme (matching reference)
    BG_MAIN = "#F5F3FF"
    BG_CARD = "#FFFFFF"
    BG_CARD2 = "#FAFAFE"
    TEXT_PRIMARY = "#1E1B4B"
    TEXT_SECONDARY = "#475569"
    TEXT_MUTED = "#94A3B8"
    BORDER_COLOR = "rgba(30, 27, 75, 0.08)"
    NAV_BG = "#1E1B3A"
    SIDEBAR_BG = "linear-gradient(180deg, #1E1B3A 0%, #2D2A5E 100%)"
    HERO_BG = "linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 40%, #F3E8FF 70%, #FCE7F3 100%)"
    FORM_BG = "#FFFFFF"
    CODE_BG = "rgba(124, 58, 237, 0.05)"
    HOVER_SHADOW = "rgba(124, 58, 237, 0.08)"

# Shared accent colors
PURPLE = "#7C3AED"
PURPLE_LIGHT = "#A78BFA"
YELLOW = "#EAB308"
YELLOW_BG = "#FEF9C3"
YELLOW_BORDER = "#EAB308"
CYAN = "#06B6D4"
CYAN_BG = "#22D3EE"
GREEN = "#10B981"
GREEN_BG = "rgba(16, 185, 129, 0.1)"
RED = "#EF4444"
RED_BG = "rgba(239, 68, 68, 0.1)"
ORANGE = "#F59E0B"
ORANGE_BG = "rgba(245, 158, 11, 0.1)"

# ─────────────────────────────────────────────
#  INJECT CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    * {{ font-family: 'DM Sans', sans-serif; }}
    code, pre {{ font-family: 'JetBrains Mono', monospace; }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .stApp {{
        background: {BG_MAIN};
    }}

    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* ─── Global Text Color Overrides ─── */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {TEXT_PRIMARY};
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stSlider label, .stSlider p, .stSlider span,
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider [data-testid="stThumbValue"] {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stSelectbox label, .stSelectbox span,
    .stNumberInput label, .stTextInput label {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stRadio label, .stRadio span, .stRadio div {{
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stForm"] label,
    [data-testid="stForm"] p,
    [data-testid="stForm"] span,
    [data-testid="stForm"] h1,
    [data-testid="stForm"] h2,
    [data-testid="stForm"] h3,
    [data-testid="stForm"] h4 {{
        color: {TEXT_PRIMARY} !important;
    }}
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ─── Sidebar (Always dark like TextScribe navbar) ─── */
    [data-testid="stSidebar"] {{
        background: {SIDEBAR_BG};
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    [data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: #FFFFFF !important;
    }}

    /* ─── Yellow Badge (like reference) ─── */
    .ts-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {YELLOW_BG};
        border: 1.5px solid {YELLOW_BORDER};
        color: #92400E !important;
        padding: 6px 18px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-bottom: 20px;
        z-index: 2;
        position: relative;
    }}

    /* ─── Hero Section ─── */
    .ts-hero {{
        background: {HERO_BG};
        border-radius: 20px;
        padding: 48px 44px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        border: 1px solid {BORDER_COLOR};
    }}
    .ts-hero-title {{
        font-size: 2.8rem;
        font-weight: 900;
        color: {TEXT_PRIMARY};
        line-height: 1.15;
        margin-bottom: 16px;
        letter-spacing: -1.5px;
        position: relative;
        z-index: 2;
    }}
    .ts-hero-sub {{
        font-size: 1.05rem;
        color: {TEXT_SECONDARY};
        line-height: 1.75;
        max-width: 520px;
        position: relative;
        z-index: 2;
    }}

    /* ─── Interactive Demo Card (cyan, like reference) ─── */
    .ts-demo-card {{
        background: {CYAN_BG};
        border-radius: 16px;
        padding: 20px;
        position: relative;
        z-index: 2;
        box-shadow: 8px 8px 0px rgba(0,0,0,0.15);
        border: 2px solid rgba(0,0,0,0.1);
    }}
    .ts-demo-dots {{
        display: flex;
        gap: 6px;
        margin-bottom: 12px;
    }}
    .ts-demo-dots span {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }}
    .ts-demo-inner {{
        background: #FFFFFF;
        border: 2px solid rgba(0,0,0,0.15);
        border-radius: 8px;
        padding: 16px;
        min-height: 80px;
        margin-bottom: 12px;
    }}
    .ts-demo-output {{
        background: {GREEN};
        color: white !important;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
    }}

    /* ─── Stat Row ─── */
    .ts-stat {{
        text-align: center;
        padding: 20px 16px;
    }}
    .ts-stat-val {{
        font-size: 1.6rem;
        font-weight: 900;
        color: {TEXT_PRIMARY};
        letter-spacing: -0.5px;
    }}
    .ts-stat-label {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        font-weight: 500;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* ─── Cards ─── */
    .ts-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }}
    .ts-card:hover {{
        box-shadow: 0 8px 30px {HOVER_SHADOW};
        transform: translateY(-3px);
    }}

    /* ─── Feature Step Cards ─── */
    .ts-step {{
        background: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }}
    .ts-step:hover {{
        box-shadow: 0 8px 30px {HOVER_SHADOW};
        transform: translateY(-4px);
        border-color: {PURPLE}30;
    }}
    .ts-step-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 16px;
    }}
    .ts-step-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 10px;
    }}
    .ts-step-desc {{
        font-size: 0.88rem;
        color: {TEXT_SECONDARY};
        line-height: 1.65;
    }}

    /* ─── Purple button (like reference) ─── */
    .stButton > button {{
        background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 36px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }}
    .stButton > button:hover {{
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35) !important;
        transform: translateY(-2px) !important;
    }}

    /* ─── Score Display ─── */
    .ts-score {{
        text-align: center;
        padding: 40px;
        border-radius: 20px;
        margin: 20px 0;
        border: 2px solid;
    }}
    .ts-score-num {{
        font-size: 4.5rem;
        font-weight: 900;
        letter-spacing: -3px;
        line-height: 1;
    }}
    .ts-score-tag {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 700;
        margin-top: 8px;
    }}

    /* ─── Metric boxes ─── */
    .ts-metric {{
        background: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }}
    .ts-metric-val {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {PURPLE};
    }}
    .ts-metric-label {{
        font-size: 0.75rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-top: 6px;
    }}

    /* ─── Section headings ─── */
    .ts-heading {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {TEXT_PRIMARY};
        margin: 32px 0 6px 0;
        letter-spacing: -0.5px;
    }}
    .ts-subheading {{
        font-size: 0.95rem;
        color: {TEXT_SECONDARY};
        margin-bottom: 24px;
    }}

    /* ─── Divider ─── */
    .ts-divider {{
        height: 1px;
        background: {BORDER_COLOR};
        margin: 32px 0;
    }}

    /* ─── Glass panel ─── */
    .ts-glass {{
        background: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 16px;
    }}
    .ts-glass h4 {{
        color: {TEXT_PRIMARY};
        font-weight: 700;
        margin-bottom: 14px;
    }}
    .ts-glass p, .ts-glass li {{
        color: {TEXT_SECONDARY};
        line-height: 1.8;
    }}
    .ts-glass strong {{
        color: {TEXT_PRIMARY};
    }}

    /* ─── Form container ─── */
    .stForm {{
        background: {FORM_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 16px;
        padding: 24px;
    }}

    .stAlert {{
        border-radius: 12px !important;
    }}

    /* ─── Slider ─── */
    .stSlider > div > div > div > div {{
        background-color: {PURPLE};
    }}

    /* ─── Selectbox / Dropdown fix ─── */
    [data-baseweb="select"] {{
        background-color: {BG_CARD} !important;
    }}
    [data-baseweb="select"] > div {{
        background-color: {BG_CARD} !important;
        color: {TEXT_PRIMARY} !important;
        border-color: {BORDER_COLOR} !important;
    }}
    [data-baseweb="select"] span {{
        color: {TEXT_PRIMARY} !important;
    }}
    /* Dropdown menu (opened list) */
    [data-baseweb="popover"] {{
        background-color: {BG_CARD} !important;
    }}
    [data-baseweb="popover"] ul {{
        background-color: {BG_CARD} !important;
    }}
    [data-baseweb="popover"] li {{
        background-color: {BG_CARD} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    [data-baseweb="popover"] li:hover {{
        background-color: {PURPLE}20 !important;
    }}
    [data-baseweb="menu"] {{
        background-color: {BG_CARD} !important;
    }}
    [data-baseweb="menu"] li {{
        color: {TEXT_PRIMARY} !important;
    }}
    [role="option"] {{
        color: {TEXT_PRIMARY} !important;
        background-color: {BG_CARD} !important;
    }}
    [role="option"]:hover {{
        background-color: {PURPLE}15 !important;
    }}
    [aria-selected="true"] {{
        background-color: {PURPLE}20 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def get_model_and_data():
    save_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    model_path = os.path.join(save_dir, "linear_regression_model.pkl")

    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_encoded, encoders = encode_features(df_clean)

    if os.path.exists(model_path):
        model, encoders_saved, features, metrics = load_artifacts(save_dir)
        X = df_encoded[features]
        y = df_encoded["exam_score"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        return model, encoders_saved, features, metrics, df_encoded, y_test, y_pred
    else:
        analyze_features(df_encoded)
        selected = select_features(df_encoded)
        model, X_train, X_test, y_train, y_test = train_model(df_encoded, selected)
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        save_artifacts(model, encoders, selected, metrics)
        return model, encoders, selected, metrics, df_encoded, y_test, y_pred

model, encoders, features, metrics, df_encoded, y_test, y_pred = get_model_and_data()
r2_val = metrics.get("R2_Score", 0)


# ─────────────────────────────────────────────
#  SIDEBAR (Always dark, like TextScribe navbar)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 10px 0 6px 0;">
        <div style="font-size: 1.5rem; font-weight: 900; color: #F9A825 !important;
             letter-spacing: 1px; text-transform: uppercase;">
            STUDYGENIE
        </div>
        <div style="font-size: 0.7rem; color: #94A3B8 !important; letter-spacing: 2px;
             text-transform: uppercase; margin-top: 2px;">
            AI STUDY PLAN GENERATOR
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Predict & Plan", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    theme_icon = "☀️ Light Mode" if is_dark else "🌙 Dark Mode"
    st.button(theme_icon, on_click=toggle_theme, key="theme_toggle")

    st.markdown("---")

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); border-radius: 12px;
                padding: 16px; border: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 0.68rem; color: #64748B !important; letter-spacing: 1.5px;
             text-transform: uppercase; font-weight: 600;">Model Status</div>
        <div style="font-size: 1rem; font-weight: 700; color: #10B981 !important;
             margin: 6px 0;">● Active</div>
        <div style="font-size: 0.8rem; color: #94A3B8 !important;">
            R² Score: <b style="color: #FFFFFF !important;">{r2_val}</b>
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8 !important;">
            Algorithm: <b style="color: #FFFFFF !important;">Linear Regression</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════
if page == "🏠 Home":

    # ── Dynamic demo: pick random student from dataset ──
    demo_study = round(random.uniform(1, 8), 1)
    demo_sleep = round(random.uniform(5, 9), 1)
    demo_social = round(random.uniform(0.5, 5), 1)
    demo_attend = round(random.uniform(60, 98), 0)
    demo_exercise = random.randint(1, 6)
    demo_mental = random.randint(3, 9)
    demo_input = {
        "age": random.randint(18, 25), "gender": random.choice(["Male", "Female"]),
        "study_hours_per_day": demo_study, "social_media_hours": demo_social,
        "netflix_hours": round(random.uniform(0, 4), 1),
        "part_time_job": random.choice(["No", "Yes"]),
        "attendance_percentage": demo_attend, "sleep_hours": demo_sleep,
        "diet_quality": random.choice(["Good", "Fair", "Poor"]),
        "exercise_frequency": demo_exercise,
        "parental_education_level": random.choice(["High School", "Bachelor", "Master"]),
        "internet_quality": random.choice(["Good", "Average"]),
        "mental_health_rating": demo_mental,
        "extracurricular_participation": random.choice(["No", "Yes"]),
    }
    demo_enc = {}
    for k, v in demo_input.items():
        if k in encoders:
            try: demo_enc[k] = encoders[k].transform([str(v)])[0]
            except: demo_enc[k] = 0
        else: demo_enc[k] = v
    demo_score = predict_score(model, features, demo_enc)

    # Hero with two columns
    hero_col1, hero_col2 = st.columns([3, 2])

    with hero_col1:
        st.markdown(f"""
        <div style="padding: 20px 0;">
            <div class="ts-badge">🤖 AI-Powered Study Tool</div>
            <div class="ts-hero-title">
                Transform Your<br>Study Habits Into<br>Better Grades
            </div>
            <div class="ts-hero-sub">
                Analyze your daily habits — study hours, sleep, social media,
                attendance — and get an AI-predicted exam score with a
                <strong style="color: {TEXT_PRIMARY};">personalized study plan</strong>
                tailored just for you.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_col2:
        st.markdown(f"""
        <div class="ts-demo-card">
            <div class="ts-demo-dots">
                <span style="background: #EF4444;"></span>
                <span style="background: #F59E0B;"></span>
                <span style="background: #10B981;"></span>
            </div>
            <div style="font-weight: 700; color: #1E1B4B !important; margin-bottom: 10px;
                        font-size: 0.9rem;">📊 Live Student Analysis</div>
            <div class="ts-demo-inner">
                <div style="font-size: 0.82rem; color: #64748B;">
                    📖 Study: <b>{demo_study}</b> hrs &nbsp;|&nbsp; 😴 Sleep: <b>{demo_sleep}</b> hrs<br>
                    📱 Social Media: <b>{demo_social}</b> hrs &nbsp;|&nbsp; 🏃 Exercise: <b>{demo_exercise}</b> days/wk<br>
                    🏫 Attendance: <b>{int(demo_attend)}%</b> &nbsp;|&nbsp; 🧠 Mental Health: <b>{demo_mental}/10</b>
                </div>
            </div>
            <div class="ts-demo-output">
                🎯 Predicted Score: {demo_score} / 100
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Stats row
    st.markdown(f'<div class="ts-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="ts-stat">
            <div class="ts-stat-val">1000+</div>
            <div class="ts-stat-label">Students Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ts-stat">
            <div class="ts-stat-val">{r2_val}</div>
            <div class="ts-stat-label">R² Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="ts-stat">
            <div class="ts-stat-val">Linear Reg.</div>
            <div class="ts-stat-label">Algorithm</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="ts-stat">
            <div class="ts-stat-val">&lt;1s</div>
            <div class="ts-stat-label">Prediction Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="ts-divider"></div>', unsafe_allow_html=True)

    # How it works
    st.markdown(f"""
    <div class="ts-heading">✨ How It Works</div>
    <div class="ts-subheading">Three simple steps to get your personalized study plan</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="ts-step">
            <div class="ts-step-num" style="background: {YELLOW_BG}; color: #92400E;">1</div>
            <div class="ts-step-title">Input Your Habits</div>
            <div class="ts-step-desc">
                Enter your daily study hours, sleep, social media usage,
                attendance, and other lifestyle habits.
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ts-step">
            <div class="ts-step-num" style="background: #EDE9FE; color: {PURPLE};">2</div>
            <div class="ts-step-title">AI Predicts Score</div>
            <div class="ts-step-desc">
                Linear Regression analyzes your habits and predicts
                your exam score based on 1000+ student data points.
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="ts-step">
            <div class="ts-step-num" style="background: #D1FAE5; color: #065F46;">3</div>
            <div class="ts-step-title">Get Your Plan</div>
            <div class="ts-step-desc">
                Receive a personalized study plan with specific
                recommendations to boost your performance.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="ts-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; padding: 16px 0;">
        <p style="font-size: 1.1rem; color: {TEXT_SECONDARY};">
            Ready? Head to <strong style="color: {PURPLE};">🔮 Predict & Plan</strong> in the sidebar!
        </p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE: PREDICT & PLAN
# ═══════════════════════════════════════════════
elif page == "🔮 Predict & Plan":

    st.markdown(f"""
    <div class="ts-badge">🔮 Prediction Engine</div>
    <div class="ts-heading" style="margin-top: 8px;">Predict Your Score & Get a Study Plan</div>
    <div class="ts-subheading">Enter your daily habits below and let AI generate your personalized plan</div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("#### 📊 Your Daily Habits")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("📖 Study Hours Per Day", 0.0, 12.0, 3.0, 0.5)
            sleep_hours = st.slider("😴 Sleep Hours Per Night", 2.0, 12.0, 7.0, 0.5)
            social_media = st.slider("📱 Social Media Hours", 0.0, 8.0, 2.0, 0.5)
            netflix_hours = st.slider("📺 Entertainment Hours", 0.0, 6.0, 1.0, 0.5)
            attendance = st.slider("🏫 Attendance %", 30.0, 100.0, 80.0, 1.0)

        with col2:
            age = st.slider("🎂 Age", 16, 30, 20)
            exercise = st.slider("🏃 Exercise (days/week)", 0, 7, 3)
            mental_health = st.slider("🧠 Mental Health (1-10)", 1, 10, 5)
            gender = st.selectbox("👤 Gender", ["Male", "Female", "Other"])
            part_time = st.selectbox("💼 Part-time Job?", ["No", "Yes"])
            diet = st.selectbox("🥗 Diet Quality", ["Good", "Fair", "Poor"])
            internet = st.selectbox("🌐 Internet Quality", ["Good", "Average", "Poor"])
            parent_edu = st.selectbox("🎓 Parental Education", ["High School", "Bachelor", "Master", "None"])
            extra = st.selectbox("🎭 Extracurricular?", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict My Score & Generate Plan")

    if submitted:
        student_raw = {
            "age": age, "gender": gender,
            "study_hours_per_day": study_hours,
            "social_media_hours": social_media,
            "netflix_hours": netflix_hours,
            "part_time_job": part_time,
            "attendance_percentage": attendance,
            "sleep_hours": sleep_hours,
            "diet_quality": diet,
            "exercise_frequency": exercise,
            "parental_education_level": parent_edu,
            "internet_quality": internet,
            "mental_health_rating": mental_health,
            "extracurricular_participation": extra,
        }

        student_encoded = {}
        for key, val in student_raw.items():
            if key in encoders:
                try:
                    student_encoded[key] = encoders[key].transform([str(val)])[0]
                except ValueError:
                    student_encoded[key] = 0
            else:
                student_encoded[key] = val

        predicted = predict_score(model, features, student_encoded)
        plan = generate_study_plan(predicted, student_raw)

        st.markdown("---")

        # Score
        tier = plan["tier"]
        tc = plan["tier_color"]
        if tier == "intensive":
            sbg = f"{'#1a0505' if is_dark else '#FEF2F2'}"
        elif tier == "moderate":
            sbg = f"{'#1a1505' if is_dark else '#FFFBEB'}"
        else:
            sbg = f"{'#051a0f' if is_dark else '#ECFDF5'}"

        st.markdown(f"""
        <div class="ts-score" style="background: {sbg}; border-color: {tc}30;">
            <div class="ts-score-tag" style="color: {tc};">PREDICTED EXAM SCORE</div>
            <div class="ts-score-num" style="color: {tc};">{predicted}</div>
            <div class="ts-score-tag" style="color: {TEXT_MUTED};">out of 100</div>
            <div style="margin-top: 14px;">
                <span class="ts-badge" style="background: {tc}12; color: {tc}; border-color: {tc}40;">
                    {plan['tier_label']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary
        st.info(plan['summary'])

        # ─── CHARTS: Habit Analysis ───
        st.markdown("### 📊 Your Habit Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Bar chart: Your habits vs ideal
            habit_labels = ['Study\nHours', 'Sleep\nHours', 'Social\nMedia', 'Netflix\nHours',
                            'Attendance\n(%÷10)', 'Exercise\n(days)', 'Mental\nHealth']
            your_vals = [study_hours, sleep_hours, social_media, netflix_hours,
                         attendance / 10, exercise, mental_health]
            ideal_vals = [5, 7.5, 1.5, 1.0, 8.5, 4, 8]

            fig_habits, ax_habits = plt.subplots(figsize=(8, 4.5))
            fig_habits.patch.set_facecolor('#0B0E18' if is_dark else '#F5F3FF')
            ax_habits.set_facecolor('#0B0E18' if is_dark else '#F5F3FF')

            x_pos = np.arange(len(habit_labels))
            width = 0.35
            bars1 = ax_habits.bar(x_pos - width/2, your_vals, width, label='Your Habits',
                                  color='#7C3AED')
            bars2 = ax_habits.bar(x_pos + width/2, ideal_vals, width, label='Ideal Range',
                                  color='#22D3EE', alpha=0.7)

            ax_habits.set_xticks(x_pos)
            ax_habits.set_xticklabels(habit_labels, fontsize=8,
                                      color='#E2E8F0' if is_dark else '#1E1B4B')
            ax_habits.tick_params(axis='y', colors='#94A3B8' if is_dark else '#475569')
            ax_habits.legend(fontsize=9, facecolor='#141929' if is_dark else '#FFFFFF',
                             edgecolor='none',
                             labelcolor='#E2E8F0' if is_dark else '#1E1B4B')
            ax_habits.set_title('Your Habits vs Ideal', fontsize=13, fontweight='bold',
                                color='#E2E8F0' if is_dark else '#1E1B4B', pad=12)
            ax_habits.spines['top'].set_visible(False)
            ax_habits.spines['right'].set_visible(False)
            ax_habits.spines['left'].set_color('#333' if is_dark else '#DDD')
            ax_habits.spines['bottom'].set_color('#333' if is_dark else '#DDD')
            plt.tight_layout()
            st.pyplot(fig_habits)
            plt.close(fig_habits)

        with chart_col2:
            # Donut chart: Time distribution
            time_data = {
                'Study': study_hours,
                'Sleep': sleep_hours,
                'Social Media': social_media,
                'Entertainment': netflix_hours,
                'Other': max(0, 24 - study_hours - sleep_hours - social_media - netflix_hours - 8)
            }
            labels_pie = list(time_data.keys())
            sizes_pie = list(time_data.values())
            colors_pie = ['#7C3AED', '#22D3EE', '#F59E0B', '#EF4444', '#6B7280']

            fig_pie, ax_pie = plt.subplots(figsize=(8, 4.5))
            fig_pie.patch.set_facecolor('#0B0E18' if is_dark else '#F5F3FF')

            wedges, texts, autotexts = ax_pie.pie(
                sizes_pie, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%',
                startangle=90, pctdistance=0.78,
                wedgeprops=dict(width=0.45, edgecolor='#0B0E18' if is_dark else '#F5F3FF',
                                linewidth=2)
            )
            for t in texts:
                t.set_color('#E2E8F0' if is_dark else '#1E1B4B')
                t.set_fontsize(9)
            for t in autotexts:
                t.set_color('#FFFFFF')
                t.set_fontsize(8)
                t.set_fontweight('bold')
            ax_pie.set_title('Your 24-Hour Time Distribution', fontsize=13, fontweight='bold',
                              color='#E2E8F0' if is_dark else '#1E1B4B', pad=12)
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        st.markdown("---")

        # ─── Schedule (Table) + Goals ───
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### 📅 Recommended Daily Schedule")
            schedule_data = []
            for time_slot, activity in plan["study_schedule"].items():
                schedule_data.append({"⏰ Time Slot": time_slot, "📝 Activity": activity})
            schedule_df = pd.DataFrame(schedule_data)
            st.dataframe(
                schedule_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "⏰ Time Slot": st.column_config.TextColumn(width="medium"),
                    "📝 Activity": st.column_config.TextColumn(width="large"),
                }
            )

        with col2:
            st.markdown("### 🎯 This Week's Goals")
            if plan["weekly_goals"]:
                for i, goal in enumerate(plan["weekly_goals"], 1):
                    st.markdown(f"**{i}.** {goal}")
            else:
                st.success("✅ Your habits are great! Keep it up!")

        # ─── Habit Analysis Cards (like reference screenshot) ───
        st.markdown(f"""
        <div class="ts-heading">📋 Habit Analysis</div>
        """, unsafe_allow_html=True)

        # Count statuses
        critical_count = sum(1 for r in plan["recommendations"] if r["status"] == "critical")
        warning_count = sum(1 for r in plan["recommendations"] if r["status"] == "warning")
        good_count = sum(1 for r in plan["recommendations"] if r["status"] == "good")

        # Status summary row
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""
            <div style="padding: 12px 0;">
                <span style="color: #EF4444; font-size: 0.85rem; font-weight: 600;">
                    🔴 Critical Issues
                </span>
                <div style="font-size: 2rem; font-weight: 900; color: {TEXT_PRIMARY}; margin-top: 4px;">
                    {critical_count}
                </div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div style="padding: 12px 0;">
                <span style="color: #F59E0B; font-size: 0.85rem; font-weight: 600;">
                    🟡 Needs Improvement
                </span>
                <div style="font-size: 2rem; font-weight: 900; color: {TEXT_PRIMARY}; margin-top: 4px;">
                    {warning_count}
                </div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div style="padding: 12px 0;">
                <span style="color: #10B981; font-size: 0.85rem; font-weight: 600;">
                    🟢 Going Well
                </span>
                <div style="font-size: 2rem; font-weight: 900; color: {TEXT_PRIMARY}; margin-top: 4px;">
                    {good_count}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Individual recommendation cards
        for rec in plan["recommendations"]:
            status = rec["status"]

            if status == "critical":
                dot_color = "#EF4444"
                border_color = "#EF4444"
                card_bg = "#1a0505" if is_dark else "#FEF2F2"
            elif status == "warning":
                dot_color = "#F59E0B"
                border_color = "#F59E0B"
                card_bg = "#1a1505" if is_dark else "#FFFBEB"
            else:
                dot_color = "#10B981"
                border_color = "#10B981"
                card_bg = "#051a0f" if is_dark else "#ECFDF5"

            st.markdown(f"""
            <div style="
                background: {card_bg};
                border-left: 4px solid {border_color};
                border-radius: 12px;
                padding: 20px 24px;
                margin-bottom: 14px;
                border: 1px solid {border_color}25;
                border-left: 4px solid {border_color};
            ">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <span style="color: {dot_color}; font-size: 1.4rem;">●</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: {dot_color};">
                        {rec['icon']} {rec['area']}
                    </span>
                </div>
                <div style="font-size: 0.88rem; color: {TEXT_SECONDARY}; margin-bottom: 8px;">
                    Current: <strong style="color: {TEXT_PRIMARY};">{rec['current']}</strong>
                    &nbsp;|&nbsp;
                    Recommended: <strong style="color: {dot_color};">{rec['target']}</strong>
                </div>
                <div style="font-size: 0.88rem; color: {TEXT_SECONDARY}; line-height: 1.6;">
                    {rec['tip']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.success(plan['motivation'])


# ═══════════════════════════════════════════════
#  PAGE: MODEL INSIGHTS
# ═══════════════════════════════════════════════
elif page == "📊 Model Insights":

    st.markdown(f"""
    <div class="ts-badge">📊 Analytics Dashboard</div>
    <div class="ts-heading" style="margin-top: 8px;">Model Performance & Insights</div>
    <div class="ts-subheading">Understand how the Linear Regression model works and performs</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="ts-metric"><div class="ts-metric-val">{metrics.get("R2_Score","N/A")}</div><div class="ts-metric-label">R² Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="ts-metric"><div class="ts-metric-val">{metrics.get("MAE","N/A")}</div><div class="ts-metric-label">MAE</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="ts-metric"><div class="ts-metric-val">{metrics.get("MSE","N/A")}</div><div class="ts-metric-label">MSE</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="ts-metric"><div class="ts-metric-val">{metrics.get("RMSE","N/A")}</div><div class="ts-metric-label">RMSE</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="ts-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ts-glass">
        <h4>📖 Understanding the Metrics</h4>
        <ul>
            <li><strong>R² Score</strong>: How well the model explains variance (closer to 1.0 = better)</li>
            <li><strong>MAE</strong>: Average prediction error in marks (lower = better)</li>
            <li><strong>MSE</strong>: Penalizes large errors more heavily</li>
            <li><strong>RMSE</strong>: Error in same unit as scores — easy to interpret</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="ts-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Actual vs Predicted", "🔥 Feature Importance", "🗺️ Correlation Heatmap"])
    with tab1:
        st.pyplot(plot_actual_vs_predicted(y_test, y_pred))
        st.info("Each dot = a student. Closer to the red line = more accurate prediction.")
    with tab2:
        st.pyplot(plot_feature_importance(model, features))
        st.info("🟢 Green = increases score | 🔴 Red = decreases score. Longer bar = stronger effect.")
    with tab3:
        st.pyplot(plot_correlation_heatmap(df_encoded))
        st.info("Check the 'exam_score' row to see which habits matter most.")

    st.markdown(f'<div class="ts-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ts-glass">
        <h4>🤔 Why Linear Regression?</h4>
        <p><strong>1. Interpretability:</strong> Each coefficient shows exactly how much each habit affects scores.</p>
        <p><strong>2. Continuous Target:</strong> Exam scores (0-100) are continuous — perfect for regression.</p>
        <p><strong>3. Linear Relationships:</strong> Study hours & attendance have ~linear effects.</p>
        <p><strong>4. Simplicity:</strong> Easy to train, interpret, and deploy.</p>
        <p><strong>5. Speed:</strong> Trains instantly, predicts in milliseconds.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE: ABOUT
# ═══════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown(f"""
    <div class="ts-badge">ℹ️ About This Project</div>
    <div class="ts-heading" style="margin-top: 8px;">Personalized Study Plan Generator</div>
    <div class="ts-subheading">AI/ML Mini Project — BCA 4th Semester</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ts-glass">
        <h4>🎯 Project Overview</h4>
        <p>This <strong>AI/ML Mini Project</strong> analyzes student habits and predicts academic
        performance using <strong>Linear Regression</strong>. Based on the prediction, it generates
        a <strong>personalized study plan</strong> with actionable recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="ts-glass">
            <h4>🏗️ System Architecture</h4>
            <p><strong>1. Data Layer</strong> — CSV dataset, Pandas</p>
            <p><strong>2. ML Layer</strong> — Scikit-learn, Linear Regression</p>
            <p><strong>3. Logic Layer</strong> — Study plan algorithm</p>
            <p><strong>4. UI Layer</strong> — Streamlit, Light/Dark themes</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="ts-glass">
            <h4>🛠️ Tech Stack</h4>
            <p>🐍 <strong>Python 3.x</strong></p>
            <p>🐼 <strong>Pandas & NumPy</strong></p>
            <p>🤖 <strong>Scikit-learn</strong></p>
            <p>📊 <strong>Matplotlib & Seaborn</strong></p>
            <p>🌐 <strong>Streamlit</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ts-glass">
        <h4>📁 Project Structure</h4>
        <pre style="color: {PURPLE}; background: {CODE_BG};
             padding: 20px; border-radius: 12px; font-size: 0.85rem;">
anti/
├── dataset/
│   └── student_habits_performance.csv
├── saved_model/
│   ├── linear_regression_model.pkl
│   ├── encoders.pkl, features.pkl, metrics.pkl
├── .streamlit/config.toml
├── model.py          # ML pipeline
├── study_plan.py     # Plan generator
├── app.py            # Streamlit UI
├── requirements.txt
└── README.md
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ts-glass">
        <h4>🚀 Future Improvements</h4>
        <p>📈 Advanced models (Random Forest, XGBoost)</p>
        <p>🔐 User authentication & history</p>
        <p>📱 Mobile-responsive PWA</p>
        <p>🗃️ Database integration</p>
        <p>🏆 Gamification (badges, streaks)</p>
    </div>
    """, unsafe_allow_html=True)
