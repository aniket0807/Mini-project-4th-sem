import sys, os
# ── Fix Windows cp1252 UnicodeEncodeError from emoji in print() calls ──
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import streamlit as st, pandas as pd, numpy as np, matplotlib, random, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from model import (load_data, clean_data, encode_features, analyze_features,
    select_features, train_xgboost, evaluate_model, save_artifacts,
    load_artifacts, predict_score, plot_correlation_heatmap,
    plot_actual_vs_predicted, plot_feature_importance, SAVE_DIR)
from study_plan import generate_study_plan
from spaced_repetition import (init_db as sr_init_db, add_topic, get_due_items,
    get_all_items, update_review, get_retention_features, delete_topic)
from gamification import (init_db as gam_init_db, get_or_create_user,
    log_study_session, log_review_xp, get_user_stats, BADGE_DEFINITIONS, XP_PER_LEVEL)
from database import get_db_path
from auth import (init_auth_db, register_user, login_user,
    session_login, session_logout, get_session_user, is_authenticated)
from syllabus_parser import parse_syllabus_pdf
from email_service import send_study_plan_email
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="StudyGenie", page_icon="🎓",
    layout="wide", initial_sidebar_state="expanded")

for k,v in {"theme":"dark","student_name":"","authenticated":False,"auth_user":None}.items():
    if k not in st.session_state: st.session_state[k]=v

def toggle_theme():
    st.session_state.theme="light" if st.session_state.theme=="dark" else "dark"

D=st.session_state.theme=="dark"
BG="#0B0E18" if D else "#F5F3FF"
CARD="#141929" if D else "#FFFFFF"
CARD2="#1A2035" if D else "#FAFAFE"
TP="#FFFFFF" if D else "#1E1B4B"
TS="#94A3B8" if D else "#475569"
TM="#64748B" if D else "#94A3B8"
BC="rgba(148,163,184,0.12)" if D else "rgba(30,27,75,0.08)"
SBGR="linear-gradient(180deg,#0F1221 0%,#141929 100%)" if D else "linear-gradient(180deg,#1E1B3A 0%,#2D2A5E 100%)"
HERO="linear-gradient(135deg,#1a1040,#231555,#0F1221)" if D else "linear-gradient(135deg,#F5F3FF,#EDE9FE,#FCE7F3)"
FORM="#141929" if D else "#FFFFFF"
HS="rgba(124,58,237,0.15)" if D else "rgba(124,58,237,0.08)"
PU="#7C3AED"; PL="#A78BFA"; CY="#06B6D4"; GR="#10B981"; RE="#EF4444"; OR="#F59E0B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;900&display=swap');
*{{font-family:'DM Sans',sans-serif;}} #MainMenu,footer{{visibility:hidden;}}
.stApp{{background:{BG};}}
.stApp,.stApp p,.stApp span,.stApp label,.stApp div{{color:{TP};}}
.stMarkdown,.stMarkdown p,.stMarkdown li,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{{color:{TP}!important;}}
.stSlider label,.stSlider p,.stSlider span{{color:{TP}!important;}}
.stSelectbox label,.stNumberInput label,.stTextInput label,.stRadio label{{color:{TP}!important;}}
[data-testid="stSidebar"]{{background:{SBGR};border-right:1px solid rgba(255,255,255,0.06);}}
[data-testid="stSidebar"] *{{color:#E2E8F0!important;}}
.stButton>button{{background:linear-gradient(135deg,#7C3AED,#6D28D9)!important;color:white!important;
  border:none!important;border-radius:10px!important;padding:10px 24px!important;
  font-weight:700!important;transition:all 0.3s!important;font-size:0.85rem!important;}}
.stButton>button:hover{{box-shadow:0 6px 20px rgba(124,58,237,0.35)!important;transform:translateY(-2px)!important;}}
[data-testid="stForm"] .stButton>button{{width:100%!important;padding:12px 28px!important;}}
[data-testid="stSidebar"] .stButton>button{{width:100%!important;}}
.card{{background:{CARD};border:1px solid {BC};border-radius:16px;padding:24px;margin-bottom:16px;transition:all 0.3s;}}
.card:hover{{box-shadow:0 8px 30px {HS};transform:translateY(-2px);}}
.metric{{background:{CARD};border:1px solid {BC};border-radius:14px;padding:20px;text-align:center;}}
.metric-val{{font-size:1.8rem;font-weight:800;color:{PU};}}
.metric-lbl{{font-size:0.75rem;color:{TM};text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-top:6px;}}
.badge{{display:inline-flex;align-items:center;gap:6px;background:#FEF9C3;border:1.5px solid #EAB308;
  color:#92400E!important;padding:5px 16px;border-radius:6px;font-size:0.8rem;font-weight:700;margin-bottom:16px;}}
.xp-outer{{background:{CARD2};border:1px solid {BC};border-radius:999px;height:12px;overflow:hidden;margin:8px 0;}}
.xp-inner{{height:100%;border-radius:999px;background:linear-gradient(90deg,{PU},{CY});transition:width 0.5s;}}
.glass{{background:{CARD};border:1px solid {BC};border-radius:16px;padding:24px;margin-bottom:16px;}}
.glass h4{{color:{TP};font-weight:700;margin-bottom:12px;}}
.glass p,.glass li{{color:{TS};line-height:1.8;}}
.review-card{{background:{CARD};border:1px solid {BC};border-radius:16px;padding:18px 22px;margin-bottom:12px;}}
[data-baseweb="select"]>div{{background:{CARD}!important;color:{TP}!important;border-color:{BC}!important;}}
[data-baseweb="popover"] li{{background:{CARD}!important;color:{TP}!important;}}
</style>
""", unsafe_allow_html=True)

DB_PATH=get_db_path()
sr_init_db(DB_PATH); gam_init_db(DB_PATH); init_auth_db(DB_PATH)

@st.cache_resource(show_spinner="🤖 Loading model…")
def get_model():
    xp=os.path.join(SAVE_DIR,"xgboost_model.pkl")
    df_raw=load_data(); df_clean=clean_data(df_raw)
    df_enc,encoders=encode_features(df_clean)
    if os.path.exists(xp):
        model,_,features,metrics=load_artifacts(SAVE_DIR)
        X=df_clean[features]; y=df_clean["exam_score"]
        _,Xt,_,yt=train_test_split(X,y,test_size=0.2,random_state=42)
        yp=np.clip(model.predict(Xt),0,100)
    else:
        analyze_features(df_enc); sel=select_features(df_enc)
        pipeline,_,Xt,_,yt,_,_=train_xgboost(df_clean,sel,n_trials=5)
        metrics,yp=evaluate_model(pipeline,Xt,yt,None)
        save_artifacts(pipeline,encoders,sel,metrics)
        model=pipeline; features=sel
    lr_m=None
    lrp=os.path.join(SAVE_DIR,"lr_metrics_baseline.pkl")
    if os.path.exists(lrp):
        try: lr_m=joblib.load(lrp)
        except: pass
    return model,encoders,features,metrics,df_enc,yt,yp,lr_m

model,encoders,features,metrics,df_enc,y_test,y_pred,lr_metrics=get_model()
r2=metrics.get("R2_Score",0)

def cur_user():
    au=get_session_user()
    if not au: return None
    return get_or_create_user(au["display_name"],DB_PATH)

# ═══════════ AUTH WALL ═══════════
if not is_authenticated():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp { background: #080b14 !important; }
    [data-testid="stHeader"] { display: none !important; }
    .main .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; max-width: 100% !important; }

    .auth-bg { position:fixed;top:0;left:0;width:100vw;height:100vh;overflow:hidden;pointer-events:none;z-index:0; }
    .orb { position:absolute;border-radius:50%;filter:blur(90px);opacity:0.2;animation:floatOrb 12s ease-in-out infinite alternate; }
    .orb1 { width:500px;height:500px;background:radial-gradient(circle,#7C3AED,#4F46E5);top:-140px;left:-140px;animation-duration:14s; }
    .orb2 { width:380px;height:380px;background:radial-gradient(circle,#06B6D4,#0EA5E9);bottom:-80px;right:-60px;animation-duration:11s;animation-delay:-3s; }
    .orb3 { width:260px;height:260px;background:radial-gradient(circle,#EC4899,#8B5CF6);top:35%;left:48%;animation-duration:16s;animation-delay:-7s; }
    @keyframes floatOrb { 0%{transform:translate(0,0) scale(1);} 100%{transform:translate(35px,25px) scale(1.1);} }

    .auth-grid { position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;
        background-image:linear-gradient(rgba(124,58,237,0.035) 1px,transparent 1px),linear-gradient(90deg,rgba(124,58,237,0.035) 1px,transparent 1px);
        background-size:44px 44px; }

    .auth-logo-icon { width:54px;height:54px;border-radius:15px;background:linear-gradient(135deg,#7C3AED,#06B6D4);
        display:flex;align-items:center;justify-content:center;font-size:1.65rem;margin-bottom:12px;
        box-shadow:0 0 28px rgba(124,58,237,0.5);animation:pulseGlow 3s ease-in-out infinite alternate; }
    @keyframes pulseGlow { from{box-shadow:0 0 16px rgba(124,58,237,0.35);} to{box-shadow:0 0 38px rgba(124,58,237,0.65);} }
    .auth-logo-title { font-family:'Inter',sans-serif;font-size:2.3rem;font-weight:800;
        background:linear-gradient(135deg,#A78BFA 0%,#38BDF8 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
        letter-spacing:-1.5px;line-height:1.1;margin-bottom:7px; }
    .auth-logo-sub { font-family:'Inter',sans-serif;font-size:0.78rem;color:#64748B;
        letter-spacing:2px;text-transform:uppercase;font-weight:500;margin-bottom:18px; }
    .auth-tagline { font-family:'Inter',sans-serif;font-size:0.93rem;color:#94A3B8;line-height:1.7;margin-bottom:20px; }
    .auth-pill { display:inline-block;background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.28);
        border-radius:99px;padding:4px 11px;font-size:0.7rem;color:#A78BFA;font-weight:600;
        font-family:'Inter',sans-serif;margin:3px 3px 3px 0; }

    .auth-card-line { height:2px;border-radius:2px;
        background:linear-gradient(90deg,transparent,#7C3AED,#06B6D4,transparent);margin-bottom:16px; }
    .auth-heading { font-family:'Inter',sans-serif;font-size:1.22rem;font-weight:700;
        color:#F1F5F9;margin-bottom:3px;letter-spacing:-0.4px; }
    .auth-sub { font-family:'Inter',sans-serif;font-size:0.77rem;color:#64748B;margin-bottom:12px; }

    .stTextInput input { background:rgba(255,255,255,0.04) !important;
        border:1.5px solid rgba(148,163,184,0.14) !important;border-radius:10px !important;
        color:#F1F5F9 !important;padding:9px 13px !important;font-size:0.86rem !important;
        font-family:'Inter',sans-serif !important;transition:all 0.2s !important; }
    .stTextInput input:focus { border-color:rgba(124,58,237,0.65) !important;
        box-shadow:0 0 0 3px rgba(124,58,237,0.13) !important;
        background:rgba(124,58,237,0.05) !important;outline:none !important; }
    .stTextInput input::placeholder { color:#3D4A5C !important; }
    .stTextInput label { font-family:'Inter',sans-serif !important;font-size:0.72rem !important;
        font-weight:600 !important;color:#94A3B8 !important;text-transform:uppercase !important;
        letter-spacing:0.8px !important;margin-bottom:3px !important; }

    [data-testid="stForm"] .stButton > button { background:linear-gradient(135deg,#7C3AED 0%,#4F46E5 60%,#06B6D4 100%) !important;
        border:none !important;border-radius:10px !important;color:#fff !important;font-weight:700 !important;
        font-size:0.87rem !important;padding:10px 22px !important;width:100% !important;
        font-family:'Inter',sans-serif !important;box-shadow:0 4px 18px rgba(124,58,237,0.32) !important;
        transition:all 0.25s !important;margin-top:4px !important; }
    [data-testid="stForm"] .stButton > button:hover { transform:translateY(-2px) !important;
        box-shadow:0 8px 26px rgba(124,58,237,0.52) !important;filter:brightness(1.08) !important; }

    .stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.04) !important;
        border-radius:10px !important;padding:3px !important;gap:3px !important;
        border-bottom:none !important;margin-bottom:12px !important; }
    .stTabs [data-baseweb="tab"] { border-radius:7px !important;padding:7px 14px !important;
        font-weight:600 !important;font-size:0.79rem !important;color:#64748B !important;
        background:transparent !important;border:none !important;
        font-family:'Inter',sans-serif !important;transition:all 0.22s !important; }
    .stTabs [aria-selected="true"] { background:linear-gradient(135deg,#7C3AED,#6D28D9) !important;
        color:#fff !important;box-shadow:0 3px 12px rgba(124,58,237,0.38) !important; }
    .stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] { display:none !important; }
    .stAlert { border-radius:10px !important;font-family:'Inter',sans-serif !important; }
    </style>
    <div class="auth-bg"><div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div></div>
    <div class="auth-grid"></div>
    """, unsafe_allow_html=True)

    col_brand, col_form = st.columns([1.1, 1])

    with col_brand:
        st.markdown("""
        <div style="display:flex;flex-direction:column;justify-content:center;
                    min-height:88vh;padding:10px 20px 10px 6px;position:relative;z-index:10;">
            <div class="auth-logo-icon">🎓</div>
            <div class="auth-logo-title">StudyGenie</div>
            <div class="auth-logo-sub">AI-Powered Study Intelligence</div>
            <div class="auth-tagline">
                Analyse your study habits, get an XGBoost-predicted exam score,
                a personalised study plan and gamified achievements — all in one place.
            </div>
            <div>
                <span class="auth-pill">⚡ XGBoost Predictions</span>
                <span class="auth-pill">🧠 Spaced Repetition</span>
                <span class="auth-pill">🏆 Gamified Learning</span>
                <span class="auth-pill">📄 Syllabus Parser</span>
                <span class="auth-pill">📧 Email Plans</span>
            </div>
            <div style="margin-top:20px;font-size:0.72rem;color:#334155;font-family:'Inter',sans-serif;">
                🔒 Data stored locally &nbsp;·&nbsp; No third-party tracking
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown("""<div style="position:relative;z-index:10;padding-top:2px;">
        <div class="auth-card-line"></div></div>""", unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔑 Sign In", "✨ Create Account"])

        with tab_login:
            st.markdown("""<div class="auth-heading">Welcome back! 👋</div>
            <div class="auth-sub">Sign in to continue your learning journey</div>""", unsafe_allow_html=True)
            with st.form("login_form"):
                lemail = st.text_input("Email Address", placeholder="you@example.com", key="l_email")
                lpw    = st.text_input("Password", type="password", placeholder="Enter your password", key="l_pw")
                lsub   = st.form_submit_button("✶ Sign In — Let's Study!")
            if lsub:
                res = login_user(lemail, lpw, DB_PATH)
                if res["success"]:
                    session_login(res["user"])
                    st.success(f"Welcome back, **{res['user']['display_name']}**! 👋")
                    st.rerun()
                else:
                    st.error(f"🚫 {res['error']}")

        with tab_reg:
            st.markdown("""<div class="auth-heading">Create your account ✨</div>
            <div class="auth-sub">Join thousands of students boosting their grades</div>""", unsafe_allow_html=True)
            with st.form("reg_form"):
                rname  = st.text_input("Display Name", placeholder="e.g. Aniket Sharma", key="r_name")
                remail = st.text_input("Email Address", placeholder="you@example.com", key="r_email")
                rpw    = st.text_input("Password", type="password", placeholder="Min. 6 characters", key="r_pw")
                rpw2   = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="r_pw2")
                rsub   = st.form_submit_button("🚀 Create My Account")
            if rsub:
                if rpw != rpw2:
                    st.error("🚫 Passwords do not match. Please try again.")
                else:
                    res = register_user(remail, rname, rpw, DB_PATH)
                    if res["success"]:
                        session_login(res["user"])
                        st.success(f"🎉 Welcome to StudyGenie, **{rname}**!")
                        st.rerun()
                    else:
                        st.error(f"🚫 {res['error']}")

    st.stop()

# ═══════════ SIDEBAR ═══════════
auth_user = get_session_user()
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 4px;">
      <div style="font-size:1.4rem;font-weight:900;color:#F9A825;">STUDYGENIE</div>
      <div style="font-size:0.65rem;color:#94A3B8;letter-spacing:2px;">XGBoost AI ENGINE</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    u=cur_user()
    if u:
        _s=get_user_stats(u["id"],DB_PATH)
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:10px 14px;
                    border:1px solid rgba(255,255,255,0.08);margin-bottom:8px;">
          <div style="font-weight:700;color:#F9A825;font-size:0.95rem;">👤 {auth_user["display_name"]}</div>
          <div style="font-size:0.78rem;color:#94A3B8;">{auth_user["email"]}</div>
          <div style="display:flex;gap:12px;margin-top:8px;">
            <span style="color:#F59E0B;font-size:0.82rem;">🔥 {_s.get("current_streak",0)}d</span>
            <span style="color:#A78BFA;font-size:0.82rem;">⚡ {_s.get("xp",0):,} XP</span>
            <span style="color:#06B6D4;font-size:0.82rem;">Lv {_s.get("level",1)}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page=st.radio("Navigate",["🏠 Home","🔮 Predict & Plan","🧠 Review Schedule",
        "🏆 Achievements","📊 Model Insights","ℹ️ About"],label_visibility="collapsed")
    st.markdown("---")
    ti="☀️ Light Mode" if D else "🌙 Dark Mode"
    st.button(ti,on_click=toggle_theme,key="theme_btn")
    st.markdown("---")
    if st.button("🚪 Sign Out",key="signout_btn"):
        session_logout(); st.rerun()
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px;
                border:1px solid rgba(255,255,255,0.06);margin-top:8px;">
      <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;">Model</div>
      <div style="font-size:0.9rem;font-weight:700;color:#10B981;margin:4px 0;">● Active</div>
      <div style="font-size:0.78rem;color:#94A3B8;">R²: <b style="color:#fff;">{r2}</b></div>
    </div>""", unsafe_allow_html=True)

# ═══════════ HOME PAGE ═══════════
if page == "🏠 Home":
    d_study=round(__import__("random").uniform(1,8),1)
    d_sleep=round(__import__("random").uniform(5,9),1)
    d_social=round(__import__("random").uniform(0.5,5),1)
    d_att=round(__import__("random").uniform(60,98),0)
    d_ex=__import__("random").randint(1,6); d_mh=__import__("random").randint(3,9)
    d_in={"age":20,"gender":"Male","study_hours_per_day":d_study,"social_media_hours":d_social,
          "netflix_hours":1.0,"part_time_job":"No","attendance_percentage":d_att,
          "sleep_hours":d_sleep,"diet_quality":"Good","exercise_frequency":d_ex,
          "parental_education_level":"Bachelor","internet_quality":"Good",
          "mental_health_rating":d_mh,"extracurricular_participation":"No"}
    try: ds=predict_score(model,features,d_in)
    except: ds="—"

    import streamlit as st2; st2=st
    c1,c2=st.columns([3,2])
    with c1:
        st.markdown(f"""
        <div class="badge">🤖 XGBoost-Powered Study Tool</div>
        <div style="font-size:2.6rem;font-weight:900;color:{TP};line-height:1.15;
                    letter-spacing:-1.5px;margin-bottom:14px;">
          Transform Your<br>Study Habits Into<br>Better Grades
        </div>
        <div style="font-size:1rem;color:{TS};line-height:1.75;max-width:500px;">
          Analyse habits, get an XGBoost-predicted exam score, a personalised study plan,
          spaced-repetition schedule, and gamified achievements — all in one place.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:#22D3EE;border-radius:16px;padding:20px;
                    box-shadow:8px 8px 0 rgba(0,0,0,0.15);border:2px solid rgba(0,0,0,0.1);">
          <div style="display:flex;gap:6px;margin-bottom:12px;">
            <span style="width:12px;height:12px;border-radius:50%;background:#EF4444;display:inline-block;"></span>
            <span style="width:12px;height:12px;border-radius:50%;background:#F59E0B;display:inline-block;"></span>
            <span style="width:12px;height:12px;border-radius:50%;background:#10B981;display:inline-block;"></span>
          </div>
          <div style="font-weight:700;color:#1E1B4B;margin-bottom:10px;">📊 Live Analysis</div>
          <div style="background:#fff;border-radius:8px;padding:14px;font-size:0.82rem;color:#64748B;">
            📖 Study: <b>{d_study}</b>h &nbsp;|&nbsp; 😴 Sleep: <b>{d_sleep}</b>h<br>
            📱 Social: <b>{d_social}</b>h &nbsp;|&nbsp; 🏃 Exercise: <b>{d_ex}</b>d/wk<br>
            🏫 Attendance: <b>{int(d_att)}%</b> &nbsp;|&nbsp; 🧠 Mental: <b>{d_mh}/10</b>
          </div>
          <div style="background:#10B981;color:white;border-radius:8px;padding:10px;
                      text-align:center;font-weight:700;margin-top:12px;">
            🎯 Predicted: {ds} / 100
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{BC};margin:28px 0;"></div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,val,lbl in [(c1,"1000+","Students"),(c2,str(r2),"R² Accuracy"),(c3,"XGBoost","Algorithm"),(c4,"SM-2","Review Engine")]:
        with col:
            st.markdown(f'<div style="text-align:center;"><div style="font-size:1.5rem;font-weight:900;color:{TP};">{val}</div><div style="font-size:0.75rem;color:{TM};text-transform:uppercase;letter-spacing:0.5px;">{lbl}</div></div>', unsafe_allow_html=True)


# ═══════════ PREDICT & PLAN ═══════════
elif page == "🔮 Predict & Plan":
    st.markdown(f'<div class="badge">🔮 Prediction Engine</div><h2 style="color:{TP};margin:4px 0 4px;">Predict Score & Get Study Plan</h2><p style="color:{TS};">Enter habits → XGBoost predicts → get your personalised plan + email it</p>', unsafe_allow_html=True)

    tab_predict, tab_syllabus = st.tabs(["📊 Habit Predictor", "📄 Syllabus PDF Parser"])

    with tab_predict:
        with st.form("pred_form"):
            st.markdown("#### 📊 Your Daily Habits")
            c1,c2=st.columns(2)
            with c1:
                study=st.slider("📖 Study Hours/Day",0.0,12.0,3.0,0.5)
                sleep=st.slider("😴 Sleep Hours",2.0,12.0,7.0,0.5)
                social=st.slider("📱 Social Media Hours",0.0,8.0,2.0,0.5)
                netflix=st.slider("📺 Entertainment Hours",0.0,6.0,1.0,0.5)
                attend=st.slider("🏫 Attendance %",30.0,100.0,80.0,1.0)
            with c2:
                age=st.slider("🎂 Age",16,30,20)
                exercise=st.slider("🏃 Exercise (days/week)",0,7,3)
                mhealth=st.slider("🧠 Mental Health (1-10)",1,10,5)
                gender=st.selectbox("👤 Gender",["Male","Female","Other"])
                ptjob=st.selectbox("💼 Part-time Job?",["No","Yes"])
                diet=st.selectbox("🥗 Diet Quality",["Good","Fair","Poor"])
                internet=st.selectbox("🌐 Internet Quality",["Good","Average","Poor"])
                pedu=st.selectbox("🎓 Parental Education",["High School","Bachelor","Master","None"])
                extra=st.selectbox("🎭 Extracurricular?",["No","Yes"])
            email_send=st.text_input("📧 Email plan to (optional):",placeholder="you@example.com")
            submitted=st.form_submit_button("🔮 Predict & Generate Plan")

        if submitted:
            raw={"age":age,"gender":gender,"study_hours_per_day":study,
                 "social_media_hours":social,"netflix_hours":netflix,"part_time_job":ptjob,
                 "attendance_percentage":attend,"sleep_hours":sleep,"diet_quality":diet,
                 "exercise_frequency":exercise,"parental_education_level":pedu,
                 "internet_quality":internet,"mental_health_rating":mhealth,
                 "extracurricular_participation":extra}
            predicted=predict_score(model,features,raw)
            plan=generate_study_plan(predicted,raw)

            # Gamification
            u=cur_user()
            if u:
                sr=log_study_session(u["id"],study,predicted,DB_PATH)
                for bk in sr.get("new_badges",[]):
                    bd=BADGE_DEFINITIONS.get(bk,{})
                    st.success(f"🎉 Badge: **{bd.get('icon','🏅')} {bd.get('name',bk)}**")
                if sr.get("xp_earned",0)>0:
                    st.info(f"⚡ +{sr['xp_earned']} XP | 🔥 Streak: {sr['streak']} days")

            tier=plan["tier"]; tc=plan["tier_color"]
            sbg=("#1a0505" if D else "#FEF2F2") if tier=="intensive" else ("#1a1505" if D else "#FFFBEB") if tier=="moderate" else ("#051a0f" if D else "#ECFDF5")
            st.markdown(f"""
            <div style="background:{sbg};border:2px solid {tc}30;border-radius:20px;
                        padding:36px;text-align:center;margin:20px 0;">
              <div style="font-size:0.75rem;font-weight:700;color:{tc};text-transform:uppercase;letter-spacing:3px;">PREDICTED SCORE</div>
              <div style="font-size:4.5rem;font-weight:900;color:{tc};line-height:1;">{predicted}</div>
              <div style="font-size:0.8rem;color:{TM};">out of 100</div>
            </div>""", unsafe_allow_html=True)
            st.info(plan["summary"])

            st.markdown("### 📅 Daily Schedule")
            sched=[{"⏰ Time":ts,"📝 Activity":a} for ts,a in plan["study_schedule"].items()]
            st.dataframe(pd.DataFrame(sched),use_container_width=True,hide_index=True)

            st.markdown("### 🎯 Weekly Goals")
            for i,g in enumerate(plan.get("weekly_goals",[]),1):
                st.markdown(f"**{i}.** {g}")

            # Habit charts
            st.markdown("### 📊 Habit Analysis")
            cc1,cc2=st.columns(2)
            _txt=("#E2E8F0" if D else "#1E1B4B")
            _bg=("#0B0E18" if D else "#F5F3FF")
            _cbg=("#141929" if D else "#FFFFFF")
            _dim=("#94A3B8" if D else "#64748B")
            with cc1:
                lbls=["Study","Sleep","Social","Netflix","Attend/10","Exercise","Mental"]
                yv=[study,sleep,social,netflix,attend/10,exercise,mhealth]
                iv=[5,7.5,1.5,1.0,8.5,4,8]
                fig,ax=plt.subplots(figsize=(7,4.5))
                fig.patch.set_facecolor(_bg)
                ax.set_facecolor(_bg)
                x=list(range(len(lbls))); w=0.32
                bars1=ax.bar([i-w/2 for i in x],yv,w,label="Yours",color="#7C3AED",edgecolor="#9F67FF",linewidth=0.5,zorder=3)
                bars2=ax.bar([i+w/2 for i in x],iv,w,label="Ideal",color="#22D3EE",alpha=0.75,edgecolor="#5DE5F5",linewidth=0.5,zorder=3)
                # Value labels on top of bars
                for bar in bars1:
                    h=bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2,h+0.15,f"{h:.1f}",ha="center",va="bottom",fontsize=7,fontweight="bold",color="#A78BFA")
                for bar in bars2:
                    h=bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2,h+0.15,f"{h:.1f}",ha="center",va="bottom",fontsize=7,fontweight="bold",color="#67E8F9")
                ax.set_xticks(x); ax.set_xticklabels(lbls,fontsize=9,fontweight="600",color=_txt)
                ax.tick_params(axis="y",colors=_dim,labelsize=8)
                ax.legend(fontsize=9,facecolor=_cbg,edgecolor="none",labelcolor=_txt,framealpha=0.9,loc="upper right")
                ax.set_title("Habits vs Ideal",color=_txt,fontweight="bold",fontsize=13,pad=12)
                for spine in ["top","right"]:
                    ax.spines[spine].set_visible(False)
                for spine in ["bottom","left"]:
                    ax.spines[spine].set_color(_dim)
                    ax.spines[spine].set_linewidth(0.5)
                ax.grid(axis="y",color=_dim,alpha=0.15,linewidth=0.5,zorder=0)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            with cc2:
                td={"Study":study,"Sleep":sleep,"Social":social,"Entertainment":netflix,
                    "Other":max(0,24-study-sleep-social-netflix-8)}
                pie_colors=["#7C3AED","#22D3EE","#F59E0B","#EF4444","#6B7280"]
                fig2,ax2=plt.subplots(figsize=(7,4.5))
                fig2.patch.set_facecolor(_bg)
                wedges,texts,autotexts=ax2.pie(
                    td.values(),labels=td.keys(),colors=pie_colors,
                    autopct="%1.0f%%",startangle=90,pctdistance=0.78,labeldistance=1.12,
                    wedgeprops=dict(width=0.55,edgecolor=_bg,linewidth=2.5))
                # Style the label texts (outer names)
                for t in texts:
                    t.set_color(_txt); t.set_fontsize(10); t.set_fontweight("600")
                # Style the percentage texts (inside wedges)
                for t in autotexts:
                    t.set_color("#FFFFFF"); t.set_fontsize(9); t.set_fontweight("bold")
                ax2.set_title("24-Hour Distribution",color=_txt,fontweight="bold",fontsize=13,pad=14)
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            st.success(plan.get("motivation","Keep going! 💪"))


            # Email sending
            if email_send and email_send.strip():
                with st.spinner('Sending study plan email...'):
                    syllabus_topics=st.session_state.get('syllabus_topics')
                    course_name=st.session_state.get('course_name')
                    er=send_study_plan_email(
                        recipient_email=email_send.strip(),
                        student_name=auth_user['display_name'],
                        predicted_score=predicted,
                        plan=plan,
                        syllabus_topics=syllabus_topics,
                        course_name=course_name,
                    )
                if er['success']:
                    st.success('Study plan sent to ' + email_send + '!')
                else:
                    st.error('Email failed: ' + er['error'])

    with tab_syllabus:
        st.markdown('### Syllabus Parser')
        st.markdown('<p>Upload your syllabus PDF - topics are extracted automatically. Uses Gemini AI when available, falls back to local text parsing otherwise.</p>', unsafe_allow_html=True)
        pdf_file=st.file_uploader('Upload Syllabus PDF',type=['pdf'],key='syllabus_pdf')
        if pdf_file:
            with st.spinner('Analysing your syllabus...'):
                result=parse_syllabus_pdf(pdf_file.read())
            if result['success']:
                st.session_state['syllabus_topics']=result['topics']
                st.session_state['course_name']=result['course_name']
                src=result.get('source','')
                if 'Gemini' in src:
                    st.success(result['course_name'] + ' - ' + str(len(result['topics'])) + ' topics via ' + src + '!')
                else:
                    st.success(result['course_name'] + ' – ' + str(len(result['topics'])) + ' topics via ' + src + '!')
                    st.info('💡 Gemini AI was unavailable — used local text parser. Re-upload in a minute to try AI extraction.')
                if result.get('gemini_note'):
                    with st.expander('ℹ️ Gemini API note'):
                        st.info('Gemini AI was temporarily unavailable — topics were extracted using the local text parser instead. '
                                'This is usually caused by high demand on Google\'s servers and resolves on its own. '
                                'Try uploading again in a minute for AI-powered extraction.')
                        with st.expander('🔍 Raw error details'):
                            st.code(result['gemini_note'], language='text')
                for unit in result.get('units',[]):
                    with st.expander(unit['unit'] + ' (' + str(len(unit['topics'])) + ' topics)'):
                        for t in unit['topics']:
                            st.markdown('- ' + t)
                st.markdown('**All Topics:**')
                pills=' '.join(result['topics'])
                st.markdown(pills)
                st.info('These topics will be included in your next study plan email.')
            else:
                st.error('Error: ' + result['error'])


# ═══════════ REVIEW SCHEDULE ═══════════
elif page == "🧠 Review Schedule":
    st.markdown(f'<div class="badge">🧠 SM-2 Spaced Repetition</div><h2 style="color:{TP};margin:4px 0 4px;">Review Schedule</h2>', unsafe_allow_html=True)
    # Scoped CSS — tighter spacing between review cards and controls
    st.markdown("""<style>
    .review-card{margin-bottom:4px!important;}
    </style>""", unsafe_allow_html=True)
    u=cur_user()
    if not u: st.warning("Profile not found."); st.stop()
    uid=u["id"]
    ac1,ac2=st.columns([5,1])
    with ac1: new_topic=st.text_input("Topic name",placeholder="e.g. Binary Trees",label_visibility="collapsed",key="nt")
    with ac2:
        if st.button("➕ Add",key="add_t") and new_topic.strip():
            ok=add_topic(uid,new_topic.strip(),DB_PATH)
            st.success(f"Added!" if ok else f"Already exists."); st.rerun()
    st.markdown(f'<div style="height:1px;background:{BC};margin:16px 0;"></div>', unsafe_allow_html=True)
    due=get_due_items(uid,DB_PATH); allv=get_all_items(uid,DB_PATH)
    if due:
        st.markdown(f"### 📅 Due Today — {len(due)} item(s)")
        for item in due:
            ec="#10B981" if item["ease_factor"]>=2.5 else "#F59E0B" if item["ease_factor"]>=1.8 else "#EF4444"
            st.markdown(f'<div class="review-card"><b style="color:{TP};">{item["topic"]}</b> <span style="font-size:0.8rem;color:{TM};">· Reps:{item["repetitions"]} · Interval:{item["interval"]}d · Ease:<span style="color:{ec};">{item["ease_factor"]:.2f}</span></span></div>', unsafe_allow_html=True)
            qc,bc,dc=st.columns([5,1,1])
            with qc:
                q=st.select_slider(f"q_{item['id']}",options=[0,1,2,3,4,5],value=3,
                    format_func=lambda x:{0:"0-Blackout",1:"1-Barely",2:"2-Wrong",3:"3-Hard",4:"4-Hesitant",5:"5-Perfect!"}[x],
                    label_visibility="collapsed",key=f"qs_{item['id']}")
            with bc:
                if st.button("✅ Mark",key=f"rb_{item['id']}"):
                    r=update_review(uid,item["topic"],q,DB_PATH)
                    xpa,rbs=log_review_xp(uid,q,DB_PATH)
                    if xpa>0: st.info(f"⚡ +{xpa} XP | Next: {r['next_review_date']}")
                    st.rerun()
            with dc:
                if st.button("🗑️",key=f"del_{item['id']}"): delete_topic(uid,item["topic"],DB_PATH); st.rerun()
    else:
        st.success("🎉 All caught up! No reviews due today.")
    fut=[i for i in allv if i not in due]
    if fut:
        st.markdown("### 📋 Upcoming")
        rows=[{"Topic":i["topic"],"Next Review":i["next_review_date"],
               "Interval":f"{i['interval']}d","Reps":i["repetitions"],
               "Ease":round(i["ease_factor"],2)} for i in fut]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    ret=get_retention_features(uid,DB_PATH)
    if ret["topics_total"]>0:
        st.markdown(f'<div style="height:1px;background:{BC};margin:20px 0;"></div>', unsafe_allow_html=True)
        st.markdown("### 🧬 Retention Analytics")
        r1,r2x,r3,r4=st.columns(4)
        for col,val,lbl in [(r1,f"{ret['avg_ease_factor']:.2f}","Avg Ease"),
                            (r2x,ret["topics_mastered"],"Mastered"),
                            (r3,f"{ret['avg_interval']:.1f}d","Avg Interval"),
                            (r4,f"{int(ret['mastery_ratio']*100)}%","Mastery")]:
            with col: st.markdown(f'<div class="metric"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)


# ═══════════ ACHIEVEMENTS ═══════════
elif page == "🏆 Achievements":
    st.markdown(f'<div class="badge">🏆 Gamification</div><h2 style="color:{TP};margin:4px 0 4px;">Your Achievements</h2>', unsafe_allow_html=True)
    u=cur_user()
    if not u: st.warning("Profile not found."); st.stop()
    stats=get_user_stats(u["id"],DB_PATH)
    s1,s2,s3,s4=st.columns(4)
    for col,val,lbl,c in [(s1,f"{stats['current_streak']} 🔥","Current Streak","#F59E0B"),
                           (s2,stats["longest_streak"],"Longest Streak","#EF4444"),
                           (s3,f"{stats['xp']:,}","Total XP",PL),
                           (s4,f"{stats['badges_earned']}/{stats['badges_total']}","Badges",CY)]:
        with col: st.markdown(f'<div class="metric"><div class="metric-val" style="color:{c};">{val}</div><div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{BC};margin:20px 0;"></div>',unsafe_allow_html=True)
    lv=stats["level"]; xp_pct=stats["xp_pct"]
    st.markdown(f"""
    <div class="glass">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="font-size:1.2rem;font-weight:900;color:{CY};">Level {lv}</span>
        <span style="font-size:0.85rem;color:{TM};">{xp_pct}% to Level {lv+1}</span>
      </div>
      <div class="xp-outer"><div class="xp-inner" style="width:{xp_pct}%;"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:{TM};margin-top:4px;">
        <span>{stats["xp_floor"]:,} XP</span><span>{stats["xp_ceil"]:,} XP</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{BC};margin:20px 0;"></div>',unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{TP};">🏅 Badges</h3>',unsafe_allow_html=True)
    earned=[b for b in stats["all_badges"] if b["earned"]]
    locked=[b for b in stats["all_badges"] if not b["earned"]]
    if earned:
        st.markdown("**✨ Earned**")
        cols=st.columns(6)
        for i,b in enumerate(earned):
            with cols[i%6]:
                st.markdown(f'<div style="background:{b["color"]}18;border:2px solid {b["color"]}44;border-radius:14px;padding:12px;text-align:center;margin-bottom:8px;"><div style="font-size:1.8rem;">{b["icon"]}</div><div style="font-size:0.62rem;font-weight:700;color:{b["color"]};">{b["name"]}</div></div>',unsafe_allow_html=True)
    st.markdown(f"**🔒 Locked ({len(locked)})**")
    cols=st.columns(6)
    for i,b in enumerate(locked):
        with cols[i%6]:
            st.markdown(f'<div style="background:{CARD2};border:2px solid {BC};border-radius:14px;padding:12px;text-align:center;margin-bottom:8px;opacity:0.35;"><div style="font-size:1.8rem;">{b["icon"]}</div><div style="font-size:0.62rem;font-weight:700;color:{TM};">{b["name"]}</div></div>',unsafe_allow_html=True)
    xp_log=stats.get("xp_log",[])
    if xp_log:
        st.markdown(f'<div style="height:1px;background:{BC};margin:20px 0;"></div>',unsafe_allow_html=True)
        st.markdown(f'<h3 style="color:{TP};">⚡ Recent XP</h3>',unsafe_allow_html=True)
        for e in xp_log:
            ts=e["timestamp"][:16].replace("T"," ")
            st.markdown(f'<div style="display:flex;justify-content:space-between;background:{CARD2};padding:10px 16px;border-radius:10px;margin-bottom:6px;border:1px solid {BC};"><span style="color:{TP};font-size:0.9rem;">{e["reason"]}</span><div><span style="color:{GR};font-weight:700;">+{e["amount"]} XP</span>&nbsp;<span style="color:{TM};font-size:0.75rem;">{ts}</span></div></div>',unsafe_allow_html=True)


# ═══════════ MODEL INSIGHTS ═══════════
elif page == "📊 Model Insights":
    st.markdown(f'<div class="badge">📊 Analytics</div><h2 style="color:{TP};margin:4px 0 4px;">Model Performance</h2>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,k,l in [(c1,"R2_Score","R² Score"),(c2,"MAE","MAE"),(c3,"RMSE","RMSE"),(c4,"MSE","MSE")]:
        with col: st.markdown(f'<div class="metric"><div class="metric-val">{metrics.get(k,"N/A")}</div><div class="metric-lbl">{l}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{BC};margin:20px 0;"></div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["📈 Actual vs Predicted","🔥 Feature Importance","🗺️ Correlation"])
    with t1: st.pyplot(plot_actual_vs_predicted(y_test,y_pred)); st.info("Each dot = one student. Closer to diagonal = better prediction.")
    with t2: st.pyplot(plot_feature_importance(model,features)); st.info("Higher importance = stronger influence on exam score.")
    with t3: st.pyplot(plot_correlation_heatmap(df_enc)); st.info("Check the exam_score row/column for habit correlations.")


# ═══════════ ABOUT ═══════════
elif page == "ℹ️ About":
    st.markdown(f'<div class="badge">ℹ️ About</div><h2 style="color:{TP};margin:4px 0 4px;">StudyGenie — XGBoost Edition</h2><p style="color:{TS};">BCA 4th Semester Mini Project</p>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="glass">
          <h4>🏗️ Architecture</h4>
          <p><b>1. Auth Layer</b> — bcrypt email/password (SQLite)</p>
          <p><b>2. ML Layer</b> — XGBoost + Optuna + scikit-learn</p>
          <p><b>3. Gemini Layer</b> — PDF syllabus parsing + topic extraction</p>
          <p><b>4. Scheduling</b> — SM-2 Spaced Repetition Engine</p>
          <p><b>5. Gamification</b> — Streaks, XP, Badges</p>
          <p><b>6. Email</b> — SMTP HTML study plan delivery</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass">
          <h4>🛠️ Tech Stack</h4>
          <p>🐍 Python 3.x</p>
          <p>⚡ XGBoost + Optuna</p>
          <p>🤖 Google Gemini API</p>
          <p>📄 PyMuPDF (PDF parsing)</p>
          <p>🔐 bcrypt (auth)</p>
          <p>📊 Matplotlib / Seaborn</p>
          <p>🗄️ SQLite &nbsp;|&nbsp; 🌐 Streamlit</p>
        </div>""", unsafe_allow_html=True)
