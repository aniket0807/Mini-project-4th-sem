
# This script writes the full new app.py to disk.
import textwrap, os

BASE = os.path.dirname(__file__)

APP = textwrap.dedent('''
import streamlit as st, pandas as pd, numpy as np, matplotlib, os, sys, random, joblib
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
@import url(\'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;900&display=swap\');
*{{font-family:\'DM Sans\',sans-serif;}} #MainMenu,footer{{visibility:hidden;}}
.stApp{{background:{BG};}}
.stApp,.stApp p,.stApp span,.stApp label,.stApp div{{color:{TP};}}
.stMarkdown,.stMarkdown p,.stMarkdown li,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{{color:{TP}!important;}}
.stSlider label,.stSlider p,.stSlider span{{color:{TP}!important;}}
.stSelectbox label,.stNumberInput label,.stTextInput label,.stRadio label{{color:{TP}!important;}}
[data-testid="stSidebar"]{{background:{SBGR};border-right:1px solid rgba(255,255,255,0.06);}}
[data-testid="stSidebar"] *{{color:#E2E8F0!important;}}
.stButton>button{{background:linear-gradient(135deg,#7C3AED,#6D28D9)!important;color:white!important;
  border:none!important;border-radius:10px!important;padding:12px 28px!important;
  font-weight:700!important;transition:all 0.3s!important;width:100%;}}
.stButton>button:hover{{box-shadow:0 6px 20px rgba(124,58,237,0.35)!important;transform:translateY(-2px)!important;}}
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
    st.markdown(f"""
    <div style="max-width:440px;margin:60px auto;">
    <div style="text-align:center;margin-bottom:32px;">
      <div style="font-size:2rem;font-weight:900;color:{PU};">🎓 StudyGenie</div>
      <div style="color:{TS};margin-top:6px;">AI-Powered Personalised Study Planner</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1,2,1])
    with col_c:
        tab_login, tab_reg = st.tabs(["🔑 Sign In", "✨ Create Account"])

        with tab_login:
            with st.form("login_form"):
                st.markdown("### Welcome back!")
                lemail = st.text_input("Email", placeholder="you@example.com", key="l_email")
                lpw    = st.text_input("Password", type="password", placeholder="••••••••", key="l_pw")
                lsub   = st.form_submit_button("Sign In →")
            if lsub:
                res = login_user(lemail, lpw, DB_PATH)
                if res["success"]:
                    session_login(res["user"])
                    st.success(f"Welcome back, {res[\'user\'][\'display_name\']}! 👋")
                    st.rerun()
                else:
                    st.error(res["error"])

        with tab_reg:
            with st.form("reg_form"):
                st.markdown("### Join StudyGenie")
                rname  = st.text_input("Display Name", placeholder="Aniket", key="r_name")
                remail = st.text_input("Email", placeholder="you@example.com", key="r_email")
                rpw    = st.text_input("Password (min 6 chars)", type="password", key="r_pw")
                rpw2   = st.text_input("Confirm Password", type="password", key="r_pw2")
                rsub   = st.form_submit_button("Create Account →")
            if rsub:
                if rpw != rpw2:
                    st.error("Passwords do not match.")
                else:
                    res = register_user(remail, rname, rpw, DB_PATH)
                    if res["success"]:
                        session_login(res["user"])
                        st.success(f"Account created! Welcome, {rname}! 🎉")
                        st.rerun()
                    else:
                        st.error(res["error"])
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
        st=_s=get_user_stats(u["id"],DB_PATH)
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
''').strip()

# Fix the sidebar stat bug (st.markdown after reassigning st)
APP = APP.replace(
    "        st=_s=get_user_stats(u[\"id\"],DB_PATH)\n        st.markdown",
    "        _s=get_user_stats(u[\"id\"],DB_PATH)\n        st.markdown"
)

with open(os.path.join(BASE, "app.py"), "w", encoding="utf-8") as f:
    f.write(APP.lstrip())

print("Part A written:", len(APP), "chars")
