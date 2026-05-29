
import os

BASE = os.path.dirname(__file__)

PART_B = '''

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
            with cc1:
                lbls=["Study","Sleep","Social","Netflix","Attend/10","Exercise","Mental"]
                yv=[study,sleep,social,netflix,attend/10,exercise,mhealth]
                iv=[5,7.5,1.5,1.0,8.5,4,8]
                fig,ax=plt.subplots(figsize=(7,4))
                fig.patch.set_facecolor("#0B0E18" if D else "#F5F3FF")
                ax.set_facecolor("#0B0E18" if D else "#F5F3FF")
                x=range(len(lbls)); w=0.35
                ax.bar([i-w/2 for i in x],yv,w,label="Yours",color="#7C3AED")
                ax.bar([i+w/2 for i in x],iv,w,label="Ideal",color="#22D3EE",alpha=0.7)
                ax.set_xticks(list(x)); ax.set_xticklabels(lbls,fontsize=8,color="#E2E8F0" if D else "#1E1B4B")
                ax.legend(fontsize=8,facecolor="#141929" if D else "#fff",edgecolor="none",labelcolor="#E2E8F0" if D else "#1E1B4B")
                ax.set_title("Habits vs Ideal",color="#E2E8F0" if D else "#1E1B4B",fontweight="bold")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            with cc2:
                td={"Study":study,"Sleep":sleep,"Social":social,"Entertainment":netflix,
                    "Other":max(0,24-study-sleep-social-netflix-8)}
                fig2,ax2=plt.subplots(figsize=(7,4))
                fig2.patch.set_facecolor("#0B0E18" if D else "#F5F3FF")
                ax2.pie(td.values(),labels=td.keys(),colors=["#7C3AED","#22D3EE","#F59E0B","#EF4444","#6B7280"],
                        autopct="%1.0f%%",startangle=90,wedgeprops=dict(width=0.45,edgecolor="#0B0E18" if D else "#F5F3FF",linewidth=2))
                ax2.set_title("24-Hour Distribution",color="#E2E8F0" if D else "#1E1B4B",fontweight="bold")
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            st.success(plan.get("motivation","Keep going! 💪"))

            # Email sending
            if email_send and email_send.strip():
                with st.spinner("📧 Sending study plan email…"):
                    syllabus_topics=st.session_state.get("syllabus_topics")
                    course_name=st.session_state.get("course_name")
                    er=send_study_plan_email(
                        recipient_email=email_send.strip(),
                        student_name=auth_user["display_name"],
                        predicted_score=predicted,
                        plan=plan,
                        syllabus_topics=syllabus_topics,
                        course_name=course_name,
                    )
                if er["success"]:
                    st.success(f"✅ Study plan sent to **{email_send}**!")
                else:
                    st.error(f"❌ Email failed: {er['error']}")

    with tab_syllabus:
        st.markdown("### 📄 Gemini AI Syllabus Parser")
        st.markdown(f"<p style='color:{TS};'>Upload your syllabus PDF — Gemini will extract all study topics automatically.</p>", unsafe_allow_html=True)
        pdf_file=st.file_uploader("Upload Syllabus PDF",type=["pdf"],key="syllabus_pdf")
        if pdf_file:
            with st.spinner("🤖 Gemini is analysing your syllabus…"):
                result=parse_syllabus_pdf(pdf_file.read())
            if result["success"]:
                st.session_state["syllabus_topics"]=result["topics"]
                st.session_state["course_name"]=result["course_name"]
                st.success(f"✅ **{result['course_name']}** — {len(result['topics'])} topics extracted!")
                for unit in result.get("units",[]):
                    with st.expander(f"📚 {unit['unit']} ({len(unit['topics'])} topics)"):
                        for t in unit["topics"]:
                            st.markdown(f"• {t}")
                st.markdown("**All Topics (flat list):**")
                pills=" ".join(f"`{t}`" for t in result["topics"])
                st.markdown(pills)
                st.info("💡 These topics will be included in your next study plan email automatically.")
            else:
                st.error(f"❌ {result['error']}")
                if "GEMINI_API_KEY" in result.get("error",""):
                    st.warning("Add `GEMINI_API_KEY=your_key` to a `.env` file in the project folder.")


# ═══════════ REVIEW SCHEDULE ═══════════
elif page == "🧠 Review Schedule":
    st.markdown(f'<div class="badge">🧠 SM-2 Spaced Repetition</div><h2 style="color:{TP};margin:4px 0 4px;">Review Schedule</h2>', unsafe_allow_html=True)
    u=cur_user()
    if not u: st.warning("Profile not found."); st.stop()
    uid=u["id"]
    ac1,ac2=st.columns([3,1])
    with ac1: new_topic=st.text_input("Topic name",placeholder="e.g. Binary Trees",label_visibility="collapsed",key="nt")
    with ac2:
        if st.button("➕ Add",key="add_t") and new_topic.strip():
            ok=add_topic(uid,new_topic.strip(),DB_PATH)
            st.success(f"Added!" if ok else f"Already exists."); st.rerun()
    st.markdown(f'<div style="height:1px;background:{BC};margin:20px 0;"></div>', unsafe_allow_html=True)
    due=get_due_items(uid,DB_PATH); allv=get_all_items(uid,DB_PATH)
    if due:
        st.markdown(f"### 📅 Due Today — {len(due)} item(s)")
        for item in due:
            ec="#10B981" if item["ease_factor"]>=2.5 else "#F59E0B" if item["ease_factor"]>=1.8 else "#EF4444"
            st.markdown(f'<div class="review-card"><b style="color:{TP};">{item["topic"]}</b> <span style="font-size:0.8rem;color:{TM};">· Reps:{item["repetitions"]} · Interval:{item["interval"]}d · Ease:<span style="color:{ec};">{item["ease_factor"]:.2f}</span></span></div>', unsafe_allow_html=True)
            qc,bc,dc=st.columns([4,2,1])
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
'''

with open(r"d:\Personal Files\BCA\4TH SEM\anti\app.py", "a", encoding="utf-8") as f:
    f.write(PART_B)

print("Part B appended OK — total lines:", len(open(r"d:\Personal Files\BCA\4TH SEM\anti\app.py", encoding="utf-8").readlines()))
