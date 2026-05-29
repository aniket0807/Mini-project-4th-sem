"""
===================================================================
EMAIL_SERVICE.PY — Personalized Study Plan Email Delivery
===================================================================
Sends a beautifully formatted HTML email containing the student's
personalized study plan, predicted score, and extracted syllabus
topics (if available).

Configuration (via .env):
  EMAIL_SENDER   — Gmail address used as sender
  EMAIL_PASSWORD — Gmail App Password (NOT your regular password)
  SMTP_HOST      — default: smtp.gmail.com
  SMTP_PORT      — default: 587

Gmail Setup:
  1. Enable 2-Step Verification on your Google Account
  2. Go to https://myaccount.google.com/apppasswords
  3. Generate an App Password for "Mail"
  4. Use that 16-char password as EMAIL_PASSWORD in .env
===================================================================
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
#  CONFIG HELPER  (reads .env fresh on every call)
# ─────────────────────────────────────────────
def _cfg():
    """Re-read .env each time so a restart isn't required after editing."""
    load_dotenv(override=True)
    return {
        "host":     os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port":     int(os.getenv("SMTP_PORT", "587")),
        "sender":   os.getenv("EMAIL_SENDER",   "").strip(),
        "password": os.getenv("EMAIL_PASSWORD", "").strip(),
    }


# ─────────────────────────────────────────────
#  HTML EMAIL BUILDER
# ─────────────────────────────────────────────
def _build_html(
    student_name:   str,
    predicted_score: int,
    plan:           Dict,
    syllabus_topics: Optional[List[str]] = None,
    course_name:    Optional[str] = None,
) -> str:
    """Build the full HTML email body."""

    tier_colors = {
        "intensive": "#EF4444",
        "moderate":  "#F59E0B",
        "good":      "#10B981",
    }
    tier       = plan.get("tier", "moderate")
    tier_color = tier_colors.get(tier, "#7C3AED")
    tier_label = plan.get("tier_label", "Your Tier")
    summary    = plan.get("summary", "")

    # ── Schedule rows ──
    schedule_rows = ""
    for time_slot, activity in plan.get("study_schedule", {}).items():
        schedule_rows += f"""
        <tr>
          <td style="padding:10px 16px;border-bottom:1px solid #E2E8F0;
                     font-weight:600;color:#1E1B4B;white-space:nowrap;">{time_slot}</td>
          <td style="padding:10px 16px;border-bottom:1px solid #E2E8F0;color:#475569;">{activity}</td>
        </tr>"""

    # ── Weekly goals ──
    goals_html = ""
    for i, goal in enumerate(plan.get("weekly_goals", []), 1):
        goals_html += f'<li style="margin-bottom:8px;color:#475569;">{goal}</li>'

    # ── Recommendations ──
    rec_html = ""
    status_colors = {"critical": "#EF4444", "warning": "#F59E0B", "good": "#10B981"}
    for rec in plan.get("recommendations", []):
        c = status_colors.get(rec.get("status", "good"), "#10B981")
        rec_html += f"""
        <div style="border-left:4px solid {c};background:#F8FAFC;
                    border-radius:8px;padding:14px 18px;margin-bottom:12px;">
          <div style="font-weight:700;color:{c};margin-bottom:6px;">
            {rec.get('icon','')} {rec.get('area','')}
          </div>
          <div style="font-size:13px;color:#64748B;margin-bottom:4px;">
            Current: <strong>{rec.get('current','')}</strong> &nbsp;|&nbsp;
            Recommended: <strong style="color:{c};">{rec.get('target','')}</strong>
          </div>
          <div style="font-size:13px;color:#475569;">{rec.get('tip','')}</div>
        </div>"""

    # ── Syllabus topics ──
    syllabus_section = ""
    if syllabus_topics:
        topic_pills = "".join(
            f'<span style="display:inline-block;background:#EDE9FE;color:#7C3AED;'
            f'padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;'
            f'margin:4px;">{t}</span>'
            for t in syllabus_topics[:30]   # cap at 30 in email
        )
        more = f'<p style="color:#94A3B8;font-size:12px;">...and {len(syllabus_topics)-30} more topics</p>' if len(syllabus_topics) > 30 else ""
        course_title = course_name or "Your Course"
        syllabus_section = f"""
        <div style="background:#FAFAFE;border:1px solid #E2E8F0;border-radius:16px;
                    padding:28px;margin-bottom:24px;">
          <h2 style="color:#1E1B4B;font-size:18px;margin:0 0 6px;">
            📚 Syllabus Topics — {course_title}
          </h2>
          <p style="color:#64748B;font-size:13px;margin:0 0 16px;">
            Extracted by Gemini AI from your uploaded syllabus PDF
          </p>
          <div>{topic_pills}</div>
          {more}
        </div>"""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Your StudyGenie Study Plan</title>
</head>
<body style="margin:0;padding:0;background:#F5F3FF;font-family:'Segoe UI',Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#F5F3FF;padding:32px 16px;">
<tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0"
       style="background:#FFFFFF;border-radius:20px;overflow:hidden;
              box-shadow:0 4px 32px rgba(124,58,237,0.10);">

  <!-- HEADER -->
  <tr>
    <td style="background:linear-gradient(135deg,#1a1040,#231555,#0F1221);
               padding:40px 36px;text-align:center;">
      <div style="font-size:28px;font-weight:900;color:#F9A825;
                  letter-spacing:2px;text-transform:uppercase;">STUDYGENIE</div>
      <div style="font-size:13px;color:#94A3B8;letter-spacing:3px;
                  text-transform:uppercase;margin-top:4px;">XGBoost AI Engine</div>
      <div style="margin-top:20px;font-size:22px;font-weight:700;color:#FFFFFF;">
        Your Personalized Study Plan
      </div>
    </td>
  </tr>

  <!-- GREETING -->
  <tr>
    <td style="padding:32px 36px 0;">
      <p style="font-size:16px;color:#1E1B4B;margin:0;">
        Hi <strong>{student_name}</strong> 👋,
      </p>
      <p style="font-size:15px;color:#475569;line-height:1.7;margin-top:10px;">
        Your personalized study plan is ready! Based on your habit inputs,
        our XGBoost model has generated a tailored plan to help you reach your potential.
      </p>
    </td>
  </tr>

  <!-- SCORE CARD -->
  <tr>
    <td style="padding:24px 36px;">
      <div style="background:{tier_color}10;border:2px solid {tier_color}30;
                  border-radius:16px;padding:28px;text-align:center;">
        <div style="font-size:11px;font-weight:700;color:{tier_color};
                    text-transform:uppercase;letter-spacing:3px;">XGBoost Predicted Score</div>
        <div style="font-size:64px;font-weight:900;color:{tier_color};
                    line-height:1;margin:12px 0;">{predicted_score}</div>
        <div style="font-size:13px;color:#94A3B8;">out of 100</div>
        <div style="margin-top:12px;display:inline-block;background:{tier_color}20;
                    color:{tier_color};padding:6px 18px;border-radius:20px;
                    font-weight:700;font-size:13px;">{tier_label}</div>
      </div>
    </td>
  </tr>

  <!-- SUMMARY -->
  <tr>
    <td style="padding:0 36px 24px;">
      <div style="background:#F0FDF4;border-left:4px solid #10B981;
                  border-radius:8px;padding:16px 20px;">
        <p style="margin:0;color:#065F46;font-size:14px;line-height:1.7;">{summary}</p>
      </div>
    </td>
  </tr>

  <!-- SYLLABUS TOPICS (conditional) -->
  <tr><td style="padding:0 36px 8px;">{syllabus_section}</td></tr>

  <!-- SCHEDULE -->
  <tr>
    <td style="padding:0 36px 24px;">
      <div style="background:#FAFAFE;border:1px solid #E2E8F0;
                  border-radius:16px;padding:28px;">
        <h2 style="color:#1E1B4B;font-size:18px;margin:0 0 20px;">
          📅 Recommended Daily Schedule
        </h2>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;">
          <thead>
            <tr style="background:#EDE9FE;">
              <th style="padding:10px 16px;text-align:left;color:#7C3AED;
                         font-size:13px;font-weight:700;">Time Slot</th>
              <th style="padding:10px 16px;text-align:left;color:#7C3AED;
                         font-size:13px;font-weight:700;">Activity</th>
            </tr>
          </thead>
          <tbody>{schedule_rows}</tbody>
        </table>
      </div>
    </td>
  </tr>

  <!-- WEEKLY GOALS -->
  <tr>
    <td style="padding:0 36px 24px;">
      <div style="background:#FAFAFE;border:1px solid #E2E8F0;
                  border-radius:16px;padding:28px;">
        <h2 style="color:#1E1B4B;font-size:18px;margin:0 0 16px;">
          🎯 This Week's Goals
        </h2>
        <ul style="margin:0;padding-left:20px;">
          {goals_html}
        </ul>
      </div>
    </td>
  </tr>

  <!-- RECOMMENDATIONS -->
  <tr>
    <td style="padding:0 36px 24px;">
      <div style="background:#FAFAFE;border:1px solid #E2E8F0;
                  border-radius:16px;padding:28px;">
        <h2 style="color:#1E1B4B;font-size:18px;margin:0 0 16px;">
          📋 Habit Analysis & Recommendations
        </h2>
        {rec_html}
      </div>
    </td>
  </tr>

  <!-- MOTIVATION -->
  <tr>
    <td style="padding:0 36px 8px;">
      <div style="background:linear-gradient(135deg,#EDE9FE,#F3E8FF);
                  border-radius:12px;padding:20px;text-align:center;">
        <p style="margin:0;color:#7C3AED;font-size:15px;font-weight:600;
                  line-height:1.7;">{plan.get('motivation','Keep going — every study session counts! 💪')}</p>
      </div>
    </td>
  </tr>

  <!-- FOOTER -->
  <tr>
    <td style="padding:32px 36px;text-align:center;border-top:1px solid #E2E8F0;
               margin-top:24px;">
      <p style="margin:0;font-size:12px;color:#94A3B8;">
        Generated by <strong style="color:#7C3AED;">StudyGenie</strong> —
        AI-powered personalized learning<br>
        BCA 4th Semester Mini Project
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ─────────────────────────────────────────────
#  PUBLIC SEND FUNCTION
# ─────────────────────────────────────────────
def send_study_plan_email(
    recipient_email:  str,
    student_name:     str,
    predicted_score:  int,
    plan:             Dict,
    syllabus_topics:  Optional[List[str]] = None,
    course_name:      Optional[str] = None,
) -> Dict:
    """
    Send a personalized study-plan email to `recipient_email`.

    Returns:
        {"success": True}
        {"success": False, "error": "..."}
    """
    c = _cfg()   # fresh read of .env on every call
    sender   = c["sender"]
    password = c["password"]

    # ── Validation ──────────────────────────────────────────────────
    if not sender:
        return {"success": False,
                "error": "EMAIL_SENDER is not set in your .env file."}

    _bad_placeholders = {
        "", "your_gmail_app_password_here", "REPLACE_WITH_16CHAR_APP_PASSWORD"
    }
    # Gmail App Passwords are exactly 16 alpha chars (Google strips spaces)
    pw_clean = password.replace(" ", "")
    if password in _bad_placeholders:
        return {
            "success": False,
            "error": (
                "EMAIL_PASSWORD is a placeholder. "
                "You need a Gmail App Password (16 characters). "
                "Steps: myaccount.google.com → Security → 2-Step Verification → "
                "App Passwords → Generate one for 'Mail'."
            ),
        }

    # Warn if it looks like a normal password (has symbols / not 16 chars)
    if len(pw_clean) != 16 or not pw_clean.isalpha():
        # Still try — user might have a non-Gmail SMTP — just surface a hint on failure
        _likely_regular_pw = True
    else:
        _likely_regular_pw = False

    # ── Build message ────────────────────────────────────────────────
    msg            = MIMEMultipart("alternative")
    msg["Subject"] = f"Your StudyGenie Study Plan - Score: {predicted_score}/100"
    msg["From"]    = f"StudyGenie AI <{sender}>"
    msg["To"]      = recipient_email

    html_body = _build_html(
        student_name=student_name,
        predicted_score=predicted_score,
        plan=plan,
        syllabus_topics=syllabus_topics,
        course_name=course_name,
    )
    plain_body = (
        f"Hi {student_name},\n\n"
        f"Your StudyGenie Study Plan\n"
        f"Predicted Score: {predicted_score}/100\n\n"
        f"{plan.get('summary', '')}\n\n"
        f"-- StudyGenie AI"
    )
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body,  "html"))

    # ── Send ────────────────────────────────────────────────────────
    try:
        with smtplib.SMTP(c["host"], c["port"], timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipient_email, msg.as_string())
        return {"success": True}
    except smtplib.SMTPAuthenticationError:
        hint = (
            " It looks like you used your regular Gmail password. "
            "Gmail requires an App Password for SMTP. "
            "Generate one at: myaccount.google.com/apppasswords"
        ) if _likely_regular_pw else ""
        return {
            "success": False,
            "error": f"Authentication failed.{hint}",
        }
    except smtplib.SMTPException as e:
        return {"success": False, "error": f"SMTP error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to send email: {e}"}
