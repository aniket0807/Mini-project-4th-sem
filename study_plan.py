"""
===================================================================
STUDY_PLAN.PY — Personalized Study Plan Generator
===================================================================
This module generates customized study improvement plans based on:
  1. The predicted exam score (from Linear Regression model)
  2. The student's current habits (input features)

The plans are categorized into 3 tiers:
  - Score < 50   →  🔴 Intensive Improvement Plan
  - Score 50-75  →  🟡 Moderate Improvement Plan
  - Score > 75   →  🟢 Advanced Optimization Plan

Each plan includes specific, actionable recommendations based on
the student's weak areas (e.g., low study hours, high social media).
===================================================================
"""


def generate_study_plan(predicted_score: float, student_features: dict) -> dict:
    """
    Generate a personalized study plan based on the predicted score
    and the student's current habits.

    Parameters:
    -----------
    predicted_score : float
        The predicted exam score (0-100) from the model.
    student_features : dict
        Dictionary of the student's current habits. Expected keys:
        - study_hours_per_day (float)
        - sleep_hours (float)
        - social_media_hours (float)
        - netflix_hours (float)
        - attendance_percentage (float)
        - exercise_frequency (int, 0-6 days/week)
        - mental_health_rating (int, 1-10)
        - diet_quality (str: 'Poor', 'Fair', 'Good')

    Returns:
    --------
    dict with keys:
        - tier (str): "intensive" / "moderate" / "advanced"
        - tier_label (str): Display label
        - tier_emoji (str): Emoji for the tier
        - tier_color (str): Color code for UI
        - summary (str): Overall assessment
        - study_schedule (dict): Recommended daily schedule
        - recommendations (list): Specific improvement tips
        - weekly_goals (list): Goals for the week
        - motivation (str): Motivational message
    """

    study_hours = student_features.get("study_hours_per_day", 3)
    sleep_hours = student_features.get("sleep_hours", 7)
    social_media = student_features.get("social_media_hours", 2)
    netflix_hours = student_features.get("netflix_hours", 1)
    attendance = student_features.get("attendance_percentage", 80)
    exercise = student_features.get("exercise_frequency", 3)
    mental_health = student_features.get("mental_health_rating", 5)
    diet = student_features.get("diet_quality", "Fair")

    # ─── Initialize the plan ───
    plan = {
        "predicted_score": round(predicted_score, 1),
        "recommendations": [],
        "weekly_goals": [],
    }

    # ═══════════════════════════════════════════
    #  TIER 1: INTENSIVE IMPROVEMENT (Score < 50)
    # ═══════════════════════════════════════════
    if predicted_score < 50:
        plan["tier"] = "intensive"
        plan["tier_label"] = "🔴 Intensive Improvement Plan"
        plan["tier_emoji"] = "🔴"
        plan["tier_color"] = "#FF4444"
        plan["summary"] = (
            f"Your predicted score is {predicted_score:.1f}/100. "
            "This indicates significant room for improvement. "
            "Don't worry — with the right changes to your daily habits, "
            "you can dramatically boost your performance! Let's build a "
            "strong foundation together."
        )
        plan["study_schedule"] = {
            "Morning (6-8 AM)": "📖 Revise previous day's topics (2 hrs)",
            "College Hours": "🏫 Attend ALL classes, take detailed notes",
            "Afternoon (2-4 PM)": "📝 Solve practice problems & assignments (2 hrs)",
            "Evening (6-8 PM)": "📚 Study new concepts, read textbook (2 hrs)",
            "Night (9-10 PM)": "🔄 Quick revision & plan next day (1 hr)",
        }
        plan["target_study_hours"] = 7
        plan["motivation"] = (
            "💪 \"Every expert was once a beginner. The key is to start now "
            "and stay consistent. Small daily improvements lead to stunning results!\""
        )

    # ═══════════════════════════════════════════
    #  TIER 2: MODERATE IMPROVEMENT (Score 50-75)
    # ═══════════════════════════════════════════
    elif predicted_score <= 75:
        plan["tier"] = "moderate"
        plan["tier_label"] = "🟡 Moderate Improvement Plan"
        plan["tier_emoji"] = "🟡"
        plan["tier_color"] = "#FFB300"
        plan["summary"] = (
            f"Your predicted score is {predicted_score:.1f}/100. "
            "You have a decent foundation! With some targeted improvements "
            "in your study habits, you can push into the top tier. "
            "Focus on consistency and quality study time."
        )
        plan["study_schedule"] = {
            "Morning (7-8 AM)": "📖 Revise key concepts (1 hr)",
            "College Hours": "🏫 Active participation in class",
            "Afternoon (3-5 PM)": "📝 Practice problems & previous papers (2 hrs)",
            "Evening (7-8:30 PM)": "📚 Deep study — focus on weak subjects (1.5 hrs)",
            "Night (9:30-10 PM)": "🔄 Light revision & flashcards (30 min)",
        }
        plan["target_study_hours"] = 5
        plan["motivation"] = (
            "🌟 \"You're already on the right track! A few smart adjustments "
            "to your routine can take you from good to great. Keep pushing!\""
        )

    # ═══════════════════════════════════════════
    #  TIER 3: ADVANCED OPTIMIZATION (Score > 75)
    # ═══════════════════════════════════════════
    else:
        plan["tier"] = "advanced"
        plan["tier_label"] = "🟢 Advanced Optimization Plan"
        plan["tier_emoji"] = "🟢"
        plan["tier_color"] = "#00C853"
        plan["summary"] = (
            f"Your predicted score is {predicted_score:.1f}/100. "
            "Excellent work! You're performing well. This plan focuses on "
            "fine-tuning your habits for peak performance and maintaining "
            "your edge while avoiding burnout."
        )
        plan["study_schedule"] = {
            "Morning (7-8 AM)": "📖 Advanced problem-solving (1 hr)",
            "College Hours": "🏫 Engage deeply, ask questions in class",
            "Afternoon (3-4:30 PM)": "📝 Competitive practice & research (1.5 hrs)",
            "Evening (7-8 PM)": "📚 Explore advanced topics / projects (1 hr)",
            "Night (9-9:30 PM)": "🔄 Relaxation & light revision (30 min)",
        }
        plan["target_study_hours"] = 4
        plan["motivation"] = (
            "🏆 \"Great students don't just study hard — they study smart. "
            "Maintain your balance, stay curious, and aim for excellence!\""
        )

    # ═══════════════════════════════════════════════════
    #  HABIT-SPECIFIC RECOMMENDATIONS (across all tiers)
    # ═══════════════════════════════════════════════════

    # 📖 Study Hours
    if study_hours < 2:
        plan["recommendations"].append({
            "icon": "📖",
            "area": "Study Hours",
            "status": "critical",
            "current": f"{study_hours} hrs/day",
            "target": "4-5 hrs/day",
            "tip": (
                "Your study hours are very low. Start by adding 30 minutes "
                "each week. Use the Pomodoro technique (25 min study + 5 min break) "
                "to build focus gradually."
            )
        })
        plan["weekly_goals"].append("📖 Increase daily study time by 30 minutes each week")
    elif study_hours < 4:
        plan["recommendations"].append({
            "icon": "📖",
            "area": "Study Hours",
            "status": "warning",
            "current": f"{study_hours} hrs/day",
            "target": "4-5 hrs/day",
            "tip": (
                "Your study time is below average. Try to dedicate 1 more hour "
                "to focused, distraction-free study. Quality matters as much as quantity!"
            )
        })
        plan["weekly_goals"].append("📖 Add 1 extra hour of focused study daily")
    else:
        plan["recommendations"].append({
            "icon": "📖",
            "area": "Study Hours",
            "status": "good",
            "current": f"{study_hours} hrs/day",
            "target": "Maintain 4+ hrs/day",
            "tip": (
                "Great study routine! Focus on making study sessions more productive "
                "with active recall and spaced repetition techniques."
            )
        })

    # 😴 Sleep Hours
    if sleep_hours < 6:
        plan["recommendations"].append({
            "icon": "😴",
            "area": "Sleep",
            "status": "critical",
            "current": f"{sleep_hours} hrs/night",
            "target": "7-8 hrs/night",
            "tip": (
                "Sleep deprivation severely impacts memory and concentration! "
                "Aim for 7-8 hours. Set a fixed bedtime, avoid screens 30 min "
                "before bed, and create a dark, quiet sleeping environment."
            )
        })
        plan["weekly_goals"].append("😴 Get at least 7 hours of sleep every night this week")
    elif sleep_hours > 9:
        plan["recommendations"].append({
            "icon": "😴",
            "area": "Sleep",
            "status": "warning",
            "current": f"{sleep_hours} hrs/night",
            "target": "7-8 hrs/night",
            "tip": (
                "Oversleeping can lead to grogginess and reduced productivity. "
                "Try setting an alarm and getting 7-8 hours of quality sleep. "
                "Use the extra time for morning study or exercise."
            )
        })
        plan["weekly_goals"].append("😴 Set a consistent wake-up time with 7-8 hrs sleep")
    else:
        plan["recommendations"].append({
            "icon": "😴",
            "area": "Sleep",
            "status": "good",
            "current": f"{sleep_hours} hrs/night",
            "target": "7-8 hrs/night",
            "tip": "Good sleep habits! Keep maintaining a consistent sleep schedule."
        })

    # 📱 Social Media Usage
    if social_media > 3:
        plan["recommendations"].append({
            "icon": "📱",
            "area": "Social Media",
            "status": "critical",
            "current": f"{social_media} hrs/day",
            "target": "< 1.5 hrs/day",
            "tip": (
                "High social media usage is a major distraction! Use app timers "
                "(built into iOS/Android) to limit usage. Try 'no phone' zones "
                "during study sessions. Uninstall non-essential apps during exams."
            )
        })
        plan["weekly_goals"].append("📱 Reduce social media to under 2 hours per day")
    elif social_media > 2:
        plan["recommendations"].append({
            "icon": "📱",
            "area": "Social Media",
            "status": "warning",
            "current": f"{social_media} hrs/day",
            "target": "< 1.5 hrs/day",
            "tip": (
                "Consider reducing screen time. Set specific times for checking "
                "social media (e.g., only after completing study goals)."
            )
        })
        plan["weekly_goals"].append("📱 Set specific times for social media, not during study")
    else:
        plan["recommendations"].append({
            "icon": "📱",
            "area": "Social Media",
            "status": "good",
            "current": f"{social_media} hrs/day",
            "target": "< 1.5 hrs/day",
            "tip": "Good control over social media! Keep it minimal during exam prep."
        })

    # 📺 Netflix / Entertainment
    if netflix_hours > 2:
        plan["recommendations"].append({
            "icon": "📺",
            "area": "Entertainment (Netflix)",
            "status": "warning",
            "current": f"{netflix_hours} hrs/day",
            "target": "< 1 hr/day",
            "tip": (
                "Too much screen entertainment affects focus. Limit to 1 episode/day "
                "and use it as a reward after completing study goals."
            )
        })
        plan["weekly_goals"].append("📺 Limit Netflix/entertainment to 1 hour per day")

    # 🏫 Attendance
    if attendance < 75:
        plan["recommendations"].append({
            "icon": "🏫",
            "area": "Attendance",
            "status": "critical",
            "current": f"{attendance}%",
            "target": "> 85%",
            "tip": (
                "Low attendance directly hurts your grades! Classroom learning is "
                "irreplaceable — you get teacher explanations, doubt solving, and "
                "exam-relevant tips. Aim for 85%+ attendance starting this week."
            )
        })
        plan["weekly_goals"].append("🏫 Attend ALL classes this week — zero absences")
    elif attendance < 85:
        plan["recommendations"].append({
            "icon": "🏫",
            "area": "Attendance",
            "status": "warning",
            "current": f"{attendance}%",
            "target": "> 85%",
            "tip": (
                "Your attendance could be better. Try to attend at least 85% "
                "of classes. Sit in the front row for better focus."
            )
        })
        plan["weekly_goals"].append("🏫 Improve attendance to at least 85% this month")
    else:
        plan["recommendations"].append({
            "icon": "🏫",
            "area": "Attendance",
            "status": "good",
            "current": f"{attendance}%",
            "target": "> 85%",
            "tip": "Excellent attendance! Keep it up — consistency in class pays off."
        })

    # 🏃 Exercise
    if exercise < 2:
        plan["recommendations"].append({
            "icon": "🏃",
            "area": "Physical Activity",
            "status": "warning",
            "current": f"{exercise} days/week",
            "target": "3-4 days/week",
            "tip": (
                "Exercise improves memory, focus, and reduces stress. Start with "
                "a 20-minute walk or light workout 3 times a week. Even stretching helps!"
            )
        })
        plan["weekly_goals"].append("🏃 Exercise at least 3 times this week (even a brisk walk)")

    # 🧠 Mental Health
    if mental_health < 4:
        plan["recommendations"].append({
            "icon": "🧠",
            "area": "Mental Health",
            "status": "critical",
            "current": f"Rating: {mental_health}/10",
            "target": "7+/10",
            "tip": (
                "Your mental health rating is low. Please consider talking to a "
                "counselor, friend, or family member. Practice meditation or deep "
                "breathing for 10 minutes daily. Mental wellbeing is the foundation "
                "of academic success!"
            )
        })
        plan["weekly_goals"].append("🧠 Practice 10 minutes of meditation or journaling daily")

    # 🥗 Diet
    if isinstance(diet, str) and diet.lower() == "poor":
        plan["recommendations"].append({
            "icon": "🥗",
            "area": "Diet Quality",
            "status": "warning",
            "current": "Poor",
            "target": "Good / Balanced",
            "tip": (
                "A poor diet affects brain function and energy levels. Include "
                "more fruits, vegetables, nuts, and water in your daily diet. "
                "Avoid excessive junk food and sugary drinks."
            )
        })
        plan["weekly_goals"].append("🥗 Eat at least 2 servings of fruits/vegetables daily")

    return plan


def get_plan_summary_text(plan: dict) -> str:
    """Convert a study plan dict into a readable text summary."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  {plan['tier_label']}")
    lines.append(f"  Predicted Score: {plan['predicted_score']}/100")
    lines.append(f"{'='*60}")
    lines.append(f"\n{plan['summary']}\n")

    lines.append("📅 RECOMMENDED DAILY SCHEDULE:")
    lines.append("-" * 40)
    for time_slot, activity in plan["study_schedule"].items():
        lines.append(f"  {time_slot}: {activity}")

    if plan["recommendations"]:
        lines.append(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
        lines.append("-" * 40)
        for rec in plan["recommendations"]:
            status_icon = {"critical": "🔴", "warning": "🟡", "good": "🟢"}.get(rec["status"], "⚪")
            lines.append(f"\n  {rec['icon']} {rec['area']} {status_icon}")
            lines.append(f"     Current: {rec['current']}  →  Target: {rec['target']}")
            lines.append(f"     💡 {rec['tip']}")

    if plan["weekly_goals"]:
        lines.append(f"\n🎯 THIS WEEK'S GOALS:")
        lines.append("-" * 40)
        for goal in plan["weekly_goals"]:
            lines.append(f"  ☐ {goal}")

    lines.append(f"\n{plan['motivation']}")

    return "\n".join(lines)


# ─── Quick test ───
if __name__ == "__main__":
    # Test with a struggling student
    test_student = {
        "study_hours_per_day": 1.5,
        "sleep_hours": 5.0,
        "social_media_hours": 4.5,
        "netflix_hours": 3.0,
        "attendance_percentage": 65.0,
        "exercise_frequency": 1,
        "mental_health_rating": 3,
        "diet_quality": "Poor",
    }
    plan = generate_study_plan(35.0, test_student)
    print(get_plan_summary_text(plan))
