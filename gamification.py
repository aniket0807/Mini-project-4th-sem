"""
===================================================================
GAMIFICATION.PY — Streaks, XP & Badges
===================================================================
Provides backend state management for gamification.

SQLite Tables:
  users  — id, name, xp, level, current_streak, longest_streak,
            last_study_date, streak_freeze_count
  badges — id, user_id, badge_key, unlocked_at
  xp_log — id, user_id, amount, reason, timestamp

XP Formulas:
  Study session:         +50 × study_hours XP
  Correct review (q≥3):  +10 XP
  Perfect review (q=5):  +25 XP
  Streak bonus (7-day):  +100 XP
  Streak bonus (30-day): +500 XP
  Level                   floor(XP / 500) + 1

Streak Rules:
  - Gap = 0 days (same day): no change
  - Gap = 1 day:             current_streak += 1
  - Gap = 2 days + freeze:   consume 1 freeze, keep streak
  - Gap > 2 days:            reset to 1
  - Freeze refill:           1 freeze every 7 days of activity (auto)
  - Max freezes banked:      3
===================================================================
"""

import sqlite3
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from database import get_db_path


# ─────────────────────────────────────────────
#  BADGE REGISTRY
# ─────────────────────────────────────────────
BADGE_DEFINITIONS: Dict[str, Dict] = {
    # Streak badges
    "first_session":  {"name": "First Step 🌱",       "desc": "Logged your very first study session",          "icon": "🌱", "color": "#10B981"},
    "streak_3":       {"name": "On a Roll 🔥",         "desc": "Maintained a 3-day study streak",               "icon": "🔥", "color": "#F59E0B"},
    "streak_7":       {"name": "Week Warrior 🗓️",      "desc": "Maintained a 7-day study streak",               "icon": "🗓️", "color": "#EAB308"},
    "streak_14":      {"name": "Fortnight Fighter ⚔️", "desc": "Maintained a 14-day study streak",              "icon": "⚔️", "color": "#8B5CF6"},
    "streak_30":      {"name": "Iron Will 🏆",         "desc": "Maintained a 30-day study streak",              "icon": "🏆", "color": "#7C3AED"},
    "streak_100":     {"name": "Centurion 💯",          "desc": "Maintained a 100-day study streak",             "icon": "💯", "color": "#EF4444"},
    # Time-of-day badges
    "night_owl":      {"name": "Night Owl 🦉",         "desc": "Logged a study session after 10 PM",            "icon": "🦉", "color": "#6366F1"},
    "early_bird":     {"name": "Early Bird 🌅",        "desc": "Logged a study session before 6 AM",            "icon": "🌅", "color": "#F97316"},
    # Score badges
    "speed_demon":    {"name": "Speed Demon ⚡",       "desc": "Predicted score ≥ 90",                          "icon": "⚡", "color": "#06B6D4"},
    "overachiever":   {"name": "Overachiever 🌟",      "desc": "Predicted score ≥ 95",                          "icon": "🌟", "color": "#EAB308"},
    "perfect_score":  {"name": "Perfectionist 💎",     "desc": "Predicted score of 100",                        "icon": "💎", "color": "#EC4899"},
    # XP milestones
    "xp_500":         {"name": "Rising Star ⭐",       "desc": "Accumulated 500 XP",                            "icon": "⭐", "color": "#F59E0B"},
    "xp_1000":        {"name": "Dedicated Scholar 📚", "desc": "Accumulated 1,000 XP",                          "icon": "📚", "color": "#3B82F6"},
    "xp_5000":        {"name": "Grand Master 👑",      "desc": "Accumulated 5,000 XP",                          "icon": "👑", "color": "#7C3AED"},
    # Review badges
    "perfect_review": {"name": "Flawless Recall 🧠",  "desc": "Gave a perfect quality-5 review",               "icon": "🧠", "color": "#10B981"},
    "review_10":      {"name": "Consistent Learner 📖","desc": "Completed 10 spaced repetition reviews",         "icon": "📖", "color": "#6366F1"},
    # Study depth
    "marathon":       {"name": "Marathon Session 🏃",  "desc": "Logged a study session of 6+ hours",            "icon": "🏃", "color": "#EF4444"},
    "consistent":     {"name": "Consistent 📅",        "desc": "Logged sessions on 5 different days",           "icon": "📅", "color": "#22C55E"},
}

# XP configuration
XP_STUDY_PER_HOUR   = 50
XP_REVIEW_CORRECT   = 10
XP_REVIEW_PERFECT   = 25
XP_STREAK_7_BONUS   = 100
XP_STREAK_30_BONUS  = 500
XP_PER_LEVEL        = 500

# Streak configuration
MAX_STREAK_FREEZES  = 3
FREEZE_REFILL_DAYS  = 7   # earn 1 freeze every N study days


# ─────────────────────────────────────────────
#  DB SCHEMA
# ─────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    NOT NULL UNIQUE,
    xp                  INTEGER NOT NULL DEFAULT 0,
    level               INTEGER NOT NULL DEFAULT 1,
    current_streak      INTEGER NOT NULL DEFAULT 0,
    longest_streak      INTEGER NOT NULL DEFAULT 0,
    last_study_date     TEXT,
    streak_freeze_count INTEGER NOT NULL DEFAULT 1,
    total_session_days  INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS badges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    badge_key   TEXT    NOT NULL,
    unlocked_at TEXT    NOT NULL,
    UNIQUE(user_id, badge_key)
);

CREATE TABLE IF NOT EXISTS xp_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL,
    amount    INTEGER NOT NULL,
    reason    TEXT    NOT NULL,
    timestamp TEXT    NOT NULL
);
"""


# ─────────────────────────────────────────────
#  INIT DB
# ─────────────────────────────────────────────
def init_db(db_path: str = None) -> None:
    """Create all gamification tables if they don't exist."""
    if db_path is None:
        db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


# ─────────────────────────────────────────────
#  USER MANAGEMENT
# ─────────────────────────────────────────────
def get_or_create_user(name: str, db_path: str = None) -> Dict:
    """
    Retrieve a user profile by name, creating one if it doesn't exist.

    Returns a dict with all user fields.
    """
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    name = name.strip()
    now  = datetime.now().isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE name = ?", (name,)
        ).fetchone()

        if row is None:
            conn.execute(
                """INSERT INTO users
                   (name, xp, level, current_streak, longest_streak,
                    last_study_date, streak_freeze_count, total_session_days, created_at)
                   VALUES (?, 0, 1, 0, 0, NULL, 1, 0, ?)""",
                (name, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM users WHERE name = ?", (name,)
            ).fetchone()

    return dict(row)


# ─────────────────────────────────────────────
#  STREAK ENGINE
# ─────────────────────────────────────────────
def update_streak(user_id: int, study_date: date, db_path: str = None) -> Dict:
    """
    Update the user's streak based on the study_date.

    Streak rules:
      gap = 0  → same day, no change
      gap = 1  → consecutive day, streak + 1
      gap = 2  → grace period: if freeze available, consume it and keep streak
      gap > 2  → streak resets to 1

    Freeze refill: every FREEZE_REFILL_DAYS total session days earns +1 freeze (capped at MAX).

    Returns a dict describing what changed:
        {streak, longest_streak, freeze_used, streak_reset, freeze_count}
    """
    if db_path is None:
        db_path = get_db_path()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT current_streak, longest_streak, last_study_date, streak_freeze_count, total_session_days FROM users WHERE id=?",
            (user_id,),
        ).fetchone()

        current_streak  = user["current_streak"]
        longest_streak  = user["longest_streak"]
        last_date_str   = user["last_study_date"]
        freeze_count    = user["streak_freeze_count"]
        total_days      = user["total_session_days"]

        result = {
            "streak":         current_streak,
            "longest_streak": longest_streak,
            "freeze_used":    False,
            "streak_reset":   False,
            "freeze_count":   freeze_count,
            "bonus_xp":       0,
            "newly_achieved": [],
        }

        # Compute gap
        if last_date_str is None:
            gap = 1  # First ever session
        else:
            last_date = date.fromisoformat(last_date_str)
            gap = (study_date - last_date).days

        if gap == 0:
            # Already logged today — no streak change
            pass
        elif gap == 1:
            # Consecutive day
            current_streak += 1
            total_days     += 1
        elif gap == 2 and freeze_count > 0:
            # Grace period: burn a freeze
            freeze_count   -= 1
            current_streak += 1
            total_days     += 1
            result["freeze_used"] = True
        else:
            # Streak broken
            current_streak = 1
            total_days     += 1
            result["streak_reset"] = True

        # Update longest streak
        if current_streak > longest_streak:
            longest_streak = current_streak

        # Streak milestone XP
        if current_streak == 7:
            result["bonus_xp"] += XP_STREAK_7_BONUS
            result["newly_achieved"].append("streak_7")
        if current_streak == 14:
            result["newly_achieved"].append("streak_14")
        if current_streak == 30:
            result["bonus_xp"] += XP_STREAK_30_BONUS
            result["newly_achieved"].append("streak_30")
        if current_streak == 3:
            result["newly_achieved"].append("streak_3")
        if current_streak == 100:
            result["newly_achieved"].append("streak_100")

        # Freeze refill every FREEZE_REFILL_DAYS
        if total_days > 0 and total_days % FREEZE_REFILL_DAYS == 0:
            if freeze_count < MAX_STREAK_FREEZES:
                freeze_count += 1

        conn.execute(
            """UPDATE users SET current_streak=?, longest_streak=?,
               last_study_date=?, streak_freeze_count=?, total_session_days=?
               WHERE id=?""",
            (current_streak, longest_streak, study_date.isoformat(),
             freeze_count, total_days, user_id),
        )
        conn.commit()

    result["streak"]         = current_streak
    result["longest_streak"] = longest_streak
    result["freeze_count"]   = freeze_count
    return result


# ─────────────────────────────────────────────
#  XP ENGINE
# ─────────────────────────────────────────────
def _award_xp(user_id: int, amount: int, reason: str, db_path: str, conn: sqlite3.Connection) -> int:
    """
    Award XP and update level. Uses an existing connection (no commit here).
    Returns new total XP.
    """
    conn.execute(
        "UPDATE users SET xp = xp + ? WHERE id = ?", (amount, user_id)
    )
    conn.execute(
        "INSERT INTO xp_log (user_id, amount, reason, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, amount, reason, datetime.now().isoformat()),
    )
    new_xp = conn.execute("SELECT xp FROM users WHERE id=?", (user_id,)).fetchone()[0]
    new_level = (new_xp // XP_PER_LEVEL) + 1
    conn.execute("UPDATE users SET level=? WHERE id=?", (new_level, user_id))
    return new_xp


# ─────────────────────────────────────────────
#  BADGE ENGINE
# ─────────────────────────────────────────────
def check_and_award_badges(
    user_id: int,
    event: str,
    context: Dict,
    db_path: str = None,
) -> List[str]:
    """
    Event-driven badge engine. Call after any meaningful user action.

    Parameters
    ----------
    user_id : int
    event   : str — one of: 'study_session', 'review', 'xp_update', 'streak_update'
    context : dict — event-specific data (e.g. predicted_score, study_hours, quality, xp, streak)

    Returns a list of newly unlocked badge_keys.
    """
    if db_path is None:
        db_path = get_db_path()

    def has_badge(conn, key):
        return conn.execute(
            "SELECT 1 FROM badges WHERE user_id=? AND badge_key=?", (user_id, key)
        ).fetchone() is not None

    def award(conn, key):
        try:
            conn.execute(
                "INSERT INTO badges (user_id, badge_key, unlocked_at) VALUES (?, ?, ?)",
                (user_id, key, datetime.now().isoformat()),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    new_badges = []
    now_hour   = datetime.now().hour

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        if event == "study_session":
            predicted_score = context.get("predicted_score", 0)
            study_hours     = context.get("study_hours", 0)
            total_days      = context.get("total_session_days", 0)

            # First session ever
            if total_days <= 1 and not has_badge(conn, "first_session"):
                if award(conn, "first_session"):
                    new_badges.append("first_session")

            # Time-of-day
            if now_hour >= 22 and not has_badge(conn, "night_owl"):
                if award(conn, "night_owl"):
                    new_badges.append("night_owl")
            if now_hour < 6 and not has_badge(conn, "early_bird"):
                if award(conn, "early_bird"):
                    new_badges.append("early_bird")

            # Score badges
            if predicted_score >= 100 and not has_badge(conn, "perfect_score"):
                if award(conn, "perfect_score"):
                    new_badges.append("perfect_score")
            elif predicted_score >= 95 and not has_badge(conn, "overachiever"):
                if award(conn, "overachiever"):
                    new_badges.append("overachiever")
            elif predicted_score >= 90 and not has_badge(conn, "speed_demon"):
                if award(conn, "speed_demon"):
                    new_badges.append("speed_demon")

            # Marathon session
            if study_hours >= 6 and not has_badge(conn, "marathon"):
                if award(conn, "marathon"):
                    new_badges.append("marathon")

            # Consistent (5 session days)
            if total_days >= 5 and not has_badge(conn, "consistent"):
                if award(conn, "consistent"):
                    new_badges.append("consistent")

        elif event == "review":
            quality       = context.get("quality", 0)
            total_reviews = context.get("total_reviews", 0)

            if quality == 5 and not has_badge(conn, "perfect_review"):
                if award(conn, "perfect_review"):
                    new_badges.append("perfect_review")

            if total_reviews >= 10 and not has_badge(conn, "review_10"):
                if award(conn, "review_10"):
                    new_badges.append("review_10")

        elif event == "streak_update":
            streak = context.get("streak", 0)
            for key, threshold in [("streak_3", 3), ("streak_7", 7), ("streak_14", 14),
                                   ("streak_30", 30), ("streak_100", 100)]:
                if streak >= threshold and not has_badge(conn, key):
                    if award(conn, key):
                        new_badges.append(key)

        elif event == "xp_update":
            xp = context.get("xp", 0)
            for key, threshold in [("xp_500", 500), ("xp_1000", 1000), ("xp_5000", 5000)]:
                if xp >= threshold and not has_badge(conn, key):
                    if award(conn, key):
                        new_badges.append(key)

        conn.commit()

    return new_badges


# ─────────────────────────────────────────────
#  LOG STUDY SESSION (main entry point from UI)
# ─────────────────────────────────────────────
def log_study_session(
    user_id: int,
    study_hours: float,
    predicted_score: float,
    db_path: str = None,
) -> Dict:
    """
    Called whenever a user submits the Predict & Plan form.
    Awards XP, updates streak, checks badges.

    Returns a summary dict:
        xp_earned, new_xp, level, streak, longest_streak,
        freeze_count, new_badges, streak_info
    """
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    today = date.today()

    # Streak update
    streak_info = update_streak(user_id, today, db_path)

    with sqlite3.connect(db_path) as conn:
        user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        total_days = user[8]  # total_session_days column

        # Award session XP
        session_xp = int(XP_STUDY_PER_HOUR * study_hours)
        new_xp     = _award_xp(user_id, session_xp, f"Study session ({study_hours:.1f} hrs)", db_path=db_path, conn=conn)

        # Award streak bonus XP if milestone hit
        if streak_info["bonus_xp"] > 0:
            new_xp = _award_xp(user_id, streak_info["bonus_xp"], f"Streak milestone bonus", db_path=db_path, conn=conn)

        conn.commit()
        level = conn.execute("SELECT level FROM users WHERE id=?", (user_id,)).fetchone()[0]

    # Badge checks
    session_badges  = check_and_award_badges(user_id, "study_session", {
        "predicted_score":   predicted_score,
        "study_hours":       study_hours,
        "total_session_days": total_days,
    }, db_path)
    streak_badges   = check_and_award_badges(user_id, "streak_update", {
        "streak": streak_info["streak"],
    }, db_path)
    xp_badges       = check_and_award_badges(user_id, "xp_update", {"xp": new_xp}, db_path)

    all_new_badges = session_badges + streak_badges + streak_info.get("newly_achieved", []) + xp_badges
    # Deduplicate
    seen = set()
    unique_badges = []
    for b in all_new_badges:
        if b not in seen:
            seen.add(b)
            unique_badges.append(b)

    return {
        "xp_earned":      int(XP_STUDY_PER_HOUR * study_hours) + streak_info["bonus_xp"],
        "new_xp":         new_xp,
        "level":          level,
        "streak":         streak_info["streak"],
        "longest_streak": streak_info["longest_streak"],
        "freeze_count":   streak_info["freeze_count"],
        "new_badges":     unique_badges,
        "streak_info":    streak_info,
    }


# ─────────────────────────────────────────────
#  LOG A REVIEW (called from Review Schedule page)
# ─────────────────────────────────────────────
def log_review_xp(user_id: int, quality: int, db_path: str = None) -> Tuple[int, List[str]]:
    """
    Award XP for completing a spaced repetition review.
    Returns (xp_earned, new_badges).
    """
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    if quality == 5:
        xp_amount = XP_REVIEW_PERFECT
    elif quality >= 3:
        xp_amount = XP_REVIEW_CORRECT
    else:
        xp_amount = 0

    with sqlite3.connect(db_path) as conn:
        new_xp = _award_xp(user_id, xp_amount, f"Review completed (quality={quality})", db_path, conn)
        total_reviews = conn.execute(
            "SELECT COUNT(*) FROM xp_log WHERE user_id=? AND reason LIKE 'Review%'", (user_id,)
        ).fetchone()[0]
        conn.commit()

    badges = check_and_award_badges(user_id, "review", {
        "quality":       quality,
        "total_reviews": total_reviews,
    }, db_path)
    xp_badges = check_and_award_badges(user_id, "xp_update", {"xp": new_xp}, db_path)

    return xp_amount, list(set(badges + xp_badges))


# ─────────────────────────────────────────────
#  USER STATS (for Achievements page)
# ─────────────────────────────────────────────
def get_user_stats(user_id: int, db_path: str = None) -> Dict:
    """
    Return a comprehensive stats dict for the Achievements UI page.
    """
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if user is None:
            return {}
        user_dict = dict(user)

        # Earned badges
        badge_rows = conn.execute(
            "SELECT badge_key, unlocked_at FROM badges WHERE user_id=? ORDER BY unlocked_at",
            (user_id,),
        ).fetchall()
        earned_badges = {r["badge_key"]: r["unlocked_at"] for r in badge_rows}

        # XP log (last 15 entries)
        xp_log = conn.execute(
            "SELECT amount, reason, timestamp FROM xp_log WHERE user_id=? ORDER BY timestamp DESC LIMIT 15",
            (user_id,),
        ).fetchall()

    # Build full badge list with earned status
    all_badges = []
    for key, defn in BADGE_DEFINITIONS.items():
        all_badges.append({
            "key":        key,
            "name":       defn["name"],
            "desc":       defn["desc"],
            "icon":       defn["icon"],
            "color":      defn["color"],
            "earned":     key in earned_badges,
            "earned_at":  earned_badges.get(key),
        })

    # Level progress
    xp       = user_dict["xp"]
    level    = user_dict["level"]
    xp_floor = (level - 1) * XP_PER_LEVEL
    xp_ceil  = level * XP_PER_LEVEL
    xp_pct   = round(((xp - xp_floor) / XP_PER_LEVEL) * 100, 1)

    return {
        **user_dict,
        "all_badges":  all_badges,
        "xp_log":      [dict(r) for r in xp_log],
        "level":       level,
        "xp_pct":      xp_pct,
        "xp_floor":    xp_floor,
        "xp_ceil":     xp_ceil,
        "badges_earned": len(earned_badges),
        "badges_total":  len(BADGE_DEFINITIONS),
    }



# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    tmp_db = os.path.join(tempfile.gettempdir(), "gamification_test.db")
    print("=== Gamification Smoke Test ===")
    print(f"DB: {tmp_db}\n")

    init_db(tmp_db)

    # Create user
    user = get_or_create_user("TestStudent", tmp_db)
    uid  = user["id"]
    print(f"✅ User created: id={uid}, name={user['name']}")

    # Log 3 study sessions over 3 days (simulated)
    from datetime import date, timedelta
    for i, (hrs, score) in enumerate([(2.5, 72.4), (4.0, 81.1), (6.0, 91.8)]):
        session = log_study_session(uid, hrs, score, tmp_db)
        print(f"\n  Day {i+1}: {hrs}h study, predicted={score}")
        print(f"    XP earned:   {session['xp_earned']}")
        print(f"    Streak:      {session['streak']}")
        print(f"    New badges:  {session['new_badges']}")

    # Log reviews
    for q in [5, 4, 2, 5]:
        xp, badges = log_review_xp(uid, q, tmp_db)
        print(f"\n  Review q={q}: +{xp} XP | badges: {badges}")

    # Stats
    stats = get_user_stats(uid, tmp_db)
    print(f"\n📊 User Stats:")
    print(f"  XP: {stats['xp']} | Level: {stats['level']} ({stats['xp_pct']}% to next)")
    print(f"  Streak: {stats['current_streak']} | Longest: {stats['longest_streak']}")
    print(f"  Badges earned: {stats['badges_earned']} / {stats['badges_total']}")
    for b in [x for x in stats["all_badges"] if x["earned"]]:
        print(f"    {b['icon']} {b['name']}")

    print("\n✅ Gamification smoke test passed!")
