"""
===================================================================
SPACED_REPETITION.PY — SM-2 Spaced Repetition Scheduling Engine
===================================================================
Implements the SuperMemo SM-2 algorithm for optimal review scheduling.

Algorithm Reference: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2

Data Schema (SQLite table: review_items):
  - id              INTEGER PK
  - user_id         TEXT    (student name)
  - topic           TEXT    (subject / flashcard topic)
  - ease_factor     REAL    (default 2.5; adjusted by performance quality)
  - interval        INTEGER (days until next review)
  - repetitions     INTEGER (consecutive successful reviews ≥ 3)
  - next_review_date TEXT   (ISO date: YYYY-MM-DD)
  - last_score      INTEGER (0-5 quality rating of last review)
  - created_at      TEXT    (ISO timestamp)

SM-2 Quality Scale:
  5 — Perfect response
  4 — Correct with slight hesitation
  3 — Correct but required significant effort
  2 — Incorrect; the correct answer was easy to recall
  1 — Incorrect; recalled after seeing the answer
  0 — Complete blackout
===================================================================
"""

import sqlite3
import os
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional

from database import get_db_path


# ─────────────────────────────────────────────
#  DB SCHEMA
# ─────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL,
    topic            TEXT    NOT NULL,
    ease_factor      REAL    NOT NULL DEFAULT 2.5,
    interval         INTEGER NOT NULL DEFAULT 1,
    repetitions      INTEGER NOT NULL DEFAULT 0,
    next_review_date TEXT    NOT NULL,
    last_score       INTEGER,
    created_at       TEXT    NOT NULL,
    UNIQUE(user_id, topic)
);
"""


# ─────────────────────────────────────────────
#  INITIALISE DB
# ─────────────────────────────────────────────
def init_db(db_path: str = None) -> None:
    """Create the review_items table if it doesn't exist."""
    if db_path is None:
        db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


# ─────────────────────────────────────────────
#  ADD / REGISTER A NEW TOPIC
# ─────────────────────────────────────────────
def add_topic(user_id: str, topic: str, db_path: str = None) -> bool:
    """
    Register a new topic for spaced repetition review.
    The first review is scheduled for today.

    Returns True if added, False if topic already exists.
    """
    if db_path is None:
        db_path = get_db_path()

    init_db(db_path)
    today = date.today().isoformat()
    now   = datetime.now().isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO review_items
                    (user_id, topic, ease_factor, interval, repetitions,
                     next_review_date, last_score, created_at)
                VALUES (?, ?, 2.5, 1, 0, ?, NULL, ?)
                """,
                (user_id, topic.strip(), today, now),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Already exists


# ─────────────────────────────────────────────
#  GET ITEMS DUE TODAY
# ─────────────────────────────────────────────
def get_due_items(user_id: str, db_path: str = None) -> List[Dict]:
    """
    Return all review items due today or overdue for the given user.

    Returns a list of dicts with keys:
        id, topic, ease_factor, interval, repetitions,
        next_review_date, last_score, days_overdue
    """
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    today = date.today().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, topic, ease_factor, interval, repetitions,
                   next_review_date, last_score
            FROM review_items
            WHERE user_id = ? AND next_review_date <= ?
            ORDER BY next_review_date ASC
            """,
            (user_id, today),
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        try:
            due_date = date.fromisoformat(d["next_review_date"])
            d["days_overdue"] = max(0, (date.today() - due_date).days)
        except Exception:
            d["days_overdue"] = 0
        result.append(d)
    return result


# ─────────────────────────────────────────────
#  GET ALL ITEMS FOR A USER
# ─────────────────────────────────────────────
def get_all_items(user_id: str, db_path: str = None) -> List[Dict]:
    """Return all review items (due and future) for a user."""
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, topic, ease_factor, interval, repetitions,
                   next_review_date, last_score
            FROM review_items
            WHERE user_id = ?
            ORDER BY next_review_date ASC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
#  SM-2 CORE UPDATE
# ─────────────────────────────────────────────
def _sm2_update(ease_factor: float, interval: int, repetitions: int, quality: int):
    """
    Apply the SM-2 algorithm update.

    Parameters
    ----------
    ease_factor  : current EF value (≥ 1.3)
    interval     : current interval in days
    repetitions  : consecutive successful reviews (quality ≥ 3)
    quality      : review quality 0–5

    Returns
    -------
    (new_ease_factor, new_interval, new_repetitions)
    """
    # Update ease factor
    new_ef = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)  # EF floor at 1.3

    if quality < 3:
        # Failed review — reset repetitions, restart from interval=1
        new_repetitions = 0
        new_interval = 1
    else:
        # Successful review
        new_repetitions = repetitions + 1
        if new_repetitions == 1:
            new_interval = 1
        elif new_repetitions == 2:
            new_interval = 6
        else:
            new_interval = round(interval * new_ef)

    return round(new_ef, 4), new_interval, new_repetitions


# ─────────────────────────────────────────────
#  UPDATE REVIEW (call after user rates an item)
# ─────────────────────────────────────────────
def update_review(
    user_id: str,
    topic: str,
    quality: int,
    db_path: str = None,
) -> Dict:
    """
    Apply SM-2 update for a reviewed item and persist the new schedule.

    Parameters
    ----------
    user_id : str
    topic   : str — exact topic string
    quality : int — 0 to 5 rating

    Returns
    -------
    dict with updated fields: ease_factor, interval, repetitions,
                              next_review_date, quality
    """
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    quality = max(0, min(5, int(quality)))

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT ease_factor, interval, repetitions FROM review_items WHERE user_id=? AND topic=?",
            (user_id, topic),
        ).fetchone()

        if row is None:
            raise ValueError(f"Topic '{topic}' not found for user '{user_id}'")

        ef, iv, reps = row
        new_ef, new_iv, new_reps = _sm2_update(ef, iv, reps, quality)
        next_date = (date.today() + timedelta(days=new_iv)).isoformat()

        conn.execute(
            """
            UPDATE review_items
            SET ease_factor=?, interval=?, repetitions=?, next_review_date=?, last_score=?
            WHERE user_id=? AND topic=?
            """,
            (new_ef, new_iv, new_reps, next_date, quality, user_id, topic),
        )
        conn.commit()

    return {
        "topic": topic,
        "ease_factor": new_ef,
        "interval": new_iv,
        "repetitions": new_reps,
        "next_review_date": next_date,
        "quality": quality,
    }


# ─────────────────────────────────────────────
#  RETENTION FEATURES → feed into XGBoost
# ─────────────────────────────────────────────
def get_retention_features(user_id: str, db_path: str = None) -> Dict:
    """
    Compute retention-related features for a user and return them as a dict
    that can be merged into the XGBoost prediction input.

    Features returned:
        avg_ease_factor   — mean EF across all items (proxy for overall retention)
        overdue_count     — number of items past their review date
        avg_interval      — mean review interval in days (proxy for mastery level)
        topics_mastered   — count of items with repetitions ≥ 5
        topics_total      — total topics being tracked
        mastery_ratio     — topics_mastered / max(1, topics_total)
    """
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)

    today = date.today().isoformat()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ease_factor, interval, repetitions, next_review_date FROM review_items WHERE user_id=?",
            (user_id,),
        ).fetchall()

    if not rows:
        return {
            "avg_ease_factor":  2.5,
            "overdue_count":    0,
            "avg_interval":     1,
            "topics_mastered":  0,
            "topics_total":     0,
            "mastery_ratio":    0.0,
        }

    efs       = [r[0] for r in rows]
    intervals = [r[1] for r in rows]
    reps      = [r[2] for r in rows]
    due_dates = [r[3] for r in rows]

    overdue   = sum(1 for d in due_dates if d <= today)
    mastered  = sum(1 for r in reps if r >= 5)
    total     = len(rows)

    return {
        "avg_ease_factor": round(sum(efs) / total, 4),
        "overdue_count":   overdue,
        "avg_interval":    round(sum(intervals) / total, 2),
        "topics_mastered": mastered,
        "topics_total":    total,
        "mastery_ratio":   round(mastered / total, 4),
    }


# ─────────────────────────────────────────────
#  DELETE A TOPIC
# ─────────────────────────────────────────────
def delete_topic(user_id: str, topic: str, db_path: str = None) -> bool:
    """Remove a topic from the review schedule. Returns True if deleted."""
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM review_items WHERE user_id=? AND topic=?",
            (user_id, topic),
        )
        conn.commit()
    return cursor.rowcount > 0


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    tmp_db = os.path.join(tempfile.gettempdir(), "sr_test.db")
    print("=== Spaced Repetition Smoke Test ===")
    print(f"DB: {tmp_db}\n")

    init_db(tmp_db)

    # Add topics
    add_topic("Alice", "Mathematics", tmp_db)
    add_topic("Alice", "Data Structures", tmp_db)
    add_topic("Alice", "Operating Systems", tmp_db)
    print("✅ Added 3 topics for Alice")

    # Simulate reviews
    r1 = update_review("Alice", "Mathematics", quality=5, db_path=tmp_db)
    print(f"  Math review (q=5): interval={r1['interval']}d, EF={r1['ease_factor']}, next={r1['next_review_date']}")

    r2 = update_review("Alice", "Data Structures", quality=2, db_path=tmp_db)
    print(f"  DS review   (q=2): interval={r2['interval']}d, EF={r2['ease_factor']}, next={r2['next_review_date']}")

    r3 = update_review("Alice", "Operating Systems", quality=4, db_path=tmp_db)
    print(f"  OS review   (q=4): interval={r3['interval']}d, EF={r3['ease_factor']}, next={r3['next_review_date']}")

    # Retention features
    feats = get_retention_features("Alice", tmp_db)
    print(f"\n📊 Retention features: {feats}")

    # Due items (none should be due since we just reviewed them)
    due = get_due_items("Alice", tmp_db)
    print(f"\n📅 Items due today: {len(due)}")

    print("\n✅ Spaced Repetition smoke test passed!")
