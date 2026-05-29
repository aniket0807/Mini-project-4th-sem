"""
===================================================================
AUTH.PY — User Authentication Module
===================================================================
Provides secure email/password authentication backed by SQLite.

Features:
  • bcrypt password hashing (industry standard, salted)
  • Email + display name registration
  • Login / logout session management via st.session_state
  • User profile stored in the shared studygenie.db

SQLite Table:
  auth_users — id, email, display_name, password_hash, created_at
===================================================================
"""

import sqlite3
import bcrypt
import re
from datetime import datetime
from typing import Optional, Dict
import streamlit as st

from database import get_db_path


# ─────────────────────────────────────────────
#  SCHEMA
# ─────────────────────────────────────────────
_AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
"""


# ─────────────────────────────────────────────
#  DB INIT
# ─────────────────────────────────────────────
def init_auth_db(db_path: str = None) -> None:
    """Create the auth_users table if it doesn't exist."""
    if db_path is None:
        db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_AUTH_SCHEMA)
        conn.commit()


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _validate_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, email.strip()))


def _hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt. Returns a UTF-8 string."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────
def register_user(
    email: str,
    display_name: str,
    password: str,
    db_path: str = None,
) -> Dict:
    """
    Register a new user.

    Returns:
        {"success": True, "user": {...}}
        {"success": False, "error": "<reason>"}
    """
    if db_path is None:
        db_path = get_db_path()
    init_auth_db(db_path)

    email        = email.strip().lower()
    display_name = display_name.strip()
    password     = password.strip()

    # Validations
    if not _validate_email(email):
        return {"success": False, "error": "Please enter a valid email address."}
    if len(display_name) < 2:
        return {"success": False, "error": "Display name must be at least 2 characters."}
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    pw_hash = _hash_password(password)
    now     = datetime.now().isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """INSERT INTO auth_users (email, display_name, password_hash, created_at)
                   VALUES (?, ?, ?, ?)""",
                (email, display_name, pw_hash, now),
            )
            conn.commit()
            user = conn.execute(
                "SELECT id, email, display_name, created_at FROM auth_users WHERE email=?",
                (email,),
            ).fetchone()
        return {"success": True, "user": dict(user)}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "An account with this email already exists."}
    except Exception as e:
        return {"success": False, "error": f"Registration failed: {e}"}


def login_user(email: str, password: str, db_path: str = None) -> Dict:
    """
    Authenticate a user.

    Returns:
        {"success": True, "user": {...}}
        {"success": False, "error": "<reason>"}
    """
    if db_path is None:
        db_path = get_db_path()
    init_auth_db(db_path)

    email    = email.strip().lower()
    password = password.strip()

    if not email or not password:
        return {"success": False, "error": "Email and password are required."}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, email, display_name, password_hash, created_at FROM auth_users WHERE email=?",
            (email,),
        ).fetchone()

    if row is None:
        return {"success": False, "error": "No account found with this email."}

    if not _verify_password(password, row["password_hash"]):
        return {"success": False, "error": "Incorrect password."}

    user = {
        "id":           row["id"],
        "email":        row["email"],
        "display_name": row["display_name"],
        "created_at":   row["created_at"],
    }
    return {"success": True, "user": user}


def get_user_by_id(user_id: int, db_path: str = None) -> Optional[Dict]:
    """Fetch a user by primary key. Returns None if not found."""
    if db_path is None:
        db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, email, display_name, created_at FROM auth_users WHERE id=?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────
#  STREAMLIT SESSION HELPERS
# ─────────────────────────────────────────────
def session_login(user: Dict) -> None:
    """Persist the authenticated user into Streamlit session state."""
    st.session_state["auth_user"]     = user
    st.session_state["authenticated"] = True
    # Keep display name in sync with the rest of the app
    st.session_state["student_name"]  = user["display_name"]


def session_logout() -> None:
    """Clear authentication from session state."""
    for key in ("auth_user", "authenticated", "student_name"):
        st.session_state.pop(key, None)


def get_session_user() -> Optional[Dict]:
    """Return the currently logged-in user dict, or None."""
    if st.session_state.get("authenticated"):
        return st.session_state.get("auth_user")
    return None


def is_authenticated() -> bool:
    """True if a user is logged in."""
    return bool(st.session_state.get("authenticated"))
