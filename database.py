"""
===================================================================
DATABASE.PY — Centralised SQLite path helper
===================================================================
All modules (spaced_repetition.py, gamification.py) import
get_db_path() from here so the database file location is
configured in exactly one place.
===================================================================
"""

import os


def get_db_path() -> str:
    """Return the absolute path to the SQLite database file."""
    return os.path.join(os.path.dirname(__file__), "studygenie.db")
