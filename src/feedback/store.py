"""SQLite blacklist / suppression store (§9).

- blacklist: permanent drops from WRONG_PERSON / BOUNCE outcomes.
- suppression: time-boxed drops from NOT_RECRUITING (now + 365 days).
- outcomes: raw append-only log for weight learning.

All identity keyed on OpenAlex author IDs.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import config

SUPPRESSION_DAYS = 365


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.FEEDBACK_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS blacklist (
                supervisor_id TEXT PRIMARY KEY,
                reason TEXT,
                added_at TEXT
            );
            CREATE TABLE IF NOT EXISTS suppression (
                supervisor_id TEXT PRIMARY KEY,
                until TEXT
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                student_id TEXT,
                supervisor_id TEXT,
                institution TEXT,
                area TEXT,
                sent_at TEXT,
                outcome TEXT
            );
            """
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_blacklist(supervisor_id: str, reason: str, db_path: Path | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO blacklist(supervisor_id, reason, added_at) VALUES (?,?,?) "
            "ON CONFLICT(supervisor_id) DO UPDATE SET reason=excluded.reason",
            (supervisor_id, reason, _now_iso()),
        )


def add_suppression(supervisor_id: str, db_path: Path | None = None) -> None:
    until = (datetime.now(timezone.utc) + timedelta(days=SUPPRESSION_DAYS)).isoformat()
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO suppression(supervisor_id, until) VALUES (?,?) "
            "ON CONFLICT(supervisor_id) DO UPDATE SET until=excluded.until",
            (supervisor_id, until),
        )


def append_outcome(row: dict, db_path: Path | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO outcomes(student_id, supervisor_id, institution, area, sent_at, outcome) "
            "VALUES (?,?,?,?,?,?)",
            (
                row.get("student_id"),
                row.get("supervisor_id"),
                row.get("institution"),
                row.get("area"),
                row.get("sent_at"),
                row.get("outcome"),
            ),
        )


def excluded_ids(db_path: Path | None = None) -> set[str]:
    """Set of author IDs to drop: all blacklisted + suppressions still within TTL."""
    path = db_path or config.FEEDBACK_DB_PATH
    if not Path(path).exists():
        return set()
    out: set[str] = set()
    now = _now_iso()
    with closing(_connect(db_path)) as conn:
        out.update(r["supervisor_id"] for r in conn.execute("SELECT supervisor_id FROM blacklist"))
        out.update(
            r["supervisor_id"]
            for r in conn.execute("SELECT supervisor_id, until FROM suppression WHERE until > ?", (now,))
        )
    return out


def all_outcomes(db_path: Path | None = None) -> list[dict]:
    path = db_path or config.FEEDBACK_DB_PATH
    if not Path(path).exists():
        return []
    with closing(_connect(db_path)) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM outcomes")]
