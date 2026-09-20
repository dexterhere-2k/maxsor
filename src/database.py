from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from . import config

_ACTION_RE = re.compile(r"^[A-Z][A-Z_]*$")

def _action_sql_list() -> str:
    unexpected = sorted(a for a in config.ACTIONS if not _ACTION_RE.match(a))
    if unexpected:
        raise ValueError(f"refusing to inline unexpected action values into SQL: {unexpected}")
    return ", ".join(f"'{action}'" for action in sorted(config.ACTIONS))

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS tickets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message             TEXT    NOT NULL,
    order_value_inr     REAL,
    days_since_delivery INTEGER,
    days_since_dispatch INTEGER,
    product_type        TEXT,
    opened_status       TEXT,
    order_status        TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id     INTEGER NOT NULL UNIQUE REFERENCES tickets(id) ON DELETE CASCADE,
    action        TEXT    NOT NULL CHECK (action IN ({_action_sql_list()})),
    reason        TEXT    NOT NULL,
    confidence    REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    sources       TEXT    NOT NULL,
    path          TEXT    NOT NULL CHECK (path IN ('cag', 'fallback')),
    prompt_tokens INTEGER,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON tickets(user_id);
"""

_TICKET_SELECT = """
SELECT t.id,
       t.message,
       t.order_value_inr,
       t.days_since_delivery,
       t.days_since_dispatch,
       t.product_type,
       t.opened_status,
       t.order_status,
       t.created_at,
       d.action        AS decision_action,
       d.reason        AS decision_reason,
       d.confidence    AS decision_confidence,
       d.sources       AS decision_sources,
       d.path          AS decision_path,
       d.prompt_tokens AS decision_prompt_tokens
FROM tickets t
LEFT JOIN decisions d ON d.ticket_id = t.id
"""

_DECISION_KEYS = (
    "decision_action",
    "decision_reason",
    "decision_confidence",
    "decision_sources",
    "decision_path",
    "decision_prompt_tokens",
)

def _resolve(db_path: str | None) -> str:
    return str(db_path or config.DATABASE_PATH)

def get_conn(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(_resolve(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

@contextmanager
def _connection(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = get_conn(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db(db_path: str | None = None) -> None:
    conn = get_conn(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()

def _row_to_ticket(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    decision = None
    if data.get("decision_action") is not None:
        decision = {
            "action": data["decision_action"],
            "reason": data["decision_reason"],
            "confidence": data["decision_confidence"],
            "sources": json.loads(data["decision_sources"] or "[]"),
            "path": data["decision_path"],
            "prompt_tokens": data["decision_prompt_tokens"],
        }
    for key in _DECISION_KEYS:
        data.pop(key, None)
    data["decision"] = decision
    return data

def create_user(email: str, password_hash: str, db_path: str | None = None) -> dict[str, Any]:
    with _connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?);",
            (email.strip().lower(), password_hash),
        )
        row = conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?;", (cursor.lastrowid,)
        ).fetchone()
    return dict(row)

def get_user_by_email(email: str, db_path: str | None = None) -> dict[str, Any] | None:
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?;",
            (email.strip().lower(),),
        ).fetchone()
    return dict(row) if row else None

def get_user_by_id(user_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?;", (user_id,)
        ).fetchone()
    return dict(row) if row else None

def insert_ticket(
    user_id: int,
    ticket: dict[str, Any],
    decision: dict[str, Any],
    db_path: str | None = None,
) -> dict[str, Any]:
    with _connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO tickets (
                user_id, message, order_value_inr, days_since_delivery,
                days_since_dispatch, product_type, opened_status, order_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                user_id,
                ticket.get("message", ""),
                ticket.get("order_value_inr"),
                ticket.get("days_since_delivery"),
                ticket.get("days_since_dispatch"),
                ticket.get("product_type"),
                ticket.get("opened_status"),
                ticket.get("order_status"),
            ),
        )
        ticket_id = cursor.lastrowid
        conn.execute(
            """
            INSERT INTO decisions (
                ticket_id, action, reason, confidence, sources, path, prompt_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                ticket_id,
                decision["action"],
                decision["reason"],
                float(decision["confidence"]),
                json.dumps(list(decision.get("sources") or [])),
                decision["path"],
                decision.get("prompt_tokens"),
            ),
        )
        row = conn.execute(
            _TICKET_SELECT + " WHERE t.id = ? AND t.user_id = ?;", (ticket_id, user_id)
        ).fetchone()
    if row is None:
        raise RuntimeError(f"ticket {ticket_id} disappeared immediately after insert")
    return _row_to_ticket(row)

def list_tickets(user_id: int, db_path: str | None = None) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute(
            _TICKET_SELECT + " WHERE t.user_id = ? ORDER BY t.id DESC;", (user_id,)
        ).fetchall()
    return [_row_to_ticket(row) for row in rows]

def get_ticket(
    user_id: int, ticket_id: int, db_path: str | None = None
) -> dict[str, Any] | None:
    with _connection(db_path) as conn:
        row = conn.execute(
            _TICKET_SELECT + " WHERE t.id = ? AND t.user_id = ?;", (ticket_id, user_id)
        ).fetchone()
    return _row_to_ticket(row) if row else None

if __name__ == "__main__":
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = str(pathlib.Path(folder) / "self-check.db")
        init_db(path)
        with get_conn(path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert {"users", "tickets", "decisions"} <= tables, f"missing tables: {sorted(tables)}"
        assert journal.lower() == "wal", f"expected WAL, got {journal}"
        user = create_user("self-check@example.com", "not-a-real-hash", path)
        assert get_user_by_email("self-check@example.com", path)["id"] == user["id"]
        assert list_tickets(user["id"], path) == []
        assert get_ticket(user["id"] + 1, 1, path) is None, "another user must not read the row"
        print(f"tables {sorted(tables)}, journal {journal}, tenant filter scoped")
