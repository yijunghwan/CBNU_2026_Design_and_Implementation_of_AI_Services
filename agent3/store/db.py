"""장기기억 저장소: agent2 전용 SQLite (stdlib sqlite3, 완전 자체 소유).

테이블
- users       : user_code별 표준화 프로필 + 동의
- messages    : 장기 대화 기록
- summaries   : user_code별 장기 요약본(단일 레코드)

프로필 필드는 맵핑 테이블 표준값을 그대로 저장한다.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from agent3.config import settings

_DB_PATH = settings.policy_docs_path.parent / "agent2.db"

_PROFILE_FIELDS = [
    "age", "region", "income", "employment_status",
    "marriage_status", "children_count", "housing_status",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LongTermStore:
    """SQLite 기반 장기기억. 스레드 안전(락 + check_same_thread=False)."""

    def __init__(self, db_path: Optional[Path] = None):
        self._path = Path(db_path or _DB_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    user_code TEXT UNIQUE NOT NULL,
                    age INTEGER, region TEXT, income INTEGER,
                    employment_status TEXT, marriage_status TEXT,
                    children_count INTEGER, housing_status TEXT,
                    privacy_consent INTEGER DEFAULT 1,
                    created_at TEXT, updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    user_id TEXT PRIMARY KEY,
                    summary TEXT,
                    updated_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);
                """
            )
            self._conn.commit()

    # ---- users ----
    def get_user(self, user_code: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE user_code = ?", (user_code,)
            ).fetchone()
        return dict(row) if row else None

    def ensure_user(self, user_code: str) -> str:
        existing = self.get_user(user_code)
        if existing:
            return existing["user_id"]
        user_id = str(uuid4())
        with self._lock:
            self._conn.execute(
                "INSERT INTO users(user_id, user_code, privacy_consent, created_at, updated_at) "
                "VALUES(?,?,1,?,?)",
                (user_id, user_code, _now(), _now()),
            )
            self._conn.commit()
        return user_id

    def update_profile(self, user_code: str, fields: dict[str, Any]) -> None:
        user_id = self.ensure_user(user_code)
        updates = {k: v for k, v in fields.items() if k in _PROFILE_FIELDS}
        if not updates:
            return
        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [_now(), user_id]
        with self._lock:
            self._conn.execute(
                f"UPDATE users SET {sets}, updated_at = ? WHERE user_id = ?", values
            )
            self._conn.commit()

    def get_profile(self, user_code: str) -> dict[str, Any]:
        user = self.get_user(user_code)
        if not user:
            return {}
        return {k: user.get(k) for k in _PROFILE_FIELDS if user.get(k) not in (None, "")}

    def clear_profile(self, user_code: str) -> None:
        user = self.get_user(user_code)
        if not user:
            return
        sets = ", ".join(f"{k} = NULL" for k in _PROFILE_FIELDS)
        with self._lock:
            self._conn.execute(
                f"UPDATE users SET {sets}, updated_at = ? WHERE user_id = ?",
                (_now(), user["user_id"]),
            )
            self._conn.commit()

    def delete_user(self, user_code: str) -> None:
        user = self.get_user(user_code)
        if not user:
            return
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE user_id = ?", (user["user_id"],))
            self._conn.execute("DELETE FROM summaries WHERE user_id = ?", (user["user_id"],))
            self._conn.execute("DELETE FROM users WHERE user_id = ?", (user["user_id"],))
            self._conn.commit()

    # ---- messages ----
    def save_message(self, user_code: str, role: str, content: str) -> None:
        user_id = self.ensure_user(user_code)
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(id, user_id, role, content, created_at) VALUES(?,?,?,?,?)",
                (str(uuid4()), user_id, role, content, _now()),
            )
            self._conn.commit()

    def get_recent_messages(self, user_code: str, limit: int = 40) -> list[dict[str, str]]:
        user = self.get_user(user_code)
        if not user:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user["user_id"], limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def trim_messages(self, user_code: str, max_messages: int = 40) -> None:
        user = self.get_user(user_code)
        if not user:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE user_id = ? AND id NOT IN "
                "(SELECT id FROM messages WHERE user_id = ? ORDER BY created_at DESC LIMIT ?)",
                (user["user_id"], user["user_id"], max_messages),
            )
            self._conn.commit()

    def delete_messages(self, user_code: str) -> int:
        user = self.get_user(user_code)
        if not user:
            return 0
        with self._lock:
            cur = self._conn.execute("DELETE FROM messages WHERE user_id = ?", (user["user_id"],))
            self._conn.commit()
            return cur.rowcount

    # ---- summary ----
    def upsert_summary(self, user_code: str, summary: str) -> None:
        user_id = self.ensure_user(user_code)
        with self._lock:
            self._conn.execute(
                "INSERT INTO summaries(user_id, summary, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at",
                (user_id, summary, _now()),
            )
            self._conn.commit()

    def get_summary(self, user_code: str) -> str:
        user = self.get_user(user_code)
        if not user:
            return ""
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM summaries WHERE user_id = ?", (user["user_id"],)
            ).fetchone()
        return (row["summary"] if row and row["summary"] else "")
