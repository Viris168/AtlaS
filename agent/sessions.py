"""agent/sessions.py — SessionStore: the web API's sessions, kept in SQLite (Milestone 12).

One row per session: its Agent.state() as JSON, plus a title and timestamps for listing.
Written after every turn, so a restarted server picks each conversation up where its last
finished turn left it. A turn that was still running when the server stopped is lost.

A new connection per call: sqlite3 connections can't be shared across threads, and the server
saves from each session's worker thread.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    state      TEXT NOT NULL
)
"""


def plain_block(block) -> dict:
    """A content block as plain JSON-ready data. SDK models keep exactly the fields the API sent,
    so the block goes back to the API unchanged. Tests use SimpleNamespace blocks."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "to_dict"):
        return block.to_dict(mode="json")
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    return dict(vars(block))


def plain_messages(messages: list) -> list:
    return [m if isinstance(m["content"], str) else {**m, "content": [plain_block(b) for b in m["content"]]}
            for m in messages]


def title_from(messages: list, limit: int = 60) -> str | None:
    """The first thing the user asked, to name the session in a list."""
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], str):
            text = " ".join(m["content"].split())
            return text if len(text) <= limit else text[:limit - 1] + "…"
    return None


def display_messages(messages: list, result_chars: int = 2000) -> list[dict]:
    """The history as a person reads it: {"role": "user" | "assistant", "text"} and
    {"role": "tool", "name", "args", "result"}, for the web page to redraw a reloaded session."""
    items, tools = [], {}
    for m in plain_messages(messages):
        if isinstance(m["content"], str):
            items.append({"role": m["role"], "text": m["content"]})
            continue
        for block in m["content"]:
            kind = block.get("type")
            if kind == "text" and block.get("text", "").strip():
                items.append({"role": m["role"], "text": block["text"]})
            elif kind in ("tool_use", "server_tool_use"):
                item = {"role": "tool", "name": block["name"], "args": block.get("input", {}), "result": None}
                tools[block["id"]] = item
                items.append(item)
            elif kind == "tool_result" and block.get("tool_use_id") in tools:
                content = block.get("content")
                if isinstance(content, list):  # MCP results can be a list of blocks
                    content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
                content = str(content)
                tools[block["tool_use_id"]]["result"] = (content if len(content) <= result_chars
                                                         else content[:result_chars] + "…")
    return items


class SessionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        """Commits on success, and always closes: sqlite3's own `with` commits but leaves it open."""
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, session_id: str, state: dict, title: str | None = None) -> None:
        """Insert or replace the state. The title is set once, from the first message, and then kept."""
        now = time.time()
        with self._connect() as db:
            db.execute(
                """INSERT INTO sessions (id, title, created_at, updated_at, state) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET title = COALESCE(sessions.title, excluded.title),
                                                 updated_at = excluded.updated_at, state = excluded.state""",
                (session_id, title, now, now, json.dumps(state, default=str, ensure_ascii=False)),
            )

    def load(self, session_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT state FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return json.loads(row["state"]) if row else None

    def delete(self, session_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0

    def all(self) -> list[dict]:
        """Newest activity first."""
        with self._connect() as db:
            rows = db.execute("SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC").fetchall()
        return [{"session_id": r["id"], "title": r["title"], "created_at": r["created_at"], "updated_at": r["updated_at"]}
                for r in rows]

    def count(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def older_than(self, cutoff: float) -> list[str]:
        """Sessions with no activity since `cutoff` (a time.time() value)."""
        with self._connect() as db:
            return [r["id"] for r in db.execute("SELECT id FROM sessions WHERE updated_at < ?", (cutoff,))]
