"""A JSON-file memory store with add, search, recent and delete (wired in at M5).

Search is simple keyword overlap. Upgrade idea (stretch goal): swap it for
embeddings using the vector store from Milestone 5B, keeping this interface.
"""
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Memory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, text: str, tags: list[str] | None = None) -> dict:
        entry = {
            "id": uuid.uuid4().hex[:8],
            "text": text,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.entries.append(entry)
        self._save()
        return entry

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = _words(query)
        if not q:
            return []
        scored = []
        for e in self.entries:
            score = len(q & (_words(e["text"]) | _words(" ".join(e["tags"]))))
            if score:
                scored.append((score, e))
        scored.sort(key=lambda pair: (pair[0], pair[1]["created_at"]), reverse=True)
        return [e for _, e in scored[:k]]

    def recent(self, n: int = 3) -> list[dict]:
        return self.entries[-n:][::-1]

    def delete(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        if len(self.entries) != before:
            self._save()
            return True
        return False
