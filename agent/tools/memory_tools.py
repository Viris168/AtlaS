"""agent/tools/memory_tools.py — remember, recall, forget (Milestone 5).

Long-term memory that survives between sessions, backed by the JSON store in agent/memory.py.
remember and recall are safe; forget deletes a saved fact for good, so it needs approval.
"""
from agent.memory import Memory
from agent.tools.registry import Tool


def render_entries(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        tags = f" [{', '.join(e['tags'])}]" if e["tags"] else ""
        lines.append(f"- ({e['id']}, {e['created_at'][:10]}) {e['text']}{tags}")
    return "\n".join(lines)


def memory_tools(memory: Memory) -> list[Tool]:
    def remember(text: str, tags: list[str] | None = None) -> str:
        text = str(text).strip()
        if not text:
            raise ValueError("'text' is empty; say what to remember")
        if tags is not None and not isinstance(tags, list):
            raise ValueError("'tags' must be a list of strings")
        clean_tags = [str(t).strip().lower() for t in tags or [] if str(t).strip()]
        entry = memory.add(text, clean_tags)
        return f"Remembered (id {entry['id']}): {entry['text']}"

    def recall(query: str, k: int = 5) -> str:
        k = max(1, min(int(k), 20))
        found = memory.search(query, k)
        if not found:
            return f"No memories match '{query}'."
        return f"{len(found)} matching memories:\n" + render_entries(found)

    def forget(id: str) -> str:
        if memory.delete(str(id).strip()):
            return f"Forgot memory {id}."
        raise ValueError(f"no memory with id '{id}'; use recall to find the right id")

    return [
        Tool(
            name="remember",
            description=(
                "Save a fact to long-term memory so it is still known in future sessions. "
                "Use it for durable things worth keeping: the user's preferences, names, project details, "
                "decisions. Write one self-contained fact per call. Don't save passing details of the current task."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The fact, as a full sentence that makes sense on its own."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional keywords that help find it later, e.g. ['preference', 'python'].",
                    },
                },
                "required": ["text"],
            },
            fn=remember,
        ),
        Tool(
            name="recall",
            description=(
                "Search long-term memory by keywords. Use it when earlier sessions might know something useful, "
                "e.g. the user's preferences or past decisions. Returns matching facts with their ids."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to look for."},
                    "k": {"type": "integer", "description": "Maximum results (1-20, default 5)."},
                },
                "required": ["query"],
            },
            fn=recall,
        ),
        Tool(
            name="forget",
            description=(
                "Delete one memory by id, e.g. when it is wrong or the user asks you to forget it. "
                "Find the id with recall first. This cannot be undone."
            ),
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The memory id shown by recall."}},
                "required": ["id"],
            },
            fn=forget,
            risky=True,
        ),
    ]
