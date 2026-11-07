"""agent/tools/files.py — read_file, write_file, list_dir inside the workspace (Milestone 2).

Every path the model gives is relative to the workspace folder.
Paths that try to escape it (like "../secret.txt" or "/etc/passwd") are refused.
"""
from pathlib import Path

from agent.guardrails import safe_path
from agent.tools.registry import Tool


def read_file(workspace: Path, path: str) -> str:
    file = safe_path(workspace, path)
    if not file.is_file():
        raise FileNotFoundError(f"No file at '{path}'")
    return file.read_text(encoding="utf-8", errors="replace")


def write_file(workspace: Path, path: str, content: str) -> str:
    file = safe_path(workspace, path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to '{path}'"


def list_dir(workspace: Path, path: str = ".") -> str:
    folder = safe_path(workspace, path)
    if not folder.is_dir():
        raise NotADirectoryError(f"No folder at '{path}'")
    entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = [entry.name + ("/" if entry.is_dir() else "") for entry in entries]
    return "\n".join(lines) or "(empty folder)"


def file_tools(workspace: Path) -> list[Tool]:
    """The three file tools, all locked to one workspace folder."""
    return [
        Tool(
            name="read_file",
            description="Read a text file from the workspace. The path is relative to the workspace.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "e.g. 'notes/todo.md'"}},
                "required": ["path"],
            },
            fn=lambda path: read_file(workspace, path),
        ),
        Tool(
            name="write_file",
            description=(
                "Create or overwrite a text file in the workspace. Missing folders are created. "
                "This replaces the whole file, so read it first if you only want to change part of it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "e.g. 'notes/todo.md'"},
                    "content": {"type": "string", "description": "The full new contents of the file."},
                },
                "required": ["path", "content"],
            },
            fn=lambda path, content: write_file(workspace, path, content),
            risky=True,
        ),
        Tool(
            name="list_dir",
            description="List the files and folders in a workspace folder. Folders end with '/'.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Folder to list. Defaults to the workspace root."}},
                "required": [],
            },
            fn=lambda path=".": list_dir(workspace, path),
        ),
    ]
