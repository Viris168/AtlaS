"""System prompt builder (Milestone 1)."""
from datetime import date

from agent.config import Config

ROLE = (
    "You are Atlas, a command-line AI agent. You help the user by answering "
    "questions and completing tasks, using the tools you have been given."
)

RULES = [
    "Use a tool when it would give a better or more accurate answer than you can from memory. "
    "Don't call tools for things you already know.",
    "If you're unsure or a tool result is unclear, say so plainly instead of guessing.",
    "If a tool returns an error, read it and try a different approach.",
    "When the task is done, give a clear final answer and stop calling tools.",
]


def build_system_prompt(config: Config) -> str:
    sections = [
        ROLE,
        f"Today's date is {date.today().isoformat()}.",
        f"Your workspace is {config.workspace}. All file paths are relative to it.",
        "## Rules\n" + "\n".join(f"- {rule}" for rule in RULES),
    ]
    return "\n\n".join(sections)
