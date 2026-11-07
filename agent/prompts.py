"""System prompt builder (Milestones 1, 4 and 5)."""
from datetime import date

from agent.config import Config
from agent.tools.memory_tools import render_entries

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
    "For tasks with 3 or more steps, write a todo list with todo_write first, keep exactly one item "
    "in progress, and mark items done as you finish them. Send the whole list every time. "
    "Skip the list for simple questions.",
    "You have long-term memory that lasts between sessions. Use remember to save durable facts "
    "(the user's preferences, names, project details, decisions), and recall to look them up when "
    "earlier sessions might help. Use forget when a memory is wrong or the user asks.",
]


def build_system_prompt(config: Config, memories: list[dict] | None = None) -> str:
    sections = [
        ROLE,
        f"Today's date is {date.today().isoformat()}.",
        f"Your workspace is {config.workspace}. All file paths are relative to it.",
        "## Rules\n" + "\n".join(f"- {rule}" for rule in RULES),
    ]
    if memories:
        # Just the latest few, as a hint; recall searches the rest.
        sections.append("## Recent memories\n" + render_entries(memories))
    return "\n\n".join(sections)
