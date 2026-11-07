"""agent/tools/todo.py — todo_write, todo_read (Milestone 4).

The agent's plan for the current task. todo_write replaces the whole list each time,
so there are no partial updates to get wrong. Each Agent owns its own TodoList,
so subagents (Milestone 7) get a separate plan instead of sharing a module global.
"""
from typing import Callable

from agent.tools.registry import Tool

STATUSES = ("pending", "in_progress", "done")
MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}


class TodoList:
    def __init__(self, on_change: Callable[["TodoList"], None] | None = None):
        self.items: list[dict] = []  # each {"id": str, "task": str, "status": str}
        self.on_change = on_change  # called after every successful write, e.g. to print the list

    def write(self, todos: list) -> str:
        self.items = self._validate(todos)
        if self.on_change:
            self.on_change(self)
        return "Todo list updated:\n" + self.render()

    def read(self) -> str:
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "(no todos)"
        return "\n".join(f"{MARKS[item['status']]} {item['id']}. {item['task']}" for item in self.items)

    @staticmethod
    def _validate(todos: list) -> list[dict]:
        """Check the whole new list before replacing anything. Raises ValueError with a message the model can act on."""
        if not isinstance(todos, list):
            raise ValueError("'todos' must be a list of {id, task, status} objects")
        items, seen_ids = [], set()
        for n, todo in enumerate(todos, start=1):
            if not isinstance(todo, dict):
                raise ValueError(f"item {n} must be an object with id, task and status")
            item_id = str(todo.get("id") or n)
            task = str(todo.get("task", "")).strip()
            status = todo.get("status", "pending")
            if not task:
                raise ValueError(f"item {item_id} has an empty task")
            if status not in STATUSES:
                raise ValueError(f"item {item_id} has status '{status}'; use one of {', '.join(STATUSES)}")
            if item_id in seen_ids:
                raise ValueError(f"duplicate id '{item_id}'")
            seen_ids.add(item_id)
            items.append({"id": item_id, "task": task, "status": status})

        in_progress = [item["id"] for item in items if item["status"] == "in_progress"]
        if len(in_progress) > 1:
            raise ValueError(f"only one item may be in_progress at a time, got {len(in_progress)}: {', '.join(in_progress)}")
        return items


def todo_tools(todos: TodoList) -> list[Tool]:
    """todo_write and todo_read, bound to one agent's list. Neither is risky: they only touch memory."""
    return [
        Tool(
            name="todo_write",
            description=(
                "Write your plan for the current task as a todo list. This REPLACES the whole list, "
                "so always send every item, including ones already done. "
                "Use it for tasks with 3 or more steps: write the plan before starting, keep exactly one item "
                "'in_progress', and send the updated list each time you finish an item. "
                "Skip it for simple questions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete list, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Short stable id, e.g. '1', '2'."},
                                "task": {"type": "string", "description": "What to do, as a short imperative."},
                                "status": {"type": "string", "enum": list(STATUSES)},
                            },
                            "required": ["id", "task", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
            fn=todos.write,
        ),
        Tool(
            name="todo_read",
            description="Show the current todo list with each item's status: [ ] pending, [>] in progress, [x] done.",
            input_schema={"type": "object", "properties": {}, "required": []},
            fn=todos.read,
        ),
    ]
