"""agent/tools/registry.py — Tool dataclass + ToolRegistry (register, schemas, run) (Milestone 2).

A Tool pairs the schema the model sees with the Python function that does the work.
The registry is the single place the loop asks "which tools exist?" and "run this one".
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., str]  # called as fn(**args), returns text for the model
    risky: bool = False  # needs user approval before running (used by the guardrails in M3)

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolRegistry:
    def __init__(self, max_output_chars: int = 10_000):
        self.max_output_chars = max_output_chars
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def is_risky(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.risky

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a tool and return (content, is_error). Never raises: errors go back to the model."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self._tools) or 'none'}", True
        try:
            content, is_error = str(tool.fn(**(args or {}))), False
        except TypeError as e:
            # Usually the model sent missing or extra arguments.
            content, is_error = f"Error: bad arguments for '{name}': {e}", True
        except Exception as e:
            content, is_error = f"Error: {type(e).__name__}: {e}", True
        return self._truncate(content), is_error

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        dropped = len(text) - self.max_output_chars
        return text[: self.max_output_chars] + f"\n... [truncated {dropped} chars]"

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
