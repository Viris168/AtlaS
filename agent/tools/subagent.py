"""agent/tools/subagent.py — delegate_task (Milestone 7).

A subagent is a second, fresh Agent that works on one task and reports back. The main agent
only sees its final answer, not the files it read or the tools it ran, so a big side job
(reading many files, a long search) doesn't fill up the main agent's context.

This file only defines the tool the model sees. Building and running the child Agent is
done by Agent.delegate in agent/loop.py, which passes it in here as `run_subagent`.
"""
from typing import Callable

from agent.tools.registry import Tool


def delegate_tool(run_subagent: Callable[[str], str]) -> Tool:
    """delegate_task, bound to the agent that will spawn the child. Not risky by itself:
    the child's own risky tools (write_file, run_shell, forget) still ask for approval."""

    def delegate_task(task: str) -> str:
        task = str(task).strip()
        if not task:
            raise ValueError("'task' is empty; describe the job and everything the subagent needs to know")
        return run_subagent(task) or "(the subagent finished without a written answer)"

    return Tool(
        name="delegate_task",
        description=(
            "Hand a self-contained subtask to a subagent: a fresh assistant with the same tools but none of "
            "this conversation. It works on its own and returns only its final answer. "
            "Use it for side jobs whose steps you don't need to see, such as reading many files, researching "
            "a question, or summarizing a document. Don't use it for quick one-step tasks. "
            "The subagent can't ask you questions, so put every detail it needs in 'task': the goal, file "
            "paths, constraints, and what its answer should contain."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The full instructions for the subagent, written so they make sense on their own.",
                }
            },
            "required": ["task"],
        },
        fn=delegate_task,
    )
