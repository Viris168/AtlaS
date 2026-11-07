"""agent/tools/shell.py — run_shell with timeout and output limit (Milestone 2).

Runs one shell command inside the workspace folder and returns what it printed.
The output limit itself is applied by the ToolRegistry, like for every other tool.
"""
import os
import signal
import subprocess
from pathlib import Path

from agent.guardrails import check_command
from agent.tools.registry import Tool


def run_shell(workspace: Path, command: str, timeout: int) -> str:
    check_command(command)  # raises PermissionError, which goes back to the model as an error
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=workspace,
        stdin=subprocess.DEVNULL,  # a command waiting for input fails instead of hanging
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # one stream, in the order the command printed it
        text=True,
        errors="replace",
        start_new_session=True,  # own process group, so a timeout can kill its children too
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        raise TimeoutError(f"Command timed out after {timeout}s. Output so far:\n{output}")

    if process.returncode != 0:
        # Not an exception: a failing command is normal, and its output is the useful part.
        return f"{output}\n[exit code {process.returncode}]".strip()
    return output.strip() or "(no output)"


def shell_tool(workspace: Path, timeout: int = 30) -> Tool:
    return Tool(
        name="run_shell",
        description=(
            f"Run a shell command (bash) in the workspace folder and return its output. "
            f"Commands are killed after {timeout} seconds and cannot read input, so avoid "
            f"interactive programs and long-running servers. Each call starts fresh: "
            f"'cd' and variables do not carry over to the next call."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "e.g. 'ls -la' or 'python main.py'"}},
            "required": ["command"],
        },
        fn=lambda command: run_shell(workspace, command, timeout),
        risky=True,
    )
