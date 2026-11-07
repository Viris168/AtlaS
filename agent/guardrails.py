"""agent/guardrails.py — Approvals, safe_path sandbox, shell blocklist, token budget, retries (Milestone 3).

Each guardrail is small and independent, so the loop and the tools can each use the part they need:
  - safe_path:     file tools refuse paths outside the workspace
  - check_command: run_shell refuses obviously destructive commands
  - Approver:      the loop asks the user before running a risky tool
  - TokenBudget:   the loop stops once a run has used too many tokens
  - with_retries:  the LLM wrapper retries calls that failed for temporary reasons
"""
import random
import re
import time
from pathlib import Path
from typing import Callable

import anthropic


# --- Path sandbox ----------------------------------------------------------------------------

def safe_path(workspace: Path, path: str) -> Path:
    """Turn the model's path into a real path, but only if it stays inside the workspace."""
    full = (workspace / path).resolve()  # resolve() follows "..", and symlinks too
    if not full.is_relative_to(workspace.resolve()):
        raise PermissionError(f"'{path}' is outside the workspace")
    return full


# --- Shell blocklist -------------------------------------------------------------------------

# A last line of defence against obvious disasters, not a sandbox: a determined model can
# always find a spelling that slips past a regex. The approval prompt is the real check.
BLOCKED_COMMANDS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*(/|~|\$HOME)/?\*?(\s|$)","deleting the root or home folder"),
    (r"\bsudo\b|\bsu\s", "running as another user"),
    (r"\bmkfs(\.\w+)?\b", "formatting a disk"),
    (r"\bdd\b.*\bof=/dev/", "writing straight to a device"),
    (r">\s*/dev/(sd|nvme|hd|disk)", "writing straight to a device"),
    (r":\(\)\s*\{.*:\s*\|\s*:.*\}", "a fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "shutting down the machine"),
    (r"\bchmod\s+(-[a-zA-Z]*\s+)*777\s+/(\s|$)", "opening up the whole filesystem"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z)?sh\b", "piping a download into a shell"),
]


def check_command(command: str) -> None:
    """Raise PermissionError if the command matches the blocklist."""
    for pattern, reason in BLOCKED_COMMANDS:
        if re.search(pattern, command):
            raise PermissionError(f"Blocked command ({reason}): {command}")


# --- Approvals -------------------------------------------------------------------------------

AskFn = Callable[[str, dict], str]  # ask(tool_name, args) -> "y", "n" or "a" (always for this tool)


class Approver:
    """Decides whether a risky tool call may run.

    With auto_approve everything runs. Otherwise `ask` is called; without an `ask`
    (e.g. no one at the keyboard) risky calls are denied, so the default is safe.
    """

    def __init__(self, auto_approve: bool = False, ask: AskFn | None = None):
        self.auto_approve = auto_approve
        self.ask = ask
        self.always: set[str] = set()  # tools the user said "always" to in this session

    def approve(self, name: str, args: dict) -> bool:
        if self.auto_approve or name in self.always:
            return True
        if self.ask is None:
            return False
        answer = self.ask(name, args).strip().lower()
        if answer in ("a", "always"):
            self.always.add(name)
            return True
        return answer in ("y", "yes")


# --- Token budget ----------------------------------------------------------------------------

class TokenBudget:
    """Counts tokens across every model call in one run and says when the limit is reached."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def add(self, usage) -> None:
        self.used += usage.input_tokens + usage.output_tokens

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit


# --- Retries ---------------------------------------------------------------------------------

RETRY_STATUS = {408, 409, 429}  # timeout, conflict, rate limit; plus every 5xx (529 = overloaded)


def is_retryable(error: Exception) -> bool:
    """Temporary failures are worth retrying; bad requests or a wrong API key never will be."""
    if isinstance(error, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code in RETRY_STATUS or error.status_code >= 500
    return False


def with_retries(fn: Callable, max_retries: int = 4, base_delay: float = 1.0, max_delay: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep):
    """Call fn(); on a retryable error wait and try again, up to max_retries extra attempts.

    The wait doubles each time (1s, 2s, 4s, ...) with random jitter, so many clients
    hitting a rate limit at once don't all retry at the same moment.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries or not is_retryable(e):
                raise
            delay = min(max_delay, base_delay * 2 ** attempt)
            sleep(random.uniform(delay / 2, delay))
