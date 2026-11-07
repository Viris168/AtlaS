"""agent/logging_utils.py — Tracer: one JSON line per event in log_dir/{session_id}.jsonl (Milestone 9).

The trace is the agent's flight recorder: every model call, tool call, context trim and subagent,
in order, with timings and token counts. scripts/show_trace.py prints it as a tree.

Every event has: ts (ISO time), type, run_id (one per Agent.run, so one per user turn or subagent task).
Event types and their extra fields:
  user_input      text
  subagent_start  parent_id (the run that delegated), task, model
  llm_call        model, purpose ("agent" or "summary"), input_tokens, cache_read_tokens, output_tokens,
                  latency, stop_reason, text, tool_calls
  tool_call       name, args, approval, duration, error, output (a preview)
  context_trim    size_before, size_after, cleared, summarized
  final_answer    text, stop ("answer", "step_limit" or "token_budget"), steps, duration,
                  input_tokens, cache_read_tokens, output_tokens
  subagent_end    parent_id, plus the same fields as final_answer
  error           error (the exception, logged just before it propagates out of run)
"""
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

PREVIEW_CHARS = 300  # tool output and model text kept in the trace; the full text lives in the history

# Things that look like credentials. A trace gets shared and pasted, so it must never hold one.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI, DeepSeek and Anthropic style keys
    re.compile(r"\b(ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),  # Google API key
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
]
# "api_key=abc123..." keeps the "api_key=" part, so the trace still says what was there.
KEY_VALUE_PATTERN = re.compile(r"(?i)\b((?:api[_-]?key|secret|password|passwd|token)\s*[=:]\s*)['\"]?[^\s'\"&]{6,}")
SECRET_KEYS = re.compile(r"(?i)(api[_-]?key|secret|password|passwd|token|authorization)")
MASK = "***"


def mask_secrets(value, extra: tuple[str, ...] = ()):
    """A copy of value (str, dict, list, ...) with anything credential-like replaced by ***.
    `extra` holds known secrets, such as the configured API key, to mask wherever they appear."""
    if isinstance(value, str):
        for secret in extra:
            if secret:
                value = value.replace(secret, MASK)
        for pattern in SECRET_PATTERNS:
            value = pattern.sub(MASK, value)
        return KEY_VALUE_PATTERN.sub(lambda m: m.group(1) + MASK, value)
    if isinstance(value, dict):
        return {k: MASK if isinstance(k, str) and SECRET_KEYS.search(k) and isinstance(v, str) and v
                else mask_secrets(v, extra) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_secrets(v, extra) for v in value]
    return value


def preview(text, limit: int = PREVIEW_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def new_session_id() -> str:
    """Sorts by time, so the newest trace is the last file in the folder."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + new_id()[:4]


class Tracer:
    """Appends events to log_dir/{session_id}.jsonl. Each line is written and flushed at once,
    because a crash is exactly when the log is needed. Thread-safe, for parallel subagents and MCP."""

    def __init__(self, log_dir: Path, session_id: str | None = None, secrets: tuple[str, ...] = ()):
        self.session_id = session_id or new_session_id()
        self.path = Path(log_dir) / f"{self.session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.secrets = tuple(s for s in secrets if s)
        self._lock = threading.Lock()
        self._file = None

    def event(self, type: str, run_id: str | None = None, **fields) -> dict:
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"), "type": type, "run_id": run_id}
        record.update(mask_secrets(fields, self.secrets))
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            if self._file is None:  # opened on the first event, so a session that does nothing leaves no file
                self._file = self.path.open("a", encoding="utf-8")
            self._file.write(line + "\n")
            self._file.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class NullTracer:
    """Records nothing. The default, so tests and scripts don't fill logs/ unless they ask to."""

    session_id = None
    path = None

    def event(self, type: str, run_id: str | None = None, **fields) -> dict:
        return {}

    def close(self) -> None:
        pass


# --- Token usage and cost (for /cost) ---------------------------------------------------------

class Usage:
    """Tokens per model. Kept per model because the main agent and the subagents/summarizer use different ones."""

    def __init__(self):
        self.by_model: dict[str, dict[str, int]] = {}

    def _row(self, model: str) -> dict[str, int]:
        return self.by_model.setdefault(model, {"calls": 0, "input": 0, "cache_read": 0, "output": 0})

    def add(self, model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> None:
        row = self._row(model)
        row["calls"] += 1
        row["input"] += input_tokens
        row["cache_read"] += cache_read_tokens
        row["output"] += output_tokens

    def merge(self, other: "Usage") -> None:
        for model, row in other.by_model.items():
            mine = self._row(model)
            for key, value in row.items():
                mine[key] += value

    @property
    def input_tokens(self) -> int:
        """Uncached input only. The provider reports cache hits apart, in cache_read_tokens."""
        return sum(row["input"] for row in self.by_model.values())

    @property
    def cache_read_tokens(self) -> int:
        return sum(row["cache_read"] for row in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(row["output"] for row in self.by_model.values())


def cost_usd(input_tokens: int, output_tokens: int, cache_read_tokens: int, config) -> float:
    """Prices come from config (USD per million tokens), since they change."""
    return (input_tokens * config.price_input_per_mtok
            + cache_read_tokens * config.price_cache_read_per_mtok
            + output_tokens * config.price_output_per_mtok) / 1_000_000
