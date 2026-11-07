"""Every setting Atlas needs. Each milestone names the fields it uses.

Values can be overridden with environment variables (or a .env file),
e.g. AGENT_MAX_STEPS=1 python cli.py
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default, cast=str):
    value = os.getenv(name)
    if value is None:
        return default
    if cast is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    return cast(value)


@dataclass
class Config:
    # Provider: DeepSeek's Anthropic-compatible endpoint (M1)
    api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: _env("AGENT_BASE_URL", "https://api.deepseek.com/anthropic"))

    # Models (M1, M6, M7)
    model: str = field(default_factory=lambda: _env("AGENT_MODEL", "deepseek-v4-pro"))
    subagent_model: str = field(default_factory=lambda: _env("AGENT_SUBAGENT_MODEL", "deepseek-flash"))
    max_tokens: int = field(default_factory=lambda: _env("AGENT_MAX_TOKENS", 4096, int))

    # Loop limits and budgets (M1, M3)
    max_steps: int = field(default_factory=lambda: _env("AGENT_MAX_STEPS", 25, int))
    max_total_tokens: int = field(default_factory=lambda: _env("AGENT_MAX_TOTAL_TOKENS", 500_000, int))

    # Guardrails (M3)
    auto_approve: bool = field(default_factory=lambda: _env("AGENT_AUTO_APPROVE", False, bool))
    max_retries: int = field(default_factory=lambda: _env("AGENT_MAX_RETRIES", 4, int))

    # Tools (M2)
    shell_timeout: int = field(default_factory=lambda: _env("AGENT_SHELL_TIMEOUT", 30, int))
    max_tool_output_chars: int = field(default_factory=lambda: _env("AGENT_MAX_TOOL_OUTPUT", 10_000, int))
    enable_web_search: bool = field(default_factory=lambda: _env("AGENT_WEB_SEARCH", True, bool))

    # Context management (M6)
    context_char_budget: int = field(default_factory=lambda: _env("AGENT_CONTEXT_CHARS", 200_000, int))
    keep_recent_messages: int = field(default_factory=lambda: _env("AGENT_KEEP_RECENT", 6, int))

    # Long-term memory (M5)
    memory_recent_in_prompt: int = field(default_factory=lambda: _env("AGENT_MEMORY_RECENT", 3, int))

    # Subagents (M7)
    max_subagent_depth: int = field(default_factory=lambda: _env("AGENT_MAX_SUBAGENT_DEPTH", 1, int))

    # Paths
    workspace: Path = field(default_factory=lambda: Path(_env("AGENT_WORKSPACE", ROOT / "workspace")).resolve())
    data_dir: Path = field(default_factory=lambda: Path(_env("AGENT_DATA_DIR", ROOT / "data")).resolve())
    log_dir: Path = field(default_factory=lambda: Path(_env("AGENT_LOG_DIR", ROOT / "logs")).resolve())
    knowledge_dir: Path = field(default_factory=lambda: Path(_env("AGENT_KNOWLEDGE_DIR", ROOT / "knowledge")).resolve())

    # Pricing in USD per million tokens (M9). Placeholders: set DeepSeek's current prices.
    price_input_per_mtok: float = field(default_factory=lambda: _env("AGENT_PRICE_IN", 3.0, float))
    price_output_per_mtok: float = field(default_factory=lambda: _env("AGENT_PRICE_OUT", 15.0, float))

    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory.json"

    def __post_init__(self):
        for path in (self.workspace, self.data_dir, self.log_dir, self.knowledge_dir):
            path.mkdir(parents=True, exist_ok=True)


config = Config()
