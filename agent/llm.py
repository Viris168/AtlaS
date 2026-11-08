"""Thin wrapper around the model client (Milestone 1, streaming from 11, prompt caching from 12).

Uses the Anthropic SDK pointed at DeepSeek's Anthropic-compatible endpoint,
so responses keep the same shape: content blocks, tool_use, stop_reason, usage.
"""
from typing import Callable

import anthropic
from anthropic.types import Message

from agent.config import Config
from agent.guardrails import with_retries
from agent.sessions import plain_block

CACHE = {"type": "ephemeral"}


class LLM:
    def __init__(self, config: Config, model: str | None = None):
        if not config.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to your .env file.")
        self.config = config
        self.model = model or config.model
        # max_retries=0: with_retries (M3) does the retrying, so the SDK's own retries would stack on top.
        self.client = anthropic.Anthropic(api_key=config.api_key, base_url=config.base_url, max_retries=0)

    def complete(self, system: str, messages: list, tools: list | None = None,
                 on_text: Callable[[str], None] | None = None) -> Message:
        """One model call. With on_text the response is streamed: on_text(chunk) gets the text as it
        arrives, and the return value is the same complete Message as without it."""
        if self.config.prompt_cache:
            system, messages = with_cache_breakpoints(system, messages)
        kwargs = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if on_text is None:
            return with_retries(lambda: self.client.messages.create(**kwargs), max_retries=self.config.max_retries)

        def stream() -> Message:
            with self.client.messages.stream(**kwargs) as response:
                for chunk in response.text_stream:
                    on_text(chunk)
                return response.get_final_message()

        # A retry after part of the text was shown shows it again; rare, and better than failing the turn.
        return with_retries(stream, max_retries=self.config.max_retries)


def with_cache_breakpoints(system: str, messages: list) -> tuple[list, list]:
    """Mark the end of the system prompt and the end of the history as cacheable (M12).

    The API caches the prompt up to each marker: tools and system prompt (unchanged for a turn's
    steps), and everything said so far (unchanged when the next step resends it with more on the end).
    Only the last message is copied and changed; the agent's own history is never touched.
    Providers without prompt caching ignore the markers.
    """
    marked_system = [{"type": "text", "text": system, "cache_control": CACHE}]
    if not messages:
        return marked_system, messages
    last = messages[-1]
    if isinstance(last["content"], str):
        content = [{"type": "text", "text": last["content"]}]
    else:
        content = [plain_block(b) for b in last["content"]]
    if not content:
        return marked_system, messages
    content[-1] = {**content[-1], "cache_control": CACHE}
    return marked_system, messages[:-1] + [{**last, "content": content}]
