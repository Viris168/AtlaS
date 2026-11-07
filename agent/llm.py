"""Thin wrapper around the model client (Milestone 1, streaming from Milestone 11).

Uses the Anthropic SDK pointed at DeepSeek's Anthropic-compatible endpoint,
so responses keep the same shape: content blocks, tool_use, stop_reason, usage.
"""
from typing import Callable

import anthropic
from anthropic.types import Message

from agent.config import Config
from agent.guardrails import with_retries


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
