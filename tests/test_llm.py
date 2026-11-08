"""Milestones 11 and 12: LLM.complete streams through on_text and marks cache breakpoints. No network."""
from types import SimpleNamespace as NS

from agent.config import Config
from agent.llm import LLM


class FakeStream:
    def __init__(self, chunks, final):
        self.text_stream, self.final = iter(chunks), final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.final


def make_llm(tmp_path, final, chunks=(), prompt_cache=False):
    config = Config(api_key="test-key", workspace=tmp_path, data_dir=tmp_path / "d", log_dir=tmp_path / "l",
                    knowledge_dir=tmp_path / "k", prompt_cache=prompt_cache)
    llm = LLM(config)
    calls = []
    llm.client = NS(messages=NS(
        create=lambda **kw: calls.append(("create", kw)) or final,
        stream=lambda **kw: calls.append(("stream", kw)) or FakeStream(chunks, final),
    ))
    return llm, calls


def test_without_on_text_it_makes_a_plain_call(tmp_path):
    final = NS(content=[], stop_reason="end_turn")
    llm, calls = make_llm(tmp_path, final)
    assert llm.complete("sys", [{"role": "user", "content": "hi"}]) is final
    assert [c[0] for c in calls] == ["create"]


def test_with_on_text_it_streams_each_chunk_and_returns_the_final_message(tmp_path):
    final = NS(content=[], stop_reason="end_turn")
    llm, calls = make_llm(tmp_path, final, chunks=["Hel", "lo"])
    seen = []
    assert llm.complete("sys", [{"role": "user", "content": "hi"}], [{"name": "t"}], on_text=seen.append) is final
    assert seen == ["Hel", "lo"]
    assert calls[0][0] == "stream" and calls[0][1]["tools"] == [{"name": "t"}]


CACHE = {"type": "ephemeral"}


def test_prompt_cache_marks_the_system_prompt_and_the_last_block(tmp_path):
    final = NS(content=[], stop_reason="end_turn")
    llm, calls = make_llm(tmp_path, final, prompt_cache=True)
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [NS(type="tool_use", id="t1", name="n", input={})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    snapshot = repr(history)
    llm.complete("sys", history)
    sent = calls[0][1]
    assert sent["system"] == [{"type": "text", "text": "sys", "cache_control": CACHE}]
    assert sent["messages"][-1]["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "ok", "cache_control": CACHE}]
    assert sent["messages"][:2] == history[:2]
    assert repr(history) == snapshot  # the agent's own history is never changed


def test_prompt_cache_turns_a_plain_string_message_into_a_marked_block(tmp_path):
    llm, calls = make_llm(tmp_path, NS(content=[], stop_reason="end_turn"), prompt_cache=True)
    llm.complete("sys", [{"role": "user", "content": "hi"}])
    assert calls[0][1]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": CACHE}]}]


def test_without_prompt_cache_nothing_is_marked(tmp_path):
    llm, calls = make_llm(tmp_path, NS(content=[], stop_reason="end_turn"))
    llm.complete("sys", [{"role": "user", "content": "hi"}])
    assert calls[0][1]["system"] == "sys" and calls[0][1]["messages"] == [{"role": "user", "content": "hi"}]
