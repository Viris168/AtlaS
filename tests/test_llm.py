"""Milestone 11: LLM.complete streams through on_text and still returns the whole message. No network."""
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


def make_llm(tmp_path, final, chunks=()):
    config = Config(api_key="test-key", workspace=tmp_path, data_dir=tmp_path / "d", log_dir=tmp_path / "l",
                    knowledge_dir=tmp_path / "k")
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
