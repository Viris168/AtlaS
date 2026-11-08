"""Milestone 12: SessionStore and Agent.state()/restore(). Offline."""
import json
import time

from anthropic.types import Message

from agent.sessions import SessionStore, display_messages, plain_messages, title_from
from tests.test_loop import FakeLLM, make_agent, reply, text, tool_use


def sdk_message(*content) -> Message:
    """A reply as the real SDK builds it, so serialization is tested on its models, not stand-ins."""
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m", "stop_reason": "tool_use",
        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}, "content": list(content),
    })


def test_store_saves_loads_lists_and_deletes(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.save("a", {"messages": []})
    store.save("b", {"messages": [1]}, title="Second")
    store.save("a", {"messages": [2]}, title="First")
    store.save("a", {"messages": [3]}, title="Changed")  # the first title sticks
    assert store.load("a") == {"messages": [3]} and store.load("nope") is None
    assert [(r["session_id"], r["title"]) for r in store.all()] == [("a", "First"), ("b", "Second")]
    assert store.count() == 2
    assert set(store.older_than(time.time() + 1)) == {"a", "b"}
    assert store.older_than(0) == []
    assert store.delete("a") is True and store.delete("a") is False and store.count() == 1


def test_sdk_blocks_round_trip_as_the_api_sent_them(tmp_path):
    message = sdk_message({"type": "text", "text": "Let me look.", "citations": None},
                          {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.txt"}})
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": message.content}]
    plain = json.loads(json.dumps(plain_messages(history)))
    assert plain[1]["content"] == [{"citations": None, "text": "Let me look.", "type": "text"},
                                   {"id": "t1", "input": {"path": "a.txt"}, "name": "read_file", "type": "tool_use"}]
    assert history[1]["content"] is message.content  # the history itself is left as it was


def test_agent_state_restores_into_a_new_agent(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "todo_write", {"todos": [{"id": "1", "task": "x", "status": "pending"}]})),
        reply("end_turn", text("Planned.")),
        reply("end_turn", text("Still here.")),
    )
    first = make_agent(tmp_path, llm)
    first.approver.always.add("write_file")
    first.run("plan it")
    state = json.loads(json.dumps(first.state()))  # through JSON, as the store does

    second = make_agent(tmp_path, llm)
    second.restore(state)
    assert second.todos.items == first.todos.items
    assert second.approver.always == {"write_file"}
    assert second.usage.by_model == first.usage.by_model and second.total_tokens == first.total_tokens
    assert len(second.messages) == len(first.messages)
    assert second.run("and now?") == "Still here."  # the restored history (plain dicts) works for the next turn
    assert second.usage.output_tokens == 3


def test_title_and_display_helpers():
    long = "word " * 30
    assert title_from([{"role": "user", "content": long}]).endswith("…")
    assert len(title_from([{"role": "user", "content": long}])) == 60
    assert title_from([]) is None
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "run_shell", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "a.txt"}]}]},
        {"role": "assistant", "content": "Stopped at step limit."},
    ]
    assert display_messages(history) == [
        {"role": "user", "text": "go"},
        {"role": "tool", "name": "run_shell", "args": {"command": "ls"}, "result": "a.txt"},
        {"role": "assistant", "text": "Stopped at step limit."},
    ]
