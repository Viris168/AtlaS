"""Milestones 11 and 12: the web API, offline. Each session's agent gets a scripted FakeLLM."""
import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from agent.config import Config
from agent.loop import Agent
from agent.sessions import SessionStore
from tests.test_loop import FakeLLM, reply, text, tool_use


def make_config(tmp_path, **overrides) -> Config:
    config = Config(workspace=tmp_path, data_dir=tmp_path / "d", log_dir=tmp_path / "l", knowledge_dir=tmp_path / "k")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_client(tmp_path, *replies, llm=None, **overrides):
    """A client whose sessions all play back `replies` in order (shared, so one script per test).
    Pass `llm` to share one script across two clients, as a server before and after a restart."""
    config = make_config(tmp_path, **overrides)
    llm = llm or FakeLLM(*replies)
    app = server.create_app(config, make_agent=lambda c, tracer, mcp: Agent(c, llm=llm, tracer=tracer), start_mcp=False)
    return TestClient(app)


def read_events(response):
    """Parse an SSE body into [(event, data), ...]."""
    events, event = [], None
    for line in response.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            events.append((event, json.loads(line[6:])))
    return events


def new_session(client) -> str:
    response = client.post("/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def workspace(client, sid):
    return Path(client.get(f"/sessions/{sid}").json()["workspace"])


def test_health(tmp_path):
    with make_client(tmp_path) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["sessions"] == 0


def test_message_streams_text_tool_calls_then_the_answer(tmp_path):
    replies = [
        reply("tool_use", text("Let me look."), tool_use("t1", "read_file", {"path": "a.txt"})),
        reply("end_turn", text("It says hi.")),
    ]
    with make_client(tmp_path, *replies) as client:
        sid = new_session(client)
        (workspace(client, sid) / "a.txt").write_text("hi")
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "read a.txt"}) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            events = read_events(response)
        assert [e for e, _ in events] == ["text", "text", "tool_call", "text", "text", "answer"]
        assert "".join(d["text"] for e, d in events[:2]) == "Let me look."
        assert events[2][1]["name"] == "read_file" and events[2][1]["result"] == "hi"
        assert events[-1][1]["text"] == "It says hi."

        info = client.get(f"/sessions/{sid}").json()
        assert info["messages"] == 4 and info["busy"] is False
        assert info["usage"]["input_tokens"] == 2 and info["cost_usd"] > 0


def test_risky_tool_waits_for_an_approval_over_http(tmp_path):
    replies = [
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "hey"})),
        reply("end_turn", text("Written.")),
    ]
    with make_client(tmp_path, *replies) as client:
        sid = new_session(client)
        events = []
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "write it"}) as response:
            lines = response.iter_lines()
            for line in lines:
                if line.startswith("data: ") and "approval_id" in line:
                    request = json.loads(line[6:])
                    break
            assert request["name"] == "write_file" and request["args"]["path"] == "a.txt"
            pending = client.get(f"/sessions/{sid}").json()["pending_approvals"]
            assert pending[0]["approval_id"] == request["approval_id"]
            assert client.post(f"/sessions/{sid}/approvals/{request['approval_id']}", json={"allow": True}).status_code == 200
            events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        assert events[-1] == {"text": "Written."}
        assert (workspace(client, sid) / "a.txt").read_text() == "hey"
        assert not (tmp_path / "a.txt").exists()  # in the session's own folder, not the shared workspace


def test_unanswered_approval_times_out_as_no(tmp_path):
    replies = [
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "hey"})),
        reply("end_turn", text("OK, not written.")),
    ]
    with make_client(tmp_path, *replies, approval_timeout=0.05) as client:
        sid = new_session(client)
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "write it"}) as response:
            events = read_events(response)
        assert [e for e, _ in events if e != "text"] == ["approval_required", "tool_call", "answer"]
        assert events[1][1]["result"].startswith("Denied")
        assert not (workspace(client, sid) / "a.txt").exists()


def test_failed_turn_is_rolled_back_and_reported(tmp_path):
    with make_client(tmp_path) as client:  # no replies scripted: the FakeLLM raises IndexError
        sid = new_session(client)
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "hi"}) as response:
            events = read_events(response)
        assert events == [("error", {"error": "IndexError: pop from empty list"})]
        assert client.get(f"/sessions/{sid}").json()["messages"] == 0


def test_reset_clears_the_history(tmp_path):
    with make_client(tmp_path, reply("end_turn", text("hi"))) as client:
        sid = new_session(client)
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "hello"}) as response:
            read_events(response)
        assert client.post(f"/sessions/{sid}/reset").status_code == 200
        info = client.get(f"/sessions/{sid}").json()
        assert info["messages"] == 0 and info["usage"]["output_tokens"] == 1  # usage outlives a reset


def test_unknown_session_approval_and_bad_input(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/sessions/nope").status_code == 404
        sid = new_session(client)
        assert client.post(f"/sessions/{sid}/approvals/x", json={"allow": True}).status_code == 404
        assert client.post(f"/sessions/{sid}/approvals/x", json={"decision": "y"}).status_code == 422
        assert client.post(f"/sessions/{sid}/messages", json={"message": ""}).status_code == 422


def test_delete_session(tmp_path):
    with make_client(tmp_path) as client:
        sid = new_session(client)
        assert client.delete(f"/sessions/{sid}").status_code == 204
        assert client.get(f"/sessions/{sid}").status_code == 404
        assert client.delete(f"/sessions/{sid}").status_code == 404
        assert client.get("/sessions").json() == []


def test_past_the_limit_the_idle_session_is_unloaded_not_lost(tmp_path):
    with make_client(tmp_path, max_sessions=1) as client:
        one = new_session(client)
        two = new_session(client)  # unloads `one` to make room
        assert set(client.app.state.sessions) == {two}
        assert client.get(f"/sessions/{one}").status_code == 200  # loaded back, unloading `two`
        assert set(client.app.state.sessions) == {one}
        health = client.get("/health").json()
        assert health["sessions"] == 1 and health["saved_sessions"] == 2


def test_429_only_when_every_loaded_session_is_busy(tmp_path):
    with make_client(tmp_path, max_sessions=1) as client:
        sid = new_session(client)
        busy = client.app.state.sessions[sid].busy
        busy.acquire()  # as if a turn were running
        try:
            assert client.post("/sessions").status_code == 429
        finally:
            busy.release()
        assert client.post("/sessions").status_code == 201


def test_tools_marks_risky_ones(tmp_path):
    with make_client(tmp_path) as client:
        tools = {t["name"]: t["risky"] for t in client.get("/tools").json()}
        assert tools["write_file"] is True and tools["read_file"] is False and "web_search" in tools


@pytest.mark.parametrize("header, status", [(None, 401), ("Bearer wrong", 401), ("Bearer s3cret-token", 200)])
def test_api_token_is_required_when_set(tmp_path, header, status):
    with make_client(tmp_path, api_token="s3cret-token") as client:
        headers = {"Authorization": header} if header else {}
        assert client.get("/health", headers=headers).status_code == status


def test_always_allows_the_tool_for_the_rest_of_the_session(tmp_path):
    replies = [
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "1"})),
        reply("tool_use", tool_use("t2", "write_file", {"path": "b.txt", "content": "2"})),
        reply("end_turn", text("Both written.")),
    ]
    with make_client(tmp_path, *replies) as client:
        sid = new_session(client)
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "write both"}) as response:
            seen = []
            for line in response.iter_lines():
                if line.startswith("event: "):
                    seen.append(line[7:])
                elif line.startswith("data: ") and seen[-1] == "approval_required":
                    approval_id = json.loads(line[6:])["approval_id"]
                    client.post(f"/sessions/{sid}/approvals/{approval_id}", json={"allow": True, "always": True})
        assert seen.count("approval_required") == 1  # the second write didn't ask again
        assert (workspace(client, sid) / "b.txt").read_text() == "2"


def test_each_session_gets_its_own_workspace(tmp_path):
    with make_client(tmp_path) as client:
        one, two = workspace(client, new_session(client)), workspace(client, new_session(client))
        assert one != two and one.is_dir() and two.is_dir()
        assert one.parent == tmp_path / "sessions"


def test_trace_endpoint_returns_the_sessions_events(tmp_path):
    with make_client(tmp_path, reply("end_turn", text("hi"))) as client:
        sid = new_session(client)
        assert client.get(f"/sessions/{sid}/trace").json() == []
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "hello"}) as response:
            read_events(response)
        types = [e["type"] for e in client.get(f"/sessions/{sid}/trace").json()]
        assert types == ["user_input", "llm_call", "final_answer"]
        assert client.get("/sessions/nope/trace").status_code == 404


# --- Milestone 12: sessions saved across restarts, and the web page -----------------------------

class RecordingLLM(FakeLLM):
    """A FakeLLM that also keeps a copy of the history each call was sent."""

    def __init__(self, *replies):
        super().__init__(*replies)
        self.sent = []

    def complete(self, system, messages, tools=None, on_text=None):
        self.sent.append(list(messages))
        return super().complete(system, messages, tools, on_text)


def chat(client, sid, message, approve=None):
    """Send a message, answering any approval request with `approve` ({"allow": ...}), and return the events."""
    events, event = [], None
    with client.stream("POST", f"/sessions/{sid}/messages", json={"message": message}) as response:
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                events.append((event, data))
                if event == "approval_required":
                    client.post(f"/sessions/{sid}/approvals/{data['approval_id']}", json=approve)
    return events


def test_a_session_carries_on_after_a_restart(tmp_path):
    llm = RecordingLLM(
        reply("tool_use", tool_use("t1", "todo_write", {"todos": [{"id": "1", "task": "write a.txt", "status": "in_progress"}]}),
              tool_use("t2", "write_file", {"path": "a.txt", "content": "1"})),
        reply("end_turn", text("Written.")),
        reply("tool_use", tool_use("t3", "write_file", {"path": "b.txt", "content": "2"})),
        reply("end_turn", text("You asked me to write a.txt.")),
    )
    with make_client(tmp_path, llm=llm) as client:
        sid = new_session(client)
        chat(client, sid, "write a.txt", approve={"allow": True, "always": True})
        before = client.get(f"/sessions/{sid}").json()

    with make_client(tmp_path, llm=llm) as client:  # a new server process, as far as the sessions know
        assert client.app.state.sessions == {}
        after = client.get(f"/sessions/{sid}").json()
        for key in ("messages", "todos", "usage", "cost_usd", "workspace"):
            assert after[key] == before[key], key
        events = chat(client, sid, "what did I ask first?")
        # "always" came back too: the second write ran without asking.
        assert [e for e, _ in events if e != "text"] == ["tool_call", "answer"]
        assert events[-1][1]["text"] == "You asked me to write a.txt."
        assert (workspace(client, sid) / "b.txt").read_text() == "2"
        # The model saw the whole conversation, the first turn included.
        assert llm.sent[2][0] == {"role": "user", "content": "write a.txt"} and len(llm.sent[2]) == 5
        # One trace file for the session, across both servers.
        types = [e["type"] for e in client.get(f"/sessions/{sid}/trace").json()]
        assert types.count("user_input") == 2


def test_sessions_are_listed_newest_first_with_a_title(tmp_path):
    with make_client(tmp_path, reply("end_turn", text("hi")), reply("end_turn", text("hey"))) as client:
        one, two = new_session(client), new_session(client)
        chat(client, two, "second   session,\nfirst message")
        chat(client, one, "first session")
        listed = client.get("/sessions").json()
        assert [s["session_id"] for s in listed] == [one, two]
        assert [s["title"] for s in listed] == ["first session", "second session, first message"]
        assert listed[0]["loaded"] is True and listed[0]["busy"] is False


def test_messages_endpoint_gives_the_history_for_display(tmp_path):
    replies = [
        reply("tool_use", text("Looking."), tool_use("t1", "read_file", {"path": "a.txt"})),
        reply("end_turn", text("It says hi.")),
    ]
    with make_client(tmp_path, *replies) as client:
        sid = new_session(client)
        (workspace(client, sid) / "a.txt").write_text("hi")
        chat(client, sid, "read a.txt")
        assert client.get(f"/sessions/{sid}/messages").json() == [
            {"role": "user", "text": "read a.txt"},
            {"role": "assistant", "text": "Looking."},
            {"role": "tool", "name": "read_file", "args": {"path": "a.txt"}, "result": "hi"},
            {"role": "assistant", "text": "It says hi."},
        ]


def test_reset_is_saved(tmp_path):
    llm = FakeLLM(reply("end_turn", text("hi")))
    with make_client(tmp_path, llm=llm) as client:
        sid = new_session(client)
        chat(client, sid, "hello")
        client.post(f"/sessions/{sid}/reset")
    with make_client(tmp_path, llm=llm) as client:
        assert client.get(f"/sessions/{sid}").json()["messages"] == 0


def test_expired_sessions_are_deleted(tmp_path):
    with make_client(tmp_path) as client:
        old, new = new_session(client), new_session(client)
    store = SessionStore(make_config(tmp_path).sessions_db)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time() - 3 * 3600, old))
    with make_client(tmp_path, session_ttl_hours=2) as client:  # purged on startup
        assert [s["session_id"] for s in client.get("/sessions").json()] == [new]
        assert client.get(f"/sessions/{old}").status_code == 404


def test_page_is_served_without_the_token(tmp_path):
    with make_client(tmp_path, api_token="s3cret-token") as client:
        response = client.get("/")
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
        assert "<title>" in response.text
        assert client.get("/sessions").status_code == 401
