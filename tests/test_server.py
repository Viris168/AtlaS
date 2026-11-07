"""Milestone 11: the web API, offline. Each session's agent gets a scripted FakeLLM."""
import json

import pytest
from fastapi.testclient import TestClient

import server
from agent.config import Config
from agent.loop import Agent
from tests.test_loop import FakeLLM, reply, text, tool_use


def make_config(tmp_path, **overrides) -> Config:
    config = Config(workspace=tmp_path, data_dir=tmp_path / "d", log_dir=tmp_path / "l", knowledge_dir=tmp_path / "k")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_client(tmp_path, *replies, **overrides):
    """A client whose sessions all play back `replies` in order (shared, so one script per test)."""
    config = make_config(tmp_path, **overrides)
    llm = FakeLLM(*replies)
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


def test_health(tmp_path):
    with make_client(tmp_path) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["sessions"] == 0


def test_message_streams_tool_calls_then_the_answer(tmp_path):
    replies = [
        reply("tool_use", tool_use("t1", "read_file", {"path": "a.txt"})),
        reply("end_turn", text("It says hi.")),
    ]
    (tmp_path / "a.txt").write_text("hi")
    with make_client(tmp_path, *replies) as client:
        sid = new_session(client)
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "read a.txt"}) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            events = read_events(response)
        assert [e for e, _ in events] == ["tool_call", "answer"]
        assert events[0][1]["name"] == "read_file" and events[0][1]["result"] == "hi"
        assert events[1][1]["text"] == "It says hi."

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
            assert client.post(f"/sessions/{sid}/approvals/{request['approval_id']}", json={"decision": "y"}).status_code == 200
            events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        assert events[-1] == {"text": "Written."}
        assert (tmp_path / "a.txt").read_text() == "hey"


def test_unanswered_approval_times_out_as_no(tmp_path):
    replies = [
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "hey"})),
        reply("end_turn", text("OK, not written.")),
    ]
    with make_client(tmp_path, *replies, approval_timeout=0.05) as client:
        sid = new_session(client)
        with client.stream("POST", f"/sessions/{sid}/messages", json={"message": "write it"}) as response:
            events = read_events(response)
        assert [e for e, _ in events] == ["approval_request", "tool_call", "answer"]
        assert events[1][1]["result"].startswith("Denied")
        assert not (tmp_path / "a.txt").exists()


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
        assert client.post(f"/sessions/{sid}/approvals/x", json={"decision": "y"}).status_code == 404
        assert client.post(f"/sessions/{sid}/approvals/x", json={"decision": "maybe"}).status_code == 422
        assert client.post(f"/sessions/{sid}/messages", json={"message": ""}).status_code == 422


def test_delete_session_and_session_limit(tmp_path):
    with make_client(tmp_path, max_sessions=1) as client:
        sid = new_session(client)
        assert client.post("/sessions").status_code == 429
        assert client.delete(f"/sessions/{sid}").status_code == 204
        assert client.get(f"/sessions/{sid}").status_code == 404
        new_session(client)


def test_tools_marks_risky_ones(tmp_path):
    with make_client(tmp_path) as client:
        tools = {t["name"]: t["risky"] for t in client.get("/tools").json()}
        assert tools["write_file"] is True and tools["read_file"] is False and "web_search" in tools


@pytest.mark.parametrize("header, status", [(None, 401), ("Bearer wrong", 401), ("Bearer s3cret-token", 200)])
def test_api_token_is_required_when_set(tmp_path, header, status):
    with make_client(tmp_path, api_token="s3cret-token") as client:
        headers = {"Authorization": header} if header else {}
        assert client.get("/health", headers=headers).status_code == status
