"""Milestone 8. Offline: a tiny MCP server (tests/fixtures/mcp_test_server.py) runs as a real stdio subprocess."""
import json
import sys
from pathlib import Path

import pytest

from agent.config import Config
from agent.loop import Agent
from agent.tools.mcp_bridge import MCPBridge, ServerConfig, load_server_configs, tool_name
from tests.test_loop import FakeLLM, reply, text, tool_use

TEST_SERVER = str(Path(__file__).parent / "fixtures" / "mcp_test_server.py")


def fixture_server(name="t", **kw):
    return ServerConfig(name, command=sys.executable, args=[TEST_SERVER], **kw)


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    b = MCPBridge(
        [fixture_server(env={"ATLAS_TEST_VAR": "hello"}), fixture_server("trusty", trusted=True)],
        tmp_path_factory.mktemp("logs"),
    ).start()
    yield b
    b.close()


def make_mcp_agent(tmp_path, llm, bridge, **kw):
    config = Config(workspace=tmp_path, data_dir=tmp_path / "d", log_dir=tmp_path / "l", knowledge_dir=tmp_path / "k")
    return Agent(config, llm=llm, mcp=bridge, **kw)


def tools_by_name(bridge):
    return {t.name: t for t in bridge.tools()}


# --- Config ------------------------------------------------------------------------------------

def test_missing_config_means_no_servers(tmp_path):
    assert load_server_configs(tmp_path / "nope.json") == []


def test_config_reads_servers_and_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "abc")
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": [
        {"name": "a", "command": "uvx", "args": ["thing"], "env": {"TOKEN": "${SECRET_TOKEN}"}},
        {"name": "b", "url": "http://localhost:8000/mcp", "trusted": True},
    ]}))
    a, b = load_server_configs(path)
    assert (a.command, a.args, a.env, a.trusted) == ("uvx", ["thing"], {"TOKEN": "abc"}, False)
    assert (b.url, b.trusted) == ("http://localhost:8000/mcp", True)


@pytest.mark.parametrize("servers", [
    [{"command": "x"}],                                   # no name
    [{"name": "a", "command": "x"}, {"name": "a", "command": "y"}],  # duplicate
    [{"name": "a"}],                                      # neither command nor url
    [{"name": "a", "command": "x", "url": "http://h"}],   # both
])
def test_bad_config_is_an_error(tmp_path, servers):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": servers}))
    with pytest.raises(ValueError):
        load_server_configs(path)


def test_tool_names_are_prefixed_and_api_safe():
    assert tool_name("git hub", "search.issues") == "mcp__git_hub__search_issues"
    assert len(tool_name("s" * 50, "t" * 50)) == 64


# --- Talking to a real server -------------------------------------------------------------------

def test_lists_every_servers_tools(bridge):
    names = set(tools_by_name(bridge))
    assert {"mcp__t__add", "mcp__t__get_env", "mcp__t__explode", "mcp__trusty__add"} <= names


def test_tool_schema_comes_from_the_server(bridge):
    add = tools_by_name(bridge)["mcp__t__add"]
    assert add.input_schema["required"] == ["a", "b"]
    assert "Add two numbers" in add.description and "MCP server 't'" in add.description


def test_calls_a_tool(bridge):
    assert tools_by_name(bridge)["mcp__t__add"].fn(a=2, b=3) == "5"


def test_env_reaches_the_server(bridge):
    assert tools_by_name(bridge)["mcp__t__get_env"].fn(name="ATLAS_TEST_VAR") == "hello"


def test_tool_error_becomes_an_error_result(bridge):
    from agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(tools_by_name(bridge)["mcp__t__explode"])
    content, is_error = registry.run("mcp__t__explode", {})
    assert is_error and "MCPToolError" in content


def test_only_trusted_servers_skip_approval(bridge):
    tools = tools_by_name(bridge)
    assert tools["mcp__t__add"].risky is True
    assert tools["mcp__trusty__add"].risky is False


def test_broken_server_is_skipped_not_fatal(tmp_path):
    b = MCPBridge([ServerConfig("ghost", command="no-such-program-atlas"), fixture_server()], tmp_path).start()
    try:
        status = dict(b.status())
        assert status["ghost"].startswith("failed") and status["t"].startswith("connected")
        assert all(t.name.startswith("mcp__t__") for t in b.tools())
    finally:
        b.close()


def test_server_that_never_answers_times_out(tmp_path):
    mute = ServerConfig("mute", command=sys.executable, args=["-c", "import time; time.sleep(60)"])
    b = MCPBridge([mute], tmp_path, startup_timeout=1).start()
    try:
        assert "no answer within 1s" in dict(b.status())["mute"]
        assert b.tools() == []
    finally:
        b.close()


def test_calls_after_close_fail_cleanly(tmp_path):
    b = MCPBridge([fixture_server()], tmp_path).start()
    add = tools_by_name(b)["mcp__t__add"]
    b.close()
    with pytest.raises(RuntimeError, match="not connected"):
        add.fn(a=1, b=1)


def test_no_servers_starts_nothing(tmp_path):
    b = MCPBridge([], tmp_path).start()
    assert b._thread is None and b.tools() == [] and b.status() == []
    b.close()


# --- Inside the agent ---------------------------------------------------------------------------

def test_agent_offers_and_runs_mcp_tools(tmp_path, bridge):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "mcp__trusty__add", {"a": 20, "b": 22})),
        reply("end_turn", text("42")),
    )
    agent = make_mcp_agent(tmp_path, llm, bridge)

    assert agent.run("add them") == "42"
    assert "mcp__t__add" in [t["name"] for t in llm.tools_seen]
    result = agent.messages[2]["content"][0]
    assert result == {"type": "tool_result", "tool_use_id": "t1", "content": "42", "is_error": False}


def test_untrusted_mcp_tool_asks_for_approval(tmp_path, bridge):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "mcp__t__add", {"a": 1, "b": 1})),
        reply("end_turn", text("ok")),
    )
    asked = []
    agent = make_mcp_agent(tmp_path, llm, bridge, ask_approval=lambda name, args: asked.append(name) or "n")

    agent.run("add")
    assert asked == ["mcp__t__add"]
    assert agent.messages[2]["content"][0]["content"].startswith("Denied")


def test_subagent_shares_the_bridge(tmp_path, bridge):
    child = FakeLLM(reply("end_turn", text("done")))
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "do it"})),
        reply("end_turn", text("ok")),
    )
    agent = make_mcp_agent(tmp_path, parent, bridge, subagent_llm=child)

    agent.run("go")
    assert "mcp__t__add" in [t["name"] for t in child.tools_seen]
