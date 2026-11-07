from types import SimpleNamespace as NS

from agent.config import Config
from agent.loop import Agent


def reply(stop_reason, *blocks):
    return NS(stop_reason=stop_reason, content=list(blocks), usage=NS(input_tokens=1, output_tokens=1))


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, args):
    return NS(type="tool_use", id=id, name=name, input=args)


class FakeLLM:
    """Plays back scripted replies and records what tools it was offered."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.tools_seen = None

    def complete(self, system, messages, tools=None):
        self.tools_seen = tools
        return self.replies.pop(0)


def make_agent(tmp_path, llm, **overrides):
    config = Config(workspace=tmp_path, data_dir=tmp_path / "d", log_dir=tmp_path / "l", knowledge_dir=tmp_path / "k")
    for key, value in overrides.items():
        setattr(config, key, value)
    return Agent(config, llm=llm)


def test_offers_all_m2_tools(tmp_path):
    llm = FakeLLM(reply("end_turn", text("hi")))
    make_agent(tmp_path, llm).run("hello")
    assert [t["name"] for t in llm.tools_seen] == ["read_file", "write_file", "list_dir", "run_shell", "web_search"]


def test_web_search_can_be_turned_off(tmp_path):
    llm = FakeLLM(reply("end_turn", text("hi")))
    make_agent(tmp_path, llm, enable_web_search=False).run("hello")
    assert "web_search" not in [t["name"] for t in llm.tools_seen]


def test_runs_tools_through_registry(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "hey"})),
        reply("end_turn", text("Done.")),
    )
    calls = []
    agent = make_agent(tmp_path, llm, auto_approve=True)
    agent.on_tool_call = lambda name, args, result: calls.append(name)

    assert agent.run("make a file") == "Done."
    assert (tmp_path / "a.txt").read_text() == "hey"
    assert calls == ["write_file"]
    result = agent.messages[2]["content"][0]
    assert result["tool_use_id"] == "t1" and result["is_error"] is False


def test_pause_turn_continues(tmp_path):
    llm = FakeLLM(reply("pause_turn", text("searching...")), reply("end_turn", text("Found it.")))
    assert make_agent(tmp_path, llm).run("search") == "Found it."


def test_risky_tool_asks_and_can_be_denied(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "hey"})),
        reply("end_turn", text("OK, I won't.")),
    )
    asked = []
    agent = make_agent(tmp_path, llm)
    agent.approver.ask = lambda name, args: asked.append(name) or "n"

    assert agent.run("make a file") == "OK, I won't."
    assert asked == ["write_file"]
    assert not (tmp_path / "a.txt").exists()
    result = agent.messages[2]["content"][0]
    assert result["is_error"] is True and "Denied" in result["content"]


def test_risky_tool_denied_without_anyone_to_ask(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "run_shell", {"command": "touch x"})),
        reply("end_turn", text("done")),
    )
    make_agent(tmp_path, llm).run("go")
    assert not (tmp_path / "x").exists()


def test_safe_tool_runs_without_asking(tmp_path):
    llm = FakeLLM(reply("tool_use", tool_use("t1", "list_dir", {})), reply("end_turn", text("empty")))
    agent = make_agent(tmp_path, llm)
    agent.approver.ask = lambda name, args: (_ for _ in ()).throw(AssertionError("should not ask"))
    assert agent.run("list") == "empty"


def test_token_budget_stops_the_run(tmp_path):
    # Each fake reply costs 2 tokens, so a budget of 3 allows two model calls.
    llm = FakeLLM(*[reply("tool_use", tool_use(f"t{i}", "list_dir", {})) for i in range(5)])
    agent = make_agent(tmp_path, llm, max_total_tokens=3)
    answer = agent.run("loop forever")
    assert answer.startswith("Stopped: token budget used up")
    assert len(llm.replies) == 3  # only two of the five replies were used
    assert agent.messages[-1] == {"role": "assistant", "content": answer}
