"""Milestone 7. Offline: the main agent and its subagent each get a scripted FakeLLM."""
from agent.prompts import build_system_prompt
from tests.test_loop import FakeLLM, make_agent, reply, text, tool_use


def names(tools):
    return [t["name"] for t in tools]


def make_parent(tmp_path, parent_llm, child_llm, **overrides):
    agent = make_agent(tmp_path, parent_llm, **overrides)
    agent._subagent_llm = child_llm
    return agent


def test_main_agent_offers_delegate_task(tmp_path):
    llm = FakeLLM(reply("end_turn", text("hi")))
    make_agent(tmp_path, llm).run("hello")
    assert "delegate_task" in names(llm.tools_seen)


def test_depth_zero_turns_subagents_off(tmp_path):
    llm = FakeLLM(reply("end_turn", text("hi")))
    make_agent(tmp_path, llm, max_subagent_depth=0).run("hello")
    assert "delegate_task" not in names(llm.tools_seen)


def test_parent_gets_only_the_childs_final_answer(tmp_path):
    (tmp_path / "notes.txt").write_text("the secret is 42")
    child = FakeLLM(
        reply("tool_use", tool_use("c1", "read_file", {"path": "notes.txt"})),
        reply("end_turn", text("The file says the secret is 42.")),
    )
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "Read notes.txt and report the secret."})),
        reply("end_turn", text("It's 42.")),
    )
    agent = make_parent(tmp_path, parent, child)

    assert agent.run("what's the secret?") == "It's 42."
    result = agent.messages[2]["content"][0]
    assert result["content"] == "The file says the secret is 42." and result["is_error"] is False
    # The child's read_file call never enters the parent's history.
    assert "read_file" not in str(agent.messages)


def test_subagent_cannot_delegate_again(tmp_path):
    child = FakeLLM(reply("end_turn", text("done")))
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "do it"})),
        reply("end_turn", text("ok")),
    )
    make_parent(tmp_path, parent, child).run("go")
    assert "delegate_task" in names(parent.tools_seen)
    assert "delegate_task" not in names(child.tools_seen)


def test_subagent_has_its_own_plan(tmp_path):
    child = FakeLLM(
        reply("tool_use", tool_use("c1", "todo_write", {"todos": [{"id": "1", "task": "child step", "status": "pending"}]})),
        reply("end_turn", text("done")),
    )
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "do it"})),
        reply("end_turn", text("ok")),
    )
    agent = make_parent(tmp_path, parent, child)
    agent.run("go")
    assert agent.todos.items == []


def test_childs_tokens_count_toward_the_parent(tmp_path):
    child = FakeLLM(reply("end_turn", text("done")))  # 2 tokens
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "do it"})),  # 2 tokens
        reply("end_turn", text("ok")),  # 2 tokens
    )
    agent = make_parent(tmp_path, parent, child)
    agent.run("go")
    assert agent.total_tokens == 6


def test_childs_risky_tools_still_need_approval(tmp_path):
    child = FakeLLM(
        reply("tool_use", tool_use("c1", "write_file", {"path": "a.txt", "content": "hey"})),
        reply("end_turn", text("I wasn't allowed to write it.")),
    )
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "write a.txt"})),
        reply("end_turn", text("ok")),
    )
    asked = []
    agent = make_parent(tmp_path, parent, child)
    agent.approver.ask = lambda name, args: asked.append(name) or "n"

    agent.run("go")
    assert asked == ["write_file"]
    assert not (tmp_path / "a.txt").exists()


def test_childs_tool_calls_are_reported_with_a_prefix(tmp_path):
    child = FakeLLM(
        reply("tool_use", tool_use("c1", "list_dir", {"path": "."})),
        reply("end_turn", text("done")),
    )
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "look around"})),
        reply("end_turn", text("ok")),
    )
    calls = []
    agent = make_parent(tmp_path, parent, child)
    agent.on_tool_call = lambda name, args, result: calls.append(name)

    agent.run("go")
    assert calls == ["subagent > list_dir", "delegate_task"]


def test_empty_task_is_an_error(tmp_path):
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "  "})),
        reply("end_turn", text("ok")),
    )
    agent = make_parent(tmp_path, parent, FakeLLM())
    agent.run("go")
    result = agent.messages[2]["content"][0]
    assert result["is_error"] is True and "empty" in result["content"]


def test_prompt_sections(tmp_path):
    config = make_agent(tmp_path, FakeLLM()).config
    assert "delegate_task" not in build_system_prompt(config)
    assert "delegate_task" in build_system_prompt(config, can_delegate=True)
    assert "## You are a subagent" in build_system_prompt(config, is_subagent=True)
