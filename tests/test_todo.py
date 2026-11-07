import pytest

from agent.tools.registry import ToolRegistry
from agent.tools.todo import TodoList, todo_tools
from tests.test_loop import FakeLLM, make_agent, reply, text, tool_use


def plan(*statuses):
    return [{"id": str(i), "task": f"step {i}", "status": s} for i, s in enumerate(statuses, start=1)]


def test_write_replaces_whole_list_and_renders():
    todos = TodoList()
    todos.write(plan("pending", "pending", "pending"))
    todos.write(plan("done", "in_progress"))
    assert todos.read() == "[x] 1. step 1\n[>] 2. step 2"


def test_empty_list_reads_clearly():
    assert TodoList().read() == "(no todos)"


@pytest.mark.parametrize(
    "bad, message",
    [
        (plan("in_progress", "in_progress"), "only one item may be in_progress"),
        (plan("started"), "status 'started'"),
        ([{"id": "1", "task": "  ", "status": "pending"}], "empty task"),
        ([{"id": "1", "task": "a", "status": "pending"}, {"id": "1", "task": "b", "status": "pending"}], "duplicate id"),
        ("not a list", "must be a list"),
    ],
)
def test_bad_lists_are_rejected_and_old_list_kept(bad, message):
    todos = TodoList()
    todos.write(plan("pending"))
    with pytest.raises(ValueError, match=message):
        todos.write(bad)
    assert todos.read() == "[ ] 1. step 1"


def test_errors_reach_the_model_through_the_registry():
    registry = ToolRegistry()
    for tool in todo_tools(TodoList()):
        registry.register(tool)
    content, is_error = registry.run("todo_write", {"todos": plan("in_progress", "in_progress")})
    assert is_error and "only one item" in content
    assert not registry.is_risky("todo_write") and not registry.is_risky("todo_read")


def test_on_change_fires_after_each_write():
    seen = []
    todos = TodoList(on_change=lambda t: seen.append(t.read()))
    todos.write(plan("in_progress", "pending"))
    todos.write(plan("done", "in_progress"))
    assert seen == ["[>] 1. step 1\n[ ] 2. step 2", "[x] 1. step 1\n[>] 2. step 2"]


def test_agent_plans_then_ticks_off_without_asking(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "todo_write", {"todos": plan("in_progress", "pending")})),
        reply("tool_use", tool_use("t2", "todo_write", {"todos": plan("done", "in_progress")})),
        reply("end_turn", text("All done.")),
    )
    agent = make_agent(tmp_path, llm)
    agent.approver.ask = lambda name, args: (_ for _ in ()).throw(AssertionError("should not ask"))

    assert agent.run("do two things") == "All done."
    assert agent.todos.read() == "[x] 1. step 1\n[>] 2. step 2"


def test_each_agent_has_its_own_list(tmp_path):
    a = make_agent(tmp_path, FakeLLM())
    b = make_agent(tmp_path, FakeLLM())
    a.todos.write(plan("pending"))
    assert b.todos.read() == "(no todos)"
