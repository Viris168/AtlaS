from agent.memory import Memory
from agent.prompts import build_system_prompt
from agent.tools.memory_tools import memory_tools
from agent.tools.registry import ToolRegistry
from tests.test_loop import FakeLLM, make_agent, reply, text, tool_use


def registry_for(memory):
    registry = ToolRegistry()
    for tool in memory_tools(memory):
        registry.register(tool)
    return registry


def test_remember_then_recall(tmp_path):
    registry = registry_for(Memory(tmp_path / "m.json"))
    content, is_error = registry.run("remember", {"text": "User prefers tabs over spaces", "tags": ["Preference"]})
    assert not is_error and "Remembered" in content

    content, is_error = registry.run("recall", {"query": "tabs"})
    assert not is_error
    assert "User prefers tabs over spaces [preference]" in content


def test_recall_with_no_match_says_so(tmp_path):
    content, is_error = registry_for(Memory(tmp_path / "m.json")).run("recall", {"query": "anything"})
    assert not is_error and "No memories match" in content


def test_bad_input_reaches_the_model_as_an_error(tmp_path):
    registry = registry_for(Memory(tmp_path / "m.json"))
    content, is_error = registry.run("remember", {"text": "   "})
    assert is_error and "empty" in content
    content, is_error = registry.run("forget", {"id": "nope"})
    assert is_error and "no memory with id 'nope'" in content


def test_forget_deletes_and_is_risky(tmp_path):
    memory = Memory(tmp_path / "m.json")
    entry = memory.add("old fact")
    registry = registry_for(memory)
    assert registry.is_risky("forget")
    assert not registry.is_risky("remember") and not registry.is_risky("recall")

    content, is_error = registry.run("forget", {"id": entry["id"]})
    assert not is_error and memory.entries == []


def test_memories_survive_a_new_session(tmp_path):
    registry_for(Memory(tmp_path / "m.json")).run("remember", {"text": "Project codename is Atlas"})
    content, _ = registry_for(Memory(tmp_path / "m.json")).run("recall", {"query": "codename"})
    assert "Project codename is Atlas" in content


def test_recent_memories_go_in_the_system_prompt(tmp_path):
    agent = make_agent(tmp_path, FakeLLM())
    assert "Recent memories" not in build_system_prompt(agent.config, agent.memory.recent())
    agent.memory.add("User's name is Viris")
    assert "User's name is Viris" in build_system_prompt(agent.config, agent.memory.recent())


def test_agent_remembers_across_agents(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "remember", {"text": "User likes short answers"})),
        reply("end_turn", text("Noted.")),
    )
    assert make_agent(tmp_path, llm).run("I like short answers") == "Noted."

    later = make_agent(tmp_path, FakeLLM())
    assert [e["text"] for e in later.memory.search("short answers")] == ["User likes short answers"]


def test_agent_asks_before_forgetting(tmp_path):
    agent = make_agent(tmp_path, FakeLLM())
    entry = agent.memory.add("keep me")
    agent.llm = FakeLLM(
        reply("tool_use", tool_use("t1", "forget", {"id": entry["id"]})),
        reply("end_turn", text("Kept it.")),
    )
    agent.approver.ask = lambda name, args: "n"

    assert agent.run("forget that") == "Kept it."
    assert [e["text"] for e in agent.memory.entries] == ["keep me"]
