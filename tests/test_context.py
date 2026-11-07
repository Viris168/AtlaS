from types import SimpleNamespace as NS

from agent.context import (
    CLEARED,
    ContextManager,
    clear_old_tool_output,
    estimate_size,
    render_transcript,
    summarize_old_half,
)


def tool_turn(n, output="x" * 1000):
    """One assistant tool call and the user message holding its result."""
    return [
        {"role": "assistant", "content": [NS(type="tool_use", id=f"t{n}", name="read_file", input={"path": f"f{n}.txt"})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": output, "is_error": False}]},
    ]


def long_run(turns=10):
    messages = [{"role": "user", "content": "read every file and summarize each"}]
    for n in range(turns):
        messages += tool_turn(n)
    return messages


def assert_valid(messages):
    """What the API checks: user first, roles alternate, every tool_result follows its own tool_use."""
    assert messages[0]["role"] == "user"
    for prev, cur in zip(messages, messages[1:]):
        assert prev["role"] != cur["role"]
    for i, m in enumerate(messages):
        if m["role"] == "user" and not isinstance(m["content"], str):
            ids = {b.id for b in messages[i - 1]["content"] if getattr(b, "type", None) == "tool_use"}
            assert i > 0 and all(b["tool_use_id"] in ids for b in m["content"])


def test_estimate_size_counts_strings_and_blocks():
    assert estimate_size([{"role": "user", "content": "hello"}]) == 5
    assert estimate_size(long_run(2)) > 2000


def test_clear_keeps_recent_results_and_every_block():
    messages = long_run(5)
    cleared = clear_old_tool_output(messages, keep_recent=4)
    assert cleared == 3  # results of turns 0-2; turns 3-4 are the last 4 messages
    results = [m["content"][0]["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    assert results[:3] == [CLEARED] * 3 and results[3:] == ["x" * 1000] * 2
    assert_valid(messages)
    assert clear_old_tool_output(messages, keep_recent=4) == 0  # already cleared ones don't count twice


def test_clear_copies_instead_of_editing_shared_dicts():
    messages = long_run(3)
    original = messages[2]["content"][0]
    clear_old_tool_output(messages, keep_recent=0)
    assert original["content"] == "x" * 1000


def test_summary_replaces_old_half_and_keeps_the_request():
    messages = long_run(10)
    seen = []
    assert summarize_old_half(messages, lambda t: seen.append(t) or "SUMMARY")
    assert_valid(messages)
    assert len(messages) < 21
    assert "SUMMARY" in messages[0]["content"]
    assert "read every file and summarize each" in messages[0]["content"]  # original goal survives
    assert "read_file" in seen[0]  # the summarizer saw the tool calls


def test_summary_does_not_copy_request_when_it_is_kept():
    messages = long_run(4) + [{"role": "assistant", "content": [NS(type="text", text="done")]},
                              {"role": "user", "content": "now the next thing"}] + tool_turn(99)
    summarize_old_half(messages, lambda t: "S")
    assert_valid(messages)
    assert "current request" not in messages[0]["content"]


def test_manager_runs_summary_only_when_clearing_is_not_enough():
    calls = []
    manager = ContextManager(budget=3000, keep_recent=2, summarize=lambda t: calls.append(t) or "S")
    messages = long_run(10)
    manager.manage(messages)
    assert (manager.clears, manager.summaries) == (1, 0)  # placeholders alone got it under budget
    assert not calls

    tight = ContextManager(budget=500, keep_recent=2, summarize=lambda t: "S")
    messages = long_run(10)
    tight.manage(messages)
    assert (tight.clears, tight.summaries) == (1, 1)
    assert_valid(messages)


def test_manager_does_nothing_under_budget():
    manager = ContextManager(budget=10**6, keep_recent=2, summarize=lambda t: "S")
    messages = long_run(3)
    before = [dict(m) for m in messages]
    manager.manage(messages)
    assert messages == before and (manager.clears, manager.summaries) == (0, 0)


def test_transcript_truncates_long_results():
    text = render_transcript(tool_turn(1, output="y" * 5000))
    assert "[truncated]" in text and len(text) < 3000


def test_second_summary_carries_the_original_request():
    messages = long_run(10)
    summarize_old_half(messages, lambda t: "FIRST")
    for n in range(10, 20):
        messages += tool_turn(n)
    summarize_old_half(messages, lambda t: "SECOND")
    assert_valid(messages)
    text = messages[0]["content"]
    assert text.count("read every file and summarize each") == 1  # once, not wrapped in the old summary
    assert "FIRST" not in text and "SECOND" in text


def test_agent_trims_during_a_long_run(tmp_path):
    from tests.test_loop import FakeLLM, make_agent, reply, text, tool_use

    (tmp_path / "big.txt").write_text("z" * 3000)
    replies = [reply("tool_use", tool_use(f"t{n}", "read_file", {"path": "big.txt"})) for n in range(12)]
    llm = FakeLLM(*replies, reply("end_turn", text("All summarized.")))
    seen = []
    llm.complete = lambda system, messages, tools=None, _orig=llm.complete: seen.append(list(messages)) or _orig(system, messages, tools)

    agent = make_agent(tmp_path, llm, context_char_budget=8000, keep_recent_messages=4, max_steps=20)
    agent._summarizer = lambda transcript: "read big.txt several times"

    assert agent.run("read every file and summarize each") == "All summarized."
    assert agent.context.clears > 0 and agent.context.summaries > 0
    for history in seen:  # every request the model got was one the API accepts
        assert_valid(history)
    assert "read every file and summarize each" in agent.messages[0]["content"]
