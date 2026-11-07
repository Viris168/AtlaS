import io
from types import SimpleNamespace as NS

from rich.console import Console

import cli
from agent.loop import Agent
from tests.test_loop import FakeLLM, make_agent, reply, text, tool_use


def test_memory_command_lists_every_entry(tmp_path, capsys):
    agent = make_agent(tmp_path, FakeLLM())
    cli.show_memory(agent)
    assert "(no memories saved)" in capsys.readouterr().out

    agent.memory.add("User's name is Virak", ["name"])
    agent.memory.add("User prefers short answers")
    cli.COMMANDS["/memory"](agent)
    out = capsys.readouterr().out
    assert "2 memories" in out
    assert "User's name is Virak [name]" in out and "User prefers short answers" in out


def test_ingest_command_prints_the_report(tmp_path, capsys, monkeypatch):
    from agent.rag import ingest
    from agent.rag.ingest import IngestReport

    monkeypatch.setattr(ingest, "run_ingest", lambda config: IngestReport(added=["a.md"], chunks_written=3))
    cli.COMMANDS["/ingest"](make_agent(tmp_path, FakeLLM()))
    out = capsys.readouterr().out
    assert "added: a.md" in out and "chunks written: 3" in out


def test_context_command_shows_size_and_counts(tmp_path, capsys):
    agent = make_agent(tmp_path, FakeLLM())
    agent.messages.append({"role": "user", "content": "hello"})
    agent.context.clears = 2
    cli.COMMANDS["/context"](agent)
    out = capsys.readouterr().out
    assert "5 of 200,000 characters" in out and "1 messages" in out
    assert "cleared: 2 times" in out and "summarized: 0 times" in out


def test_tools_command_lists_tools_and_marks_risky_ones(tmp_path, capsys):
    cli.COMMANDS["/tools"](make_agent(tmp_path, FakeLLM()))
    out = capsys.readouterr().out
    assert "13 tools" in out
    assert "* write_file:" in out and "  read_file:" in out and "web_search:" in out


def test_mcp_command_without_servers(tmp_path, capsys):
    cli.COMMANDS["/mcp"](make_agent(tmp_path, FakeLLM()))
    assert "no MCP servers configured" in capsys.readouterr().out


def test_help_lists_every_command(tmp_path, capsys):
    cli.COMMANDS["/help"](make_agent(tmp_path, FakeLLM()))
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out
    assert "Start a new conversation" in out and "/exit" in out


def test_reset_clears_history_plan_and_always_but_keeps_memory(tmp_path, capsys):
    agent = make_agent(tmp_path, FakeLLM())
    agent.messages.append({"role": "user", "content": "hello"})
    agent.todos.write([{"id": "1", "task": "x", "status": "pending"}])
    agent.approver.always.add("write_file")
    agent.memory.add("User's name is Virak")
    cli.COMMANDS["/reset"](agent)
    assert agent.messages == [] and agent.todos.items == [] and agent.approver.always == set()
    assert len(agent.memory.entries) == 1
    assert "Conversation cleared" in capsys.readouterr().out


def test_trace_command_without_a_trace(tmp_path, capsys):
    cli.COMMANDS["/trace"](make_agent(tmp_path, FakeLLM()))
    assert "Nothing traced yet" in capsys.readouterr().out


# --- Display and one-shot mode (Milestone 11) ------------------------------------------------

def make_display(stream=True):
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    return cli.Display(console, stream=stream), console.file


def display_agent(tmp_path, llm, display, **overrides):
    agent = make_agent(tmp_path, llm, **overrides)
    agent.on_text = display.text if display.stream else None
    agent.on_tool_call, agent.on_tool_start = display.tool_call, display.tool_start
    agent.todos.on_change, agent.approver.ask = display.todos, display.ask
    return agent


def test_streamed_answer_is_shown_once(tmp_path):
    display, out = make_display()
    agent = display_agent(tmp_path, FakeLLM(reply("end_turn", text("Hello **there**."))), display)
    assert cli.run_turn(agent, display, "hi") == "Hello **there**."
    assert out.getvalue().count("Hello there.") == 1  # rendered as Markdown, and not printed a second time


def test_answer_that_was_not_streamed_is_printed(tmp_path):
    display, out = make_display()
    agent = display_agent(tmp_path, FakeLLM(reply("tool_use", tool_use("t1", "list_dir", {"path": "."}))), display,
                          max_steps=1)
    answer = cli.run_turn(agent, display, "look around")
    assert answer.startswith("Stopped at step limit")
    text_out = out.getvalue()
    assert "[tool] list_dir" in text_out and "Stopped at step limit" in text_out


def test_failed_turn_is_rolled_back(tmp_path, capsys):
    display, _ = make_display()
    agent = display_agent(tmp_path, FakeLLM(), display)  # no replies: the fake raises
    assert cli.run_turn(agent, display, "hi") is None
    assert agent.messages == [] and "Error:" in capsys.readouterr().err


def test_plan_is_printed_with_marks(tmp_path):
    display, out = make_display()
    display.todos(NS(items=[{"id": "1", "task": "read", "status": "done"}, {"id": "2", "task": "write", "status": "pending"}]))
    assert "[x] 1. read" in out.getvalue() and "[ ] 2. write" in out.getvalue()


def test_one_shot_prints_only_the_answer_to_stdout(tmp_path, monkeypatch, capsys):
    llm = FakeLLM(reply("end_turn", text("42")))
    config = make_agent(tmp_path, llm).config
    monkeypatch.setattr(cli, "config", config)
    monkeypatch.setattr(cli, "start_mcp", lambda quiet=False: None)
    monkeypatch.setattr(cli, "Agent", lambda cfg, **kw: Agent(cfg, llm=llm, **kw))
    assert cli.main(["what", "is", "6*7?"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "42\n"
    assert "Trace saved to" in captured.err


def test_one_shot_exits_1_on_error(tmp_path, monkeypatch, capsys):
    llm = FakeLLM()
    config = make_agent(tmp_path, llm).config
    monkeypatch.setattr(cli, "config", config)
    monkeypatch.setattr(cli, "start_mcp", lambda quiet=False: None)
    monkeypatch.setattr(cli, "Agent", lambda cfg, **kw: Agent(cfg, llm=llm, **kw))
    assert cli.main(["fail"]) == 1
    assert capsys.readouterr().out == ""
