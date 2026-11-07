import cli
from tests.test_loop import FakeLLM, make_agent


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
