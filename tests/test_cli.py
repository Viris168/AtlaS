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
