import time

from agent.tools.registry import ToolRegistry
from agent.tools.shell import shell_tool


def make_registry(tmp_path, timeout=5):
    registry = ToolRegistry()
    registry.register(shell_tool(tmp_path, timeout))
    return registry


def test_runs_in_workspace(tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    assert make_registry(tmp_path).run("run_shell", {"command": "ls"}) == ("hello.txt", False)


def test_stderr_and_exit_code_are_returned(tmp_path):
    content, is_error = make_registry(tmp_path).run("run_shell", {"command": "echo oops >&2; exit 3"})
    assert not is_error
    assert "oops" in content and "[exit code 3]" in content


def test_no_output(tmp_path):
    assert make_registry(tmp_path).run("run_shell", {"command": "true"}) == ("(no output)", False)


def test_does_not_wait_for_input(tmp_path):
    content, is_error = make_registry(tmp_path).run("run_shell", {"command": "read x; echo got:$x"})
    assert content.startswith("got:")


def test_timeout_kills_command_and_children(tmp_path):
    start = time.monotonic()
    content, is_error = make_registry(tmp_path, timeout=1).run(
        "run_shell", {"command": "echo started; sleep 30 & sleep 30"}
    )
    assert time.monotonic() - start < 5
    assert is_error and "timed out after 1s" in content and "started" in content
