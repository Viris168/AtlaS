import pytest

from agent.tools.registry import Tool, ToolRegistry


def echo_tool():
    return Tool(
        name="echo",
        description="Echo the text back.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        fn=lambda text: text,
    )


def test_schemas_match_api_shape():
    registry = ToolRegistry()
    registry.register(echo_tool())
    assert registry.schemas() == [
        {
            "name": "echo",
            "description": "Echo the text back.",
            "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        }
    ]


def test_run_ok():
    registry = ToolRegistry()
    registry.register(echo_tool())
    assert registry.run("echo", {"text": "hi"}) == ("hi", False)


def test_duplicate_name_rejected():
    registry = ToolRegistry()
    registry.register(echo_tool())
    with pytest.raises(ValueError):
        registry.register(echo_tool())


def test_unknown_tool_is_error():
    content, is_error = ToolRegistry().run("nope", {})
    assert is_error and "unknown tool" in content


def test_bad_arguments_is_error():
    registry = ToolRegistry()
    registry.register(echo_tool())
    content, is_error = registry.run("echo", {"wrong": 1})
    assert is_error and "bad arguments" in content


def test_exception_is_error():
    def boom():
        raise RuntimeError("disk on fire")

    registry = ToolRegistry()
    registry.register(Tool("boom", "Fails.", {"type": "object", "properties": {}}, boom))
    content, is_error = registry.run("boom", {})
    assert is_error and "RuntimeError: disk on fire" in content


def test_output_truncated():
    registry = ToolRegistry(max_output_chars=10)
    registry.register(echo_tool())
    content, is_error = registry.run("echo", {"text": "x" * 25})
    assert not is_error
    assert content.startswith("x" * 10) and "truncated 15 chars" in content


def test_risky_flags():
    from pathlib import Path

    from agent.tools.files import file_tools
    from agent.tools.shell import shell_tool

    risky = {t.name: t.risky for t in file_tools(Path(".")) + [shell_tool(Path("."))]}
    assert risky == {"read_file": False, "write_file": True, "list_dir": False, "run_shell": True}
