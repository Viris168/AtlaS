from agent.tools.files import file_tools
from agent.tools.registry import ToolRegistry


def make_registry(tmp_path):
    registry = ToolRegistry()
    for tool in file_tools(tmp_path):
        registry.register(tool)
    return registry


def test_write_then_read(tmp_path):
    registry = make_registry(tmp_path)
    content, is_error = registry.run("write_file", {"path": "notes/a.txt", "content": "hello"})
    assert not is_error
    assert (tmp_path / "notes" / "a.txt").read_text() == "hello"
    assert registry.run("read_file", {"path": "notes/a.txt"}) == ("hello", False)


def test_list_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("x")
    registry = make_registry(tmp_path)
    assert registry.run("list_dir", {}) == ("sub/\nb.txt", False)


def test_list_empty_dir(tmp_path):
    assert make_registry(tmp_path).run("list_dir", {"path": "."}) == ("(empty folder)", False)


def test_read_missing_file(tmp_path):
    content, is_error = make_registry(tmp_path).run("read_file", {"path": "nope.txt"})
    assert is_error and "No file" in content


def test_escape_is_refused(tmp_path):
    registry = make_registry(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    (tmp_path / "secret.txt").write_text("password")
    for path in ["../secret.txt", str(tmp_path / "secret.txt"), "/etc/passwd"]:
        content, is_error = registry.run("read_file", {"path": path})
        assert is_error and "outside the workspace" in content
    content, is_error = registry.run("write_file", {"path": "../evil.txt", "content": "x"})
    assert is_error and not (tmp_path / "evil.txt").exists()
