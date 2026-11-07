"""Milestone 10: the eval runner's own logic, offline. Scripted FakeLLMs play the agent and the judge."""
import json
from types import SimpleNamespace as NS

import pytest

from agent.config import Config
from agent.loop import Agent
from evals.run_evals import (
    TASKS, Judge, Task, load_tasks, parse_verdict, prompt_fingerprint, run_suite, run_task, save_results, summarize,
)
from tests.test_loop import FakeLLM, reply, text, tool_use


def base_config(tmp_path) -> Config:
    return Config(api_key="sk-test", workspace=tmp_path / "w", data_dir=tmp_path / "d", log_dir=tmp_path / "l",
                  knowledge_dir=tmp_path / "k", enable_web_search=False)


def scripted(*replies):
    """A make_agent that gives every run a fresh copy of the same script."""
    def make(config, tracer, on_tool_call):
        return Agent(config, llm=FakeLLM(*replies), tracer=tracer, on_tool_call=on_tool_call)
    return make


def judge_says(passed: bool, reason="because"):
    verdict = json.dumps({"pass": passed, "reason": reason})
    return Judge(NS(complete=lambda system, messages, tools=None: reply("end_turn", text(verdict))))


def run(tmp_path, task, *replies, judge=None):
    return run_task(task, 1, base_config(tmp_path), judge, tmp_path / "traces", scripted(*replies))


def test_shipped_tasks_load():
    tasks = load_tasks(TASKS)
    assert 15 <= len(tasks) <= 25
    kinds = {c["type"] for t in tasks for c in t.check}
    assert {"run", "tool_called", "tool_not_called", "judge"} <= kinds  # programmatic, trajectory and judge
    assert sum(any(c.get("grounded") for c in t.check) for t in tasks) >= 6
    assert any(t.approve == "deny" for t in tasks)


def test_load_rejects_bad_tasks(tmp_path):
    bad = tmp_path / "t.jsonl"
    bad.write_text(json.dumps({"id": "a", "prompt": "p", "check": [{"type": "nope"}]}) + "\n")
    with pytest.raises(ValueError, match="unknown check types"):
        load_tasks(bad)
    line = json.dumps({"id": "a", "prompt": "p", "check": [{"type": "stop"}]})
    bad.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(bad)


def test_programmatic_and_trajectory_checks_pass(tmp_path):
    task = Task("write", "make hi.py", [
        {"type": "file_contains", "path": "hi.py", "text": "print"},
        {"type": "run", "cmd": ["python", "hi.py"], "contains": "hello"},
        {"type": "tool_called", "tool": "write_file"},
        {"type": "tool_not_called", "tool": "run_shell"},
        {"type": "answer_contains", "text": ["done"]},
        {"type": "max_steps", "n": 2},
        {"type": "stop", "is": "answer"},
    ])
    result = run(tmp_path, task,
                 reply("tool_use", tool_use("t1", "write_file", {"path": "hi.py", "content": "print('hello')"})),
                 reply("end_turn", text("Done.")))
    assert result.passed, result.failures()
    assert result.steps == 2 and result.tools == ["write_file"] and result.stop == "answer"
    assert result.tokens == 4  # two fake calls, 1 in + 1 out each
    assert (tmp_path / "traces" / "write-1.jsonl").exists()


def test_failures_say_why(tmp_path):
    task = Task("t", "p", [{"type": "file_exists", "path": "x.txt"}, {"type": "answer_contains", "text": "391"},
                           {"type": "tool_called", "tool": "read_file"}])
    result = run(tmp_path, task, reply("end_turn", text("I think it's 390.")))
    assert not result.passed
    assert [c.passed for c in result.checks] == [False, False, False]
    assert "x.txt is missing" in result.failures() and "'391'" in result.failures()


def test_setup_files_outside_and_memory(tmp_path):
    task = Task("setup", "p", [{"type": "file_contains", "path": "notes/a.txt", "text": "alpha"},
                               {"type": "outside_intact"}, {"type": "memory_contains", "text": "BLUEFINCH"}],
                files={"notes/a.txt": "alpha"}, outside={"keep.txt": "x"}, memory=["codename BLUEFINCH"])
    assert run(tmp_path, task, reply("end_turn", text("ok"))).passed


def test_outside_file_escape_is_caught(tmp_path):
    """An agent that gets out of the workspace through the shell fails outside_intact."""
    task = Task("escape", "p", [{"type": "outside_intact"}], outside={"keep.txt": "x"})
    result = run(tmp_path, task,
                 reply("tool_use", tool_use("t1", "run_shell", {"command": "rm ../keep.txt"})),
                 reply("end_turn", text("deleted")))
    assert not result.passed and "keep.txt" in result.failures()


def test_deny_records_the_attempt_but_never_runs_it(tmp_path):
    task = Task("deny", "p", [{"type": "tool_not_called", "tool": "run_shell"}, {"type": "outside_intact"}],
                outside={"keep.txt": "x"}, approve="deny")
    result = run(tmp_path, task,
                 reply("tool_use", tool_use("t1", "run_shell", {"command": "rm ../keep.txt"})),
                 reply("end_turn", text("I wasn't allowed.")))
    assert [c.passed for c in result.checks] == [False, True]  # the attempt counts; the file survived


def test_memory_written_by_the_agent(tmp_path):
    task = Task("mem", "p", [{"type": "memory_contains", "text": "Virak"}, {"type": "tool_called", "tool": "remember"}])
    result = run(tmp_path, task,
                 reply("tool_use", tool_use("t1", "remember", {"text": "User's name is Virak"})),
                 reply("end_turn", text("Noted.")))
    assert result.passed, result.failures()


def test_judge_check_sees_answer_and_tool_output(tmp_path):
    seen = {}

    def complete(system, messages, tools=None):
        seen["prompt"] = messages[0]["content"]
        return reply("end_turn", text('Verdict: {"pass": true, "reason": "cited the file"}'))

    task = Task("j", "read a.txt", [{"type": "judge", "rubric": "Must quote a.txt", "grounded": True}],
                files={"a.txt": "the secret word is plum"})
    result = run(tmp_path, task,
                 reply("tool_use", tool_use("t1", "read_file", {"path": "a.txt"})),
                 reply("end_turn", text("It says plum.")), judge=Judge(NS(complete=complete)))
    assert result.passed and result.checks[0].detail == "cited the file"
    assert "the secret word is plum" in seen["prompt"]  # the full tool output, not the trace preview
    assert "It says plum." in seen["prompt"] and "Must quote a.txt" in seen["prompt"]
    assert "search_knowledge" in seen["prompt"]  # the grounding rubric was added


def test_judge_fail_and_bad_verdicts(tmp_path):
    task = Task("j", "p", [{"type": "judge", "rubric": "r"}])
    result = run(tmp_path, task, reply("end_turn", text("x")), judge=judge_says(False, "made it up"))
    assert not result.passed and "made it up" in result.failures()
    assert parse_verdict("no json here")[0] is False
    assert parse_verdict('{"pass": false}') == (False, "(no reason)")
    assert run(tmp_path, task, reply("end_turn", text("x")), judge=None).checks[0].detail == "no judge configured"


def test_agent_crash_is_a_failed_run_not_a_crashed_suite(tmp_path):
    def make(config, tracer, on_tool_call):
        return Agent(config, llm=NS(complete=lambda *a: (_ for _ in ()).throw(RuntimeError("API down"))),
                     tracer=tracer, on_tool_call=on_tool_call)

    task = Task("boom", "p", [{"type": "stop"}])
    result = run_task(task, 1, base_config(tmp_path), None, tmp_path / "traces", make)
    assert not result.passed and result.stop == "crash" and "API down" in result.error


def test_knowledge_task_needs_an_index(tmp_path):
    with pytest.raises(RuntimeError, match="no index"):
        run(tmp_path, Task("k", "p", [{"type": "stop"}], knowledge=True), reply("end_turn", text("x")))


def test_suite_repeats_summarizes_and_saves(tmp_path):
    tasks = [Task("good", "p", [{"type": "stop"}]), Task("bad", "p", [{"type": "answer_contains", "text": "zzz"}])]
    seen = []
    results = run_suite(tasks, base_config(tmp_path), 3, None, tmp_path / "traces", workers=2,
                        make_agent=scripted(reply("end_turn", text("ok"))), on_result=seen.append)
    assert len(results) == len(seen) == 6
    assert sorted((r.task, r.repeat) for r in results)[:3] == [("bad", 1), ("bad", 2), ("bad", 3)]
    summary = summarize(results)
    assert (summary.runs, summary.runs_passed, summary.tasks, summary.tasks_all_passed) == (6, 3, 2, 1)
    assert summary.pass_rate == 0.5

    out = tmp_path / "results" / "x.json"
    save_results(out, results, {"prompt": prompt_fingerprint()})
    data = json.loads(out.read_text())
    assert data["summary"]["pass_rate"] == 0.5 and len(data["runs"]) == 6 and data["prompt"]


def test_prompt_fingerprint_follows_the_rules(monkeypatch):
    from agent import prompts

    before = prompt_fingerprint()
    monkeypatch.setattr(prompts, "RULES", prompts.RULES + ["Always answer in French."])
    assert prompt_fingerprint() != before
