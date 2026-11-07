"""Milestone 9: the JSONL tracer, secret masking, /cost and show_trace. Offline, with scripted FakeLLMs."""
import json

import pytest
from rich.console import Console

import cli
from agent.config import Config
from agent.context import llm_summarizer
from agent.guardrails import Approver
from agent.logging_utils import NullTracer, Tracer, Usage, cost_usd, mask_secrets
from agent.loop import _Metered
from scripts import show_trace
from tests.test_loop import FakeLLM, make_agent, reply, text, tool_use


def read_events(tracer):
    return [json.loads(line) for line in tracer.path.read_text().splitlines()]


def traced_agent(tmp_path, llm, **overrides):
    agent = make_agent(tmp_path, llm, **overrides)
    agent.tracer = Tracer(tmp_path / "logs", session_id="test")
    return agent


# --- Tracer -----------------------------------------------------------------------------------

def test_each_event_is_written_and_flushed_at_once(tmp_path):
    tracer = Tracer(tmp_path, session_id="s1")
    assert not tracer.path.exists()  # nothing logged, no file
    tracer.event("user_input", "r1", text="hi")
    # Readable before close(): a crash right now would still leave this line on disk.
    [event] = read_events(tracer)
    assert event["type"] == "user_input" and event["run_id"] == "r1" and event["text"] == "hi"
    assert tracer.path == tmp_path / "s1.jsonl" and "ts" in event


def test_secrets_are_masked_before_writing(tmp_path):
    tracer = Tracer(tmp_path, session_id="s", secrets=("my-configured-key-123",))
    tracer.event("tool_call", "r", args={"command": "curl -H 'Authorization: Bearer abcdefghijklmnop1234' x",
                                         "api_key": "plain", "note": "uses my-configured-key-123"},
                 output="key is sk-abcdefghijklmnopqrstuvwx and password=hunter2hunter2")
    raw = tracer.path.read_text()
    for secret in ("abcdefghijklmnop1234", "plain", "my-configured-key-123", "sk-abcdefghijklmnopqrstuvwx", "hunter2"):
        assert secret not in raw
    output = read_events(tracer)[0]["output"]
    assert "password=***" in output  # still says what was there


def test_masking_leaves_normal_text_alone():
    value = {"path": "notes/token_counts.md", "count": 3, "items": ["a", "b"]}
    assert mask_secrets(value) == value


def test_null_tracer_records_nothing(tmp_path):
    agent = make_agent(tmp_path, FakeLLM(reply("end_turn", text("hi"))))
    assert isinstance(agent.tracer, NullTracer)
    agent.run("hello")
    assert not (tmp_path / "l").exists() or not list((tmp_path / "l").glob("*.jsonl"))


# --- What the agent records -------------------------------------------------------------------

def test_a_turn_is_traced_in_order(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "hey"})),
        reply("end_turn", text("Done.")),
    )
    agent = traced_agent(tmp_path, llm, auto_approve=True)
    agent.run("make a file")

    events = read_events(agent.tracer)
    assert [e["type"] for e in events] == ["user_input", "llm_call", "tool_call", "llm_call", "final_answer"]
    assert len({e["run_id"] for e in events}) == 1

    call = events[1]
    assert call["input_tokens"] == 1 and call["output_tokens"] == 1 and call["tool_calls"] == ["write_file"]
    assert call["stop_reason"] == "tool_use" and call["latency"] >= 0
    tool = events[2]
    assert tool["name"] == "write_file" and tool["args"] == {"path": "a.txt", "content": "hey"}
    assert tool["approval"] == "auto" and tool["error"] is False and "duration" in tool
    end = events[-1]
    assert end["text"] == "Done." and end["stop"] == "answer" and end["steps"] == 2
    assert end["input_tokens"] + end["output_tokens"] == 4


def test_denied_and_failed_tools_are_marked(tmp_path):
    llm = FakeLLM(
        reply("tool_use", tool_use("t1", "write_file", {"path": "a.txt", "content": "x"}),
              tool_use("t2", "read_file", {"path": "missing.txt"})),
        reply("end_turn", text("ok")),
    )
    agent = traced_agent(tmp_path, llm)
    agent.approver.ask = lambda name, args: "n"
    agent.run("go")
    write, read = [e for e in read_events(agent.tracer) if e["type"] == "tool_call"]
    assert write["approval"] == "user_no" and write["error"] is True
    assert read["approval"] == "not_needed" and read["error"] is True


def test_step_limit_is_recorded(tmp_path):
    llm = FakeLLM(reply("tool_use", tool_use("t1", "list_dir", {})))
    agent = traced_agent(tmp_path, llm, max_steps=1)
    agent.run("loop")
    assert read_events(agent.tracer)[-1]["stop"] == "step_limit"


def test_a_crash_is_logged_before_it_propagates(tmp_path):
    class Broken:
        def complete(self, *a):
            raise RuntimeError("API is down")

    agent = traced_agent(tmp_path, Broken())
    with pytest.raises(RuntimeError):
        agent.run("hi")
    last = read_events(agent.tracer)[-1]
    assert last["type"] == "error" and last["error"] == "RuntimeError: API is down"


def test_subagent_events_point_at_their_parent(tmp_path):
    child = FakeLLM(reply("tool_use", tool_use("c1", "list_dir", {})), reply("end_turn", text("child report")))
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "look around"})),
        reply("end_turn", text("ok")),
    )
    agent = traced_agent(tmp_path, parent)
    agent._subagent_llm = child
    agent.run("go")

    events = read_events(agent.tracer)
    parent_run = events[0]["run_id"]
    start = next(e for e in events if e["type"] == "subagent_start")
    end = next(e for e in events if e["type"] == "subagent_end")
    assert start["parent_id"] == parent_run and start["task"] == "look around"
    assert end["parent_id"] == parent_run and end["text"] == "child report"
    child_events = [e["type"] for e in events if e["run_id"] == start["run_id"]]
    assert child_events == ["subagent_start", "llm_call", "tool_call", "llm_call", "subagent_end"]
    # The full child transcript is in the log, even though the parent's history only has the report.
    assert events[-1]["type"] == "final_answer" and events[-1]["input_tokens"] + events[-1]["output_tokens"] == 8


def test_context_trim_and_summary_are_traced_and_counted(tmp_path):
    summarizer = FakeLLM(reply("end_turn", text("the summary")))
    summarizer.model = "cheap-model"
    llm = FakeLLM(reply("end_turn", text("first")), reply("end_turn", text("second")))
    agent = traced_agent(tmp_path, llm, context_char_budget=300, keep_recent_messages=1)
    agent._summarizer = llm_summarizer(_Metered(agent, summarizer, "summary"))

    agent.run("x" * 200)
    agent.run("y" * 200)

    events = read_events(agent.tracer)
    trim = next(e for e in events if e["type"] == "context_trim")
    assert trim["summarized"] is True and trim["size_after"] < trim["size_before"]
    summary_call = next(e for e in events if e.get("purpose") == "summary")
    assert summary_call["model"] == "cheap-model"
    assert agent.usage.by_model["cheap-model"]["calls"] == 1  # the summary shows up in /cost
    assert agent.total_tokens == 6


# --- Approvals and cost -----------------------------------------------------------------------

def test_approver_says_how_it_decided():
    assert Approver(auto_approve=True).decide("x", {}) == (True, "auto")
    assert Approver().decide("x", {}) == (False, "no_one_to_ask")
    answers = iter(["a"])
    approver = Approver(ask=lambda n, a: next(answers))
    assert approver.decide("x", {}) == (True, "always")
    assert approver.decide("x", {}) == (True, "always")  # not asked again
    assert Approver(ask=lambda n, a: "y").decide("x", {}) == (True, "user_yes")


def test_usage_and_cost():
    usage, other = Usage(), Usage()
    usage.add("big", 1_000_000, 0)
    other.add("big", 0, 1_000_000, cache_read_tokens=1_000_000)
    other.add("small", 10, 5)
    usage.merge(other)
    assert usage.by_model["big"] == {"calls": 2, "input": 1_000_000, "cache_read": 1_000_000, "output": 1_000_000}
    assert usage.input_tokens == 1_000_010 and usage.output_tokens == 1_000_005
    assert usage.cache_read_tokens == 1_000_000
    prices = Config(price_input_per_mtok=3.0, price_cache_read_per_mtok=0.3, price_output_per_mtok=15.0)
    assert cost_usd(1_000_000, 1_000_000, 1_000_000, prices) == pytest.approx(18.3)


def test_cached_input_is_counted_apart(tmp_path):
    response = reply("end_turn", text("hi"))
    response.usage.cache_read_input_tokens = 500
    agent = traced_agent(tmp_path, FakeLLM(response))
    agent.run("hello")
    call = next(e for e in read_events(agent.tracer) if e["type"] == "llm_call")
    assert call["input_tokens"] == 1 and call["cache_read_tokens"] == 500
    assert agent.usage.cache_read_tokens == 500


def test_cost_command(tmp_path, capsys):
    agent = make_agent(tmp_path, FakeLLM(), price_input_per_mtok=1.0, price_output_per_mtok=2.0,
                       price_cache_read_per_mtok=0.1)
    cli.COMMANDS["/cost"](agent)
    assert "No model calls yet" in capsys.readouterr().out

    agent.usage.add("deepseek-v4-pro", 500_000, 250_000, cache_read_tokens=1_000_000)
    cli.COMMANDS["/cost"](agent)
    out = capsys.readouterr().out
    assert "deepseek-v4-pro" in out and "1,750,000 tokens" in out and "$1.1000" in out


# --- show_trace -------------------------------------------------------------------------------

def render(events) -> str:
    console = Console(record=True, width=200)
    console.print(show_trace.build_tree(events, "trace"))
    return console.export_text()


def test_show_trace_nests_subagents_under_their_parent(tmp_path):
    child = FakeLLM(reply("tool_use", tool_use("c1", "list_dir", {})), reply("end_turn", text("child report")))
    parent = FakeLLM(
        reply("tool_use", tool_use("p1", "delegate_task", {"task": "look around"})),
        reply("end_turn", text("all done")),
    )
    agent = traced_agent(tmp_path, parent)
    agent._subagent_llm = child
    agent.run("go")

    lines = render(show_trace.load(agent.tracer.path)).splitlines()
    indent = {key: next(len(l) - len(l.lstrip("│├└─ ")) for l in lines if key in l)
              for key in ("you go", "subagent", "tool list_dir", "tool delegate_task", "answer")}
    assert indent["subagent"] > indent["you go"]
    assert indent["tool list_dir"] > indent["subagent"]  # the child's tool sits under the subagent
    assert indent["tool delegate_task"] == indent["subagent"]  # the parent's tool sits beside it
    assert "all done" in "\n".join(lines) and "1 in / 1 out" in "\n".join(lines)


def test_show_trace_main_reads_a_file_and_skips_broken_lines(tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    good = {"ts": "2026-11-07T10:00:00.000", "type": "user_input", "run_id": "r", "text": "hello"}
    path.write_text(json.dumps(good) + "\n{not json\n")
    assert show_trace.main([str(path)]) == 0
    captured = capsys.readouterr()
    assert "hello" in captured.out and "Total:" in captured.out and "line 2" in captured.err
