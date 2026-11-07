"""scripts/show_trace.py — Pretty-print a JSONL trace with rich (Milestone 9).

    python scripts/show_trace.py                 # the newest trace in logs/
    python scripts/show_trace.py logs/<id>.jsonl

Each user turn is a branch of the tree, with its model calls and tool calls in order, and each
subagent nested under the run that started it. Times are seconds since the session started.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `agent` imports when run as a script

from rich.console import Console
from rich.markup import escape
from rich.tree import Tree

from agent.config import config
from agent.logging_utils import cost_usd

LINE_CHARS = 110  # one line per event; the trace file has the full text


def one_line(text, limit: int = LINE_CHARS) -> str:
    text = " ".join(str(text).split())
    return escape(text if len(text) <= limit else text[: limit - 1] + "…")


def load(path: Path) -> list[dict]:
    events = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: line {n} is not valid JSON, skipped", file=sys.stderr)  # e.g. cut off by a crash
    return events


def fmt_args(args) -> str:
    if not isinstance(args, dict):
        return str(args)
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" for k, v in args.items())


def describe(event: dict) -> str:
    """The label for one event, without its time."""
    kind = event["type"]
    if kind == "user_input":
        return f"[bold cyan]you[/] {one_line(event.get('text', ''))}"
    if kind == "subagent_start":
        return f"[bold magenta]subagent[/] [dim]{event.get('model') or ''}[/] {one_line(event.get('task', ''))}"
    if kind == "llm_call":
        tools = event.get("tool_calls") or []
        then = f"→ {', '.join(tools)}" if tools else f"→ {one_line(event.get('text', ''), 70)}"
        purpose = " [yellow](summary)[/]" if event.get("purpose") == "summary" else ""
        cached = f" (+{event['cache_read_tokens']:,} cached)" if event.get("cache_read_tokens") else ""
        return (f"[blue]llm[/]{purpose} [dim]{event.get('model')}[/] "
                f"{event.get('input_tokens', 0):,} in{cached} / {event.get('output_tokens', 0):,} out, "
                f"{event.get('latency', 0):.2f}s, {event.get('stop_reason')} {then}")
    if kind == "tool_call":
        style = "red" if event.get("error") else "green"
        approval = event.get("approval", "not_needed")
        approval = "" if approval == "not_needed" else f" [yellow]\\[{approval}][/]"
        return (f"[{style}]tool[/] {escape(event.get('name', '?'))}({one_line(fmt_args(event.get('args')), 70)})"
                f"{approval} {event.get('duration', 0):.2f}s → [dim]{one_line(event.get('output', ''), 70)}[/]")
    if kind == "context_trim":
        what = ", ".join(filter(None, [
            f"cleared {event['cleared']} tool results" if event.get("cleared") else "",
            "summarized the older half" if event.get("summarized") else "",
        ])) or "nothing to trim"
        return (f"[yellow]context[/] {event.get('size_before', 0):,} → {event.get('size_after', 0):,} chars: {what}")
    if kind in ("final_answer", "subagent_end"):
        label = "answer" if kind == "final_answer" else "report"
        stop = event.get("stop")
        stop = "" if stop == "answer" else f" [red]\\[{stop}][/]"
        return (f"[bold]{label}[/]{stop} {event.get('steps', 0)} steps, {event.get('duration', 0):.2f}s, "
                f"{sum(event.get(k, 0) for k in ('input_tokens', 'cache_read_tokens', 'output_tokens')):,} tokens: "
                f"{one_line(event.get('text', ''), 80)}")
    if kind == "error":
        return f"[bold red]error[/] {one_line(event.get('error', ''))}"
    return f"[dim]{escape(kind)}[/] {one_line({k: v for k, v in event.items() if k not in ('ts', 'type', 'run_id')})}"


def build_tree(events: list[dict], title: str) -> Tree:
    tree = Tree(title)
    if not events:
        return tree
    start = datetime.fromisoformat(events[0]["ts"])
    runs: dict[str, Tree] = {}  # run_id -> the branch its events go under
    for event in events:
        offset = (datetime.fromisoformat(event["ts"]) - start).total_seconds()
        label = f"[dim]{offset:7.2f}s[/] {describe(event)}"
        if event["type"] == "user_input":
            runs[event["run_id"]] = tree.add(label)
        elif event["type"] == "subagent_start":
            runs[event["run_id"]] = runs.get(event.get("parent_id"), tree).add(label)
        else:
            runs.get(event.get("run_id"), tree).add(label)
    return tree


def totals(events: list[dict]) -> str:
    calls = [e for e in events if e["type"] == "llm_call"]
    tools = [e for e in events if e["type"] == "tool_call"]
    tokens_in = sum(e.get("input_tokens", 0) for e in calls)
    cached = sum(e.get("cache_read_tokens", 0) for e in calls)
    tokens_out = sum(e.get("output_tokens", 0) for e in calls)
    cost = cost_usd(tokens_in, tokens_out, cached, config)
    errors = sum(1 for e in tools if e.get("error"))
    return (f"{len(calls)} model calls, {len(tools)} tool calls ({errors} errors), "
            f"{tokens_in + cached + tokens_out:,} tokens ({tokens_in:,} in, {cached:,} cached, {tokens_out:,} out), "
            f"about ${cost:.4f} at the prices in config")


def newest_trace(log_dir: Path) -> Path | None:
    traces = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return traces[-1] if traces else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretty-print an Atlas trace.")
    parser.add_argument("file", nargs="?", type=Path, help=f"a .jsonl trace (default: the newest in {config.log_dir})")
    args = parser.parse_args(argv)

    path = args.file or newest_trace(config.log_dir)
    if path is None or not path.exists():
        print(f"No trace found{f' at {path}' if path else f' in {config.log_dir}'}.", file=sys.stderr)
        return 1

    events = load(path)
    console = Console()
    console.print(build_tree(events, f"[bold]Session {escape(path.stem)}[/] [dim]({len(events)} events)[/]"))
    console.print(f"\n[bold]Total:[/] {totals(events)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
