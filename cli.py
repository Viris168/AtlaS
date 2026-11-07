"""Command-line interface for Atlas (Milestone 1, slash commands from Milestones 5, 5B, 6, 8 and 9,
subagents from 7, rich output, streaming, /help, /reset, /trace and one-shot mode from 11).

    python cli.py                  # chat
    python cli.py "one-shot task"  # run one task, print the answer to stdout, exit (1 on error)
"""
import re
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from agent.config import config
from agent.context import estimate_size
from agent.logging_utils import Tracer, cost_usd
from agent.loop import Agent
from agent.tools.mcp_bridge import MCPBridge
from agent.tools.memory_tools import render_entries
from agent.tools.todo import MARKS

EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}
STATUS_STYLES = {"done": "green", "in_progress": "yellow", "pending": "dim"}


class Display:
    """What a turn shows: the model's text streamed as Markdown, a spinner while waiting, coloured tool calls.

    rich runs one Live display at a time, so the spinner and the streaming text take turns, and both
    stop before anything else is printed or the approval prompt reads from the keyboard.
    With stream=False (one-shot mode) the text isn't shown here; the caller prints the answer.
    """

    def __init__(self, console: Console, stream: bool = True):
        self.console, self.stream = console, stream
        self._live: Live | None = None
        self._text = ""  # the text segment streaming now: the model's text up to its next tool call
        self._last_text = ""

    def _stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._text:
            self._last_text, self._text = self._text, ""

    def spin(self, label: str) -> None:
        self._stop()
        self._live = Live(Spinner("dots", Text(label, style="dim")), console=self.console, transient=True)
        self._live.start()

    def start_turn(self) -> None:
        self._last_text = ""
        self.spin("thinking...")

    def text(self, chunk: str) -> None:
        if not self._text:  # a new segment: replace the spinner with the text
            self._stop()
            self._live = Live(Markdown(""), console=self.console, refresh_per_second=10)
            self._live.start()
        self._text += chunk
        self._live.update(Markdown(self._text))

    def tool_start(self, name: str, args: dict) -> None:
        self.spin(f"running {name}...")

    def tool_call(self, name: str, args: dict, result: str) -> None:
        # A subagent's calls arrive as "subagent > read_file". Its plan is its own business, so skip todos at any level.
        if not name.split(" > ")[-1].startswith("todo_"):  # todos() already prints the main agent's plan
            self._stop()
            preview = result if len(result) <= 200 else result[:200] + "..."
            line = Text("  ")
            line.append("[tool] ", style="bold cyan")
            line.append(name, style="cyan")
            line.append(f"({args})", style="dim")
            line.append(" -> ")
            line.append(preview, style="red" if result.startswith(("Error", "Denied")) else "default")
            self.console.print(line, soft_wrap=True)
        self.spin("thinking...")  # the next model call starts right away

    def todos(self, todos) -> None:
        self._stop()
        self.console.print(Text("  [plan]", style="bold magenta"))
        for item in todos.items:
            self.console.print(Text(f"    {MARKS[item['status']]} {item['id']}. {item['task']}",
                                    style=STATUS_STYLES[item["status"]]))

    def ask(self, name: str, args: dict) -> str:
        self._stop()
        self.console.print(Text.assemble(("  [approve?] ", "bold yellow"), (name, "yellow"), (f"({args})", "dim")), soft_wrap=True)
        try:
            return self.console.input("  Run it? [y]es / [n]o / [a]lways for this tool: ")
        except EOFError:  # nobody at the keyboard (stdin closed or piped): the safe answer
            return "n"

    def end_turn(self, answer: str | None = None) -> None:
        """Stop the live display. With an answer, print it unless it was just streamed (a step-limit or
        budget message never is)."""
        self._stop()
        if answer is not None and self.stream and "".join(answer.split()) != "".join(self._last_text.split()):
            self.console.print(Markdown(answer))


def show_memory(agent: Agent) -> None:
    """List every saved long-term memory."""
    entries = agent.memory.entries
    if not entries:
        print("  (no memories saved)")
        return
    print(f"  {len(entries)} memories:")
    for line in render_entries(entries).splitlines():
        print(f"    {line}")


def ingest_knowledge(agent: Agent) -> None:
    """Index the files in the knowledge folder."""
    from agent.rag.ingest import run_ingest  # loads the embedding model, so only when asked

    print(f"  Indexing {agent.config.knowledge_dir} ...")
    try:
        report = run_ingest(agent.config)
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        return
    for line in report.summary().splitlines():
        print(f"    {line}")


def show_context(agent: Agent) -> None:
    """Show how full the context window is."""
    size, budget = estimate_size(agent.messages), agent.context.budget
    print(f"  {size:,} of {budget:,} characters ({size / budget:.0%}), {len(agent.messages)} messages")
    print(f"  Old tool output cleared: {agent.context.clears} times")
    print(f"  History summarized: {agent.context.summaries} times")


def show_tools(agent: Agent) -> None:
    """List the tools the model can call."""
    tools = agent.tool_schemas()
    print(f"  {len(tools)} tools (* asks for approval first):")
    for tool in tools:
        risky = "*" if agent.registry.is_risky(tool["name"]) else " "
        # First sentence only; a sentence starts with a capital, so "e.g. 3" doesn't cut it short.
        summary = re.split(r"\.\s+(?=[A-Z])", tool.get("description", "runs on the provider's side"))[0].rstrip(".")
        print(f"   {risky} {tool['name']}: {summary[:90]}")


def show_mcp(agent: Agent) -> None:
    """Show the status of each MCP server."""
    rows = agent.mcp.status() if agent.mcp else []
    if not rows:
        print(f"  (no MCP servers configured; add them to {agent.config.mcp_config_path.name})")
        return
    for name, state in rows:
        print(f"    {name}: {state}")


def show_cost(agent: Agent) -> None:
    """Show tokens used and the estimated cost so far."""
    config, usage = agent.config, agent.usage
    if not usage.by_model:
        print("  No model calls yet this session.")
        return
    print(f"  {'model':<24} {'calls':>5} {'input':>10} {'cached':>10} {'output':>9}")
    for model, row in usage.by_model.items():
        print(f"  {model:<24} {row['calls']:>5} {row['input']:>10,} {row['cache_read']:>10,} {row['output']:>9,}")
    total_in, cached, total_out = usage.input_tokens, usage.cache_read_tokens, usage.output_tokens
    cost = cost_usd(total_in, total_out, cached, config)
    print(f"  Total: {total_in + cached + total_out:,} tokens ({total_in:,} in, {cached:,} cached, {total_out:,} out), "
          f"about ${cost:.4f}")
    print(f"  Prices per million tokens: ${config.price_input_per_mtok:g} in, ${config.price_cache_read_per_mtok:g} cached, "
          f"${config.price_output_per_mtok:g} out (AGENT_PRICE_IN / _CACHE_READ / _OUT)")
    if agent.tracer.path:
        print(f"  Trace: {agent.tracer.path}")


def reset_conversation(agent: Agent) -> None:
    """Start a new conversation; memories and /cost totals are kept."""
    agent.reset()
    print("  Conversation cleared. Memories are kept.")


def show_trace(agent: Agent) -> None:
    """Show where this session's trace file is."""
    path = agent.tracer.path
    if not path or not path.exists():
        print("  Nothing traced yet this session.")
        return
    print(f"  {path}")
    print(f"  View it with: python scripts/show_trace.py {path}")


def show_help(agent: Agent) -> None:
    """List these commands."""
    print("  Commands (they run here and never reach the model):")
    for name, command in COMMANDS.items():
        print(f"    {name:<9} {command.__doc__.splitlines()[0]}")
    print(f"    {'/exit':<9} Quit (or type 'exit', or press Ctrl+D).")


# Slash commands run here, never reach the model. /help prints the first line of each docstring.
COMMANDS = {
    "/help": show_help,
    "/reset": reset_conversation,
    "/memory": show_memory,
    "/ingest": ingest_knowledge,
    "/context": show_context,
    "/tools": show_tools,
    "/mcp": show_mcp,
    "/cost": show_cost,
    "/trace": show_trace,
}


def start_mcp(quiet: bool = False) -> MCPBridge | None:
    try:
        bridge = MCPBridge.from_config(config)
    except ValueError as e:  # a broken mcp_servers.json: say so, but still start without MCP
        print(f"Warning: MCP disabled: {e}", file=sys.stderr)
        return None
    if bridge.servers and not quiet:
        print(f"Starting {len(bridge.servers)} MCP server(s)...")
    bridge.start()
    for name, state in bridge.status():
        if state.startswith("failed"):
            print(f"Warning: MCP server '{name}' {state}", file=sys.stderr)
    return bridge


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    one_shot = " ".join(argv).strip()
    mcp = start_mcp(quiet=bool(one_shot))
    try:
        return run_session(mcp, one_shot)
    finally:
        if mcp:
            mcp.close()


def run_session(mcp: MCPBridge | None, one_shot: str = "") -> int:
    tracer = Tracer(config.log_dir, secrets=(config.api_key,))
    try:
        # One-shot: progress goes to stderr and only the answer to stdout, so `atlas "task" > out.md` works.
        console = Console(stderr=True) if one_shot else Console()
        display = Display(console, stream=not one_shot)
        try:
            agent = Agent(config, on_tool_call=display.tool_call, ask_approval=display.ask, on_todo_change=display.todos,
                          on_tool_start=display.tool_start, on_text=display.text if display.stream else None,
                          mcp=mcp, tracer=tracer)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if one_shot:
            answer = run_turn(agent, display, one_shot)
            if answer is None:
                return 1
            print(answer)
            return 0
        return chat(agent, display)
    finally:
        tracer.close()
        if tracer.path.exists():
            print(f"Trace saved to {tracer.path}", file=sys.stderr)


def run_turn(agent: Agent, display: Display, user_input: str) -> str | None:
    """One turn with the display running. Returns the answer, or None if it failed or was interrupted."""
    # A copy, not an index: context management may rewrite earlier messages during the turn.
    checkpoint = list(agent.messages)
    display.start_turn()
    try:
        answer = agent.run(user_input)
    except KeyboardInterrupt:
        display.end_turn()
        agent.messages[:] = checkpoint
        print("\n(interrupted)", file=sys.stderr)
        return None
    except Exception as e:
        # Roll back the half-finished turn so the history stays valid for the next message.
        display.end_turn()
        agent.messages[:] = checkpoint
        print(f"Error: {e}", file=sys.stderr)
        return None
    display.end_turn(answer)
    return answer


def chat(agent: Agent, display: Display) -> int:
    console = display.console
    console.print(f"[bold]Atlas[/] ({config.model}). Type /help for commands, 'exit' or Ctrl+D to quit.")
    while True:
        try:
            user_input = console.input("\n[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            return 0
        if user_input.startswith("/"):
            command = COMMANDS.get(user_input.split()[0].lower())
            if command:
                command(agent)
            else:
                print("  Unknown command. Type /help to see them all.")
            continue

        console.print("[bold blue]atlas>[/]")
        run_turn(agent, display, user_input)


if __name__ == "__main__":
    sys.exit(main())
