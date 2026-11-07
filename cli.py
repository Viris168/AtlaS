"""Command-line REPL for Atlas (Milestone 1, slash commands from Milestones 5, 5B, 6, 8 and 9, subagents from 7, /help, /reset and /trace from 11)."""
import re
import sys

from agent.config import config
from agent.context import estimate_size
from agent.logging_utils import Tracer, cost_usd
from agent.loop import Agent
from agent.tools.mcp_bridge import MCPBridge
from agent.tools.memory_tools import render_entries

EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}


def show_tool_call(name: str, args: dict, result: str) -> None:
    # A subagent's calls arrive as "subagent > read_file". Its plan is its own business, so skip todos at any level.
    if name.split(" > ")[-1].startswith("todo_"):
        return  # show_todos already prints the main agent's plan in a readable form
    preview = result if len(result) <= 200 else result[:200] + "..."
    print(f"  [tool] {name}({args}) -> {preview}")


def show_todos(todos) -> None:
    print("  [plan]")
    for line in todos.render().splitlines():
        print(f"    {line}")


def ask_approval(name: str, args: dict) -> str:
    print(f"  [approve?] {name}({args})")
    try:
        return input("  Run it? [y]es / [n]o / [a]lways for this tool: ")
    except EOFError:
        return "n"


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


def start_mcp() -> MCPBridge | None:
    try:
        bridge = MCPBridge.from_config(config)
    except ValueError as e:  # a broken mcp_servers.json: say so, but still start without MCP
        print(f"Warning: MCP disabled: {e}", file=sys.stderr)
        return None
    if bridge.servers:
        print(f"Starting {len(bridge.servers)} MCP server(s)...")
    bridge.start()
    for name, state in bridge.status():
        if state.startswith("failed"):
            print(f"Warning: MCP server '{name}' {state}", file=sys.stderr)
    return bridge


def main() -> int:
    mcp = start_mcp()
    try:
        return repl(mcp)
    finally:
        if mcp:
            mcp.close()


def repl(mcp: MCPBridge | None) -> int:
    tracer = Tracer(config.log_dir, secrets=(config.api_key,))
    try:
        return chat(mcp, tracer)
    finally:
        tracer.close()
        if tracer.path.exists():
            print(f"Trace saved to {tracer.path}", file=sys.stderr)


def chat(mcp: MCPBridge | None, tracer: Tracer) -> int:
    try:
        agent = Agent(config, on_tool_call=show_tool_call, ask_approval=ask_approval, on_todo_change=show_todos,
                      mcp=mcp, tracer=tracer)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Atlas ({config.model}). Type /help for commands, 'exit' or Ctrl+D to quit.")
    while True:
        try:
            user_input = input("\nyou> ").strip()
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

        # A copy, not an index: context management may rewrite earlier messages during the turn.
        checkpoint = list(agent.messages)
        try:
            answer = agent.run(user_input)
        except KeyboardInterrupt:
            agent.messages[:] = checkpoint
            print("\n(interrupted)")
            continue
        except Exception as e:
            # Roll back the half-finished turn so the history stays valid for the next message.
            agent.messages[:] = checkpoint
            print(f"Error: {e}", file=sys.stderr)
            continue

        print(f"\natlas> {answer}")


if __name__ == "__main__":
    sys.exit(main())
