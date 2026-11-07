"""Command-line REPL for Atlas (Milestone 1, slash commands from Milestones 5, 5B, 6 and 8, subagents from 7, polished in Milestone 11)."""
import sys

from agent.config import config
from agent.context import estimate_size
from agent.loop import Agent
from agent.tools.mcp_bridge import MCPBridge
from agent.tools.memory_tools import render_entries

EXIT_WORDS = {"exit", "quit"}


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
    entries = agent.memory.entries
    if not entries:
        print("  (no memories saved)")
        return
    print(f"  {len(entries)} memories:")
    for line in render_entries(entries).splitlines():
        print(f"    {line}")


def ingest_knowledge(agent: Agent) -> None:
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
    size, budget = estimate_size(agent.messages), agent.context.budget
    print(f"  {size:,} of {budget:,} characters ({size / budget:.0%}), {len(agent.messages)} messages")
    print(f"  Old tool output cleared: {agent.context.clears} times")
    print(f"  History summarized: {agent.context.summaries} times")


def show_mcp(agent: Agent) -> None:
    rows = agent.mcp.status() if agent.mcp else []
    if not rows:
        print(f"  (no MCP servers configured; add them to {agent.config.mcp_config_path.name})")
        return
    for name, state in rows:
        print(f"    {name}: {state}")


# Slash commands run here, never reach the model. Milestone 11 adds /help, /tools, /cost and the rest.
COMMANDS = {
    "/memory": show_memory,
    "/ingest": ingest_knowledge,
    "/context": show_context,
    "/mcp": show_mcp,
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
    try:
        agent = Agent(config, on_tool_call=show_tool_call, ask_approval=ask_approval, on_todo_change=show_todos, mcp=mcp)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Atlas ({config.model}). Type 'exit' or press Ctrl+D to quit.")
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
                print(f"  Unknown command. Available: {', '.join(COMMANDS)}")
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
