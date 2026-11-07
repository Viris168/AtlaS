"""Command-line REPL for Atlas (Milestone 1, polished in Milestone 11)."""
import sys

from agent.config import config
from agent.loop import Agent

EXIT_WORDS = {"exit", "quit"}


def show_tool_call(name: str, args: dict, result: str) -> None:
    if name.startswith("todo_"):
        return  # show_todos already prints the plan in a readable form
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


def main() -> int:
    try:
        agent = Agent(config, on_tool_call=show_tool_call, ask_approval=ask_approval, on_todo_change=show_todos)
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

        checkpoint = len(agent.messages)
        try:
            answer = agent.run(user_input)
        except KeyboardInterrupt:
            del agent.messages[checkpoint:]
            print("\n(interrupted)")
            continue
        except Exception as e:
            # Roll back the half-finished turn so the history stays valid for the next message.
            del agent.messages[checkpoint:]
            print(f"Error: {e}", file=sys.stderr)
            continue

        print(f"\natlas> {answer}")


if __name__ == "__main__":
    sys.exit(main())
