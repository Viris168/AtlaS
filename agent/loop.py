"""The Agent class and its run() loop (Milestones 1-5)."""
from agent.config import Config, config as default_config
from agent.guardrails import Approver, AskFn, TokenBudget
from agent.llm import LLM
from agent.memory import Memory
from agent.prompts import build_system_prompt
from agent.tools.files import file_tools
from agent.tools.memory_tools import memory_tools
from agent.tools.registry import ToolRegistry
from agent.tools.shell import shell_tool
from agent.tools.todo import TodoList, todo_tools
from agent.tools.web import web_search_schema


def build_registry(config: Config, todos: TodoList, memory: Memory) -> ToolRegistry:
    """Every tool that runs on our side. Later milestones register more tools here."""
    registry = ToolRegistry(max_output_chars=config.max_tool_output_chars)
    for tool in file_tools(config.workspace):
        registry.register(tool)
    registry.register(shell_tool(config.workspace, config.shell_timeout))
    for tool in todo_tools(todos):
        registry.register(tool)
    for tool in memory_tools(memory):
        registry.register(tool)
    return registry


class Agent:
    def __init__(
        self,
        config: Config = default_config,
        llm: LLM | None = None,
        registry: ToolRegistry | None = None,
        on_tool_call=None,
        ask_approval: AskFn | None = None,
        on_todo_change=None,
    ):
        self.config = config
        self.llm = llm or LLM(config)
        # Lives on the instance, not in the module, so each subagent (M7) gets its own plan.
        self.todos = TodoList(on_change=on_todo_change)  # on_todo_change(todo_list) after every todo_write
        # Unlike the todo list, memory is one file on disk shared by every agent and session.
        self.memory = Memory(config.memory_path)
        self.registry = registry or build_registry(config, self.todos, self.memory)
        self.messages: list = []
        self.total_tokens = 0
        self.on_tool_call = on_tool_call  # called as on_tool_call(name, args, result)
        self.approver = Approver(config.auto_approve, ask_approval)

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        system = build_system_prompt(self.config, self._recalled(user_input))
        tools = self.tool_schemas()
        budget = TokenBudget(self.config.max_total_tokens)

        for _ in range(self.config.max_steps):
            # Checked before each call, so a turn that just ran tools still gets its results recorded.
            if budget.exceeded:
                answer = f"Stopped: token budget used up ({budget.used:,} of {budget.limit:,} tokens) before finishing the task."
                break
            response = self.llm.complete(system, self.messages, tools)
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            budget.add(response.usage)

            # Keep the full block list: the API needs the tool_use blocks to match results.
            self.messages.append({"role": "assistant", "content": response.content})

            # A long server-side web search can pause mid-turn; sending the history back resumes it.
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                return self._text(response.content)

            results = [self._run_tool(block) for block in response.content if block.type == "tool_use"]
            self.messages.append({"role": "user", "content": results})
        else:
            answer = f"Stopped at step limit ({self.config.max_steps} steps) before finishing the task."

        # Close the turn so the next user message doesn't follow a tool_result directly.
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def _recalled(self, user_input: str) -> list[dict]:
        """Memories that match this message, then the newest few, without repeats."""
        matches = self.memory.search(user_input, self.config.memory_recall_in_prompt)
        recent = self.memory.recent(self.config.memory_recent_in_prompt)
        seen = {e["id"] for e in matches}
        return matches + [e for e in recent if e["id"] not in seen]

    def tool_schemas(self) -> list:
        tools = self.registry.schemas()
        if self.config.enable_web_search:
            tools.append(web_search_schema())  # runs on the provider's side, so not in the registry
        return tools

    def _run_tool(self, block) -> dict:
        if self.registry.is_risky(block.name) and not self.approver.approve(block.name, block.input):
            # Tell the model, so it can try something else or ask the user, instead of retrying blindly.
            content, is_error = f"Denied: the user did not approve running '{block.name}'.", True
        else:
            content, is_error = self.registry.run(block.name, block.input)

        if self.on_tool_call:
            self.on_tool_call(block.name, block.input, content)
        return {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}

    @staticmethod
    def _text(content) -> str:
        return "\n".join(block.text for block in content if block.type == "text").strip()
