"""The Agent class and its run() loop (Milestones 1-8)."""
from agent.config import Config, config as default_config
from agent.context import ContextManager, llm_summarizer
from agent.guardrails import Approver, AskFn, TokenBudget
from agent.llm import LLM
from agent.memory import Memory
from agent.prompts import build_system_prompt
from agent.rag.ingest import open_store
from agent.rag.retriever import Retriever
from agent.tools.files import file_tools
from agent.tools.knowledge import knowledge_tools, render_outline
from agent.tools.mcp_bridge import MCPBridge
from agent.tools.memory_tools import memory_tools
from agent.tools.registry import ToolRegistry
from agent.tools.shell import shell_tool
from agent.tools.subagent import delegate_tool
from agent.tools.todo import TodoList, todo_tools
from agent.tools.web import web_search_schema


def build_registry(config: Config, todos: TodoList, memory: Memory, mcp: MCPBridge | None = None) -> ToolRegistry:
    """Every tool that runs on our side. Later milestones register more tools here."""
    registry = ToolRegistry(max_output_chars=config.max_tool_output_chars)
    for tool in file_tools(config.workspace):
        registry.register(tool)
    registry.register(shell_tool(config.workspace, config.shell_timeout))
    for tool in todo_tools(todos):
        registry.register(tool)
    for tool in memory_tools(memory):
        registry.register(tool)
    # Opened on the first search, not here: Chroma and the embedding model are slow to load.
    for tool in knowledge_tools(lambda: Retriever.from_config(config, open_store(config)), config.rag_top_k, config.rag_passage_chars):
        registry.register(tool)
    if mcp is not None:
        for tool in mcp.tools():
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
        depth: int = 0,
        subagent_llm: LLM | None = None,
        mcp: MCPBridge | None = None,
    ):
        self.config = config
        self.llm = llm or LLM(config)
        # How deep this agent is: 0 for the one the user talks to, 1 for its subagents, and so on.
        self.depth = depth
        self._subagent_llm = subagent_llm  # made on first delegate_task if not given (tests pass a fake)
        # Lives on the instance, not in the module, so each subagent (M7) gets its own plan.
        self.todos = TodoList(on_change=on_todo_change)  # on_todo_change(todo_list) after every todo_write
        # Unlike the todo list, memory is one file on disk shared by every agent and session.
        self.memory = Memory(config.memory_path)
        # Started by the caller (the CLI), not here: it launches server processes that must be shut down.
        self.mcp = mcp
        self.registry = registry or build_registry(config, self.todos, self.memory, mcp)
        # Only offered while there's room to go one level deeper, so subagents can't spawn
        # subagents forever. With the default max_subagent_depth=1, only the main agent gets it.
        if registry is None and self.can_delegate:
            self.registry.register(delegate_tool(self.delegate))
        self.messages: list = []
        self.total_tokens = 0
        self.on_tool_call = on_tool_call  # called as on_tool_call(name, args, result)
        self.approver = Approver(config.auto_approve, ask_approval)
        self.context = ContextManager(config.context_char_budget, config.keep_recent_messages, self._summarize)
        self._summarizer = None
        self._budget: TokenBudget | None = None  # the current run's budget, so delegate() can add to it

    @property
    def can_delegate(self) -> bool:
        return self.depth < self.config.max_subagent_depth

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        system = build_system_prompt(
            self.config, self._recalled(user_input), self._knowledge_outline(),
            can_delegate=self.can_delegate, is_subagent=self.depth > 0,
        )
        tools = self.tool_schemas()
        budget = self._budget = TokenBudget(self.config.max_total_tokens)

        for _ in range(self.config.max_steps):
            # Checked before each call, so a turn that just ran tools still gets its results recorded.
            if budget.exceeded:
                answer = f"Stopped: token budget used up ({budget.used:,} of {budget.limit:,} tokens) before finishing the task."
                break
            # Before every call: a turn that ran many tools can outgrow the budget partway through.
            self.context.manage(self.messages)
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

    def delegate(self, task: str) -> str:
        """Run `task` in a brand-new child Agent and return its final answer (the delegate_task tool)."""
        if self._subagent_llm is None:
            self._subagent_llm = LLM(self.config, model=self.config.subagent_model)  # cheaper, faster model
        child = Agent(
            self.config,
            llm=self._subagent_llm,
            on_tool_call=self._child_tool_call if self.on_tool_call else None,
            depth=self.depth + 1,
            subagent_llm=self._subagent_llm,
            mcp=self.mcp,  # same server sessions, not a second copy of every server
        )
        # Share the approver, so the user still approves the child's risky tools,
        # and an "always" they gave the main agent also covers the child.
        child.approver = self.approver
        answer = child.run(task)

        # The child's tokens are real spending: count them in this agent's totals and this run's budget.
        self.total_tokens += child.total_tokens
        if self._budget is not None:
            self._budget.used += child.total_tokens
        return answer

    def _child_tool_call(self, name, args, result) -> None:
        """Pass the child's tool calls up to our listener, marked so the CLI can tell them apart."""
        self.on_tool_call(f"subagent > {name}", args, result)

    def _summarize(self, transcript: str) -> str:
        """Made on first use: most sessions never need it, and tests never build a second client."""
        if self._summarizer is None:
            self._summarizer = llm_summarizer(LLM(self.config, model=self.config.subagent_model))
        return self._summarizer(transcript)

    def _recalled(self, user_input: str) -> list[dict]:
        """Memories that match this message, then the newest few, without repeats."""
        matches = self.memory.search(user_input, self.config.memory_recall_in_prompt)
        recent = self.memory.recent(self.config.memory_recent_in_prompt)
        seen = {e["id"] for e in matches}
        return matches + [e for e in recent if e["id"] not in seen]

    def _knowledge_outline(self) -> str:
        """What's in the knowledge base, so the model knows which questions to search for.
        Read fresh each turn, so it follows /ingest. Nothing ever ingested: skip it without loading Chroma."""
        index_dir = self.config.rag_index_dir
        if not index_dir.is_dir() or not any(index_dir.iterdir()):
            return ""
        try:
            return render_outline(open_store(self.config).outline())
        except Exception:
            return ""  # a broken index shouldn't stop the turn; search_knowledge will report the error

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
