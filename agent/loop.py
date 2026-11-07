"""The Agent class and its run() loop (Milestones 1-9; reset(), on_text and on_tool_start from 11)."""
import time

from agent.config import Config, config as default_config
from agent.context import ContextManager, llm_summarizer
from agent.guardrails import Approver, AskFn, TokenBudget
from agent.llm import LLM
from agent.logging_utils import NullTracer, Tracer, Usage, new_id, preview
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
        tracer: Tracer | NullTracer | None = None,
        on_text=None,
        on_tool_start=None,
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
        self.usage = Usage()  # tokens per model, subagents and summaries included, for /cost (M9)
        # Shared with subagents, so one session is one trace file. The default records nothing.
        self.tracer = tracer or NullTracer()
        self.run_id: str | None = None  # the current run, so every event can say which run it belongs to
        self.parent_run_id: str | None = None  # set on a subagent: the parent run that delegated to it
        self.on_tool_call = on_tool_call  # called as on_tool_call(name, args, result)
        self.on_tool_start = on_tool_start  # called as on_tool_start(name, args) once a tool is cleared to run
        # on_text(chunk) streams the model's text as it arrives (M11). Only this agent's own turns stream:
        # subagents and summaries work out of sight, and only their results matter.
        self.on_text = on_text
        self.approver = Approver(config.auto_approve, ask_approval)
        self.context = ContextManager(config.context_char_budget, config.keep_recent_messages, self._summarize)
        self._summarizer = None
        self._budget: TokenBudget | None = None  # the current run's budget, so delegate() can add to it

    @property
    def can_delegate(self) -> bool:
        return self.depth < self.config.max_subagent_depth

    def reset(self) -> None:
        """Start a fresh conversation (/reset, M11): history, plan and "always" approvals go.
        Memory stays (it's meant to outlive conversations), and so does usage, so /cost still covers the session."""
        self.messages.clear()
        self.todos.items = []
        self.approver.always.clear()
        self.context.clears = self.context.summaries = 0

    def run(self, user_input: str) -> str:
        run_id = self.run_id = new_id()
        if self.depth == 0:
            self.tracer.event("user_input", run_id, text=user_input)
        else:
            self.tracer.event("subagent_start", run_id, parent_id=self.parent_run_id, task=user_input,
                              model=getattr(self.llm, "model", None))
        started = time.monotonic()
        tokens_before = (self.usage.input_tokens, self.usage.cache_read_tokens, self.usage.output_tokens)
        try:
            answer, stop, steps = self._run(user_input)
        except Exception as e:
            # Logged before it propagates: a crash is when the trace is needed most.
            self.tracer.event("error", run_id, error=f"{type(e).__name__}: {e}")
            raise
        self.tracer.event(
            "final_answer" if self.depth == 0 else "subagent_end", run_id,
            **({"parent_id": self.parent_run_id} if self.depth else {}),
            text=answer, stop=stop, steps=steps, duration=round(time.monotonic() - started, 3),
            input_tokens=self.usage.input_tokens - tokens_before[0],
            cache_read_tokens=self.usage.cache_read_tokens - tokens_before[1],
            output_tokens=self.usage.output_tokens - tokens_before[2],
        )
        return answer

    def _run(self, user_input: str) -> tuple[str, str, int]:
        """The loop itself. Returns (answer, why it stopped, model calls made)."""
        self.messages.append({"role": "user", "content": user_input})
        system = build_system_prompt(
            self.config, self._recalled(user_input), self._knowledge_outline(),
            can_delegate=self.can_delegate, is_subagent=self.depth > 0,
        )
        tools = self.tool_schemas()
        budget = self._budget = TokenBudget(self.config.max_total_tokens)

        steps = 0
        for _ in range(self.config.max_steps):
            # Checked before each call, so a turn that just ran tools still gets its results recorded.
            if budget.exceeded:
                answer = f"Stopped: token budget used up ({budget.used:,} of {budget.limit:,} tokens) before finishing the task."
                stop = "token_budget"
                break
            # Before every call: a turn that ran many tools can outgrow the budget partway through.
            trim = self.context.manage(self.messages)
            if trim:
                self.tracer.event("context_trim", self.run_id, **trim)
            response = self._complete(self.llm, system, self.messages, tools)
            steps += 1

            # Keep the full block list: the API needs the tool_use blocks to match results.
            self.messages.append({"role": "assistant", "content": response.content})

            # A long server-side web search can pause mid-turn; sending the history back resumes it.
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                return self._text(response.content), "answer", steps

            results = [self._run_tool(block) for block in response.content if block.type == "tool_use"]
            self.messages.append({"role": "user", "content": results})
        else:
            answer = f"Stopped at step limit ({self.config.max_steps} steps) before finishing the task."
            stop = "step_limit"

        # Close the turn so the next user message doesn't follow a tool_result directly.
        self.messages.append({"role": "assistant", "content": answer})
        return answer, stop, steps

    def _complete(self, llm, system: str, messages: list, tools: list | None = None, purpose: str = "agent"):
        """Every model call goes through here, so each one is timed, counted and traced the same way."""
        started = time.monotonic()
        if purpose == "agent" and self.on_text:
            response = llm.complete(system, messages, tools, on_text=self.on_text)
        else:
            response = llm.complete(system, messages, tools)
        latency = time.monotonic() - started

        usage, model = response.usage, getattr(llm, "model", "unknown")
        # Cache hits come back apart from input_tokens (and cost much less), so they're counted apart too.
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usage.add(model, usage.input_tokens, usage.output_tokens, cache_read)
        self.total_tokens += usage.input_tokens + usage.output_tokens
        if self._budget is not None:
            self._budget.add(usage)
        self.tracer.event(
            "llm_call", self.run_id, model=model, purpose=purpose,
            input_tokens=usage.input_tokens, cache_read_tokens=cache_read, output_tokens=usage.output_tokens,
            latency=round(latency, 3), stop_reason=response.stop_reason,
            text=preview(self._text(response.content)),
            tool_calls=[b.name for b in response.content if b.type in ("tool_use", "server_tool_use")],
        )
        return response

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
            tracer=self.tracer,  # same trace file; its events carry parent_id so it nests under this run
        )
        child.parent_run_id = self.run_id
        # Share the approver, so the user still approves the child's risky tools,
        # and an "always" they gave the main agent also covers the child.
        child.approver = self.approver
        answer = child.run(task)

        # The child's tokens are real spending: count them in this agent's totals and this run's budget.
        self.total_tokens += child.total_tokens
        self.usage.merge(child.usage)
        if self._budget is not None:
            self._budget.used += child.total_tokens
        return answer

    def _child_tool_call(self, name, args, result) -> None:
        """Pass the child's tool calls up to our listener, marked so the CLI can tell them apart."""
        self.on_tool_call(f"subagent > {name}", args, result)

    def _summarize(self, transcript: str) -> str:
        """Made on first use: most sessions never need it, and tests never build a second client."""
        if self._summarizer is None:
            self._summarizer = llm_summarizer(_Metered(self, LLM(self.config, model=self.config.subagent_model), "summary"))
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
        allowed, approval = True, "not_needed"
        if self.registry.is_risky(block.name):
            allowed, approval = self.approver.decide(block.name, block.input)
        started = time.monotonic()  # after the approval prompt, so the user's thinking time isn't counted
        if not allowed:
            # Tell the model, so it can try something else or ask the user, instead of retrying blindly.
            content, is_error = f"Denied: the user did not approve running '{block.name}'.", True
        else:
            if self.on_tool_start:
                self.on_tool_start(block.name, block.input)
            content, is_error = self.registry.run(block.name, block.input)
        self.tracer.event(
            "tool_call", self.run_id, name=block.name, args=block.input, approval=approval,
            duration=round(time.monotonic() - started, 3), error=is_error, output=preview(content),
        )

        if self.on_tool_call:
            self.on_tool_call(block.name, block.input, content)
        return {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}

    @staticmethod
    def _text(content) -> str:
        return "\n".join(block.text for block in content if block.type == "text").strip()


class _Metered:
    """Wraps an LLM so its calls go through Agent._complete: counted in /cost and the budget, and traced.
    Used for the context summarizer, which otherwise calls the model behind the agent's back."""

    def __init__(self, agent: Agent, llm: LLM, purpose: str):
        self.agent, self.llm, self.purpose = agent, llm, purpose

    def complete(self, system: str, messages: list, tools: list | None = None):
        return self.agent._complete(self.llm, system, messages, tools, purpose=self.purpose)
