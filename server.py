"""server.py — FastAPI web API with SSE and approvals over HTTP (Milestone 11).

Run it with `python server.py` (AGENT_HOST / AGENT_PORT) or `uvicorn server:app`.

Each session is one Agent with its own history, plan and trace file, the same as one CLI run.
A message streams back as Server-Sent Events while the agent works:
  tool_call         {"name", "args", "result"}       a tool ran (result is a preview)
  todos             {"items"}                         the plan changed
  approval_request  {"approval_id", "name", "args"}   a risky tool is waiting: answer it with
                                                      POST /sessions/{id}/approvals/{approval_id}
  answer            {"text"}                          the final answer; the stream ends after it
  error             {"error"}                         the turn failed and was rolled back; the stream ends

Endpoints (each needs "Authorization: Bearer <AGENT_API_TOKEN>" when that's set):
  GET    /health
  GET    /tools
  POST   /sessions                                  -> {"session_id"}
  GET    /sessions/{id}                             -> messages, context, tokens and cost
  DELETE /sessions/{id}
  POST   /sessions/{id}/reset
  POST   /sessions/{id}/messages                    {"message": "..."} -> text/event-stream
  POST   /sessions/{id}/approvals/{approval_id}     {"decision": "y" | "n" | "a"}
"""
import asyncio
import json
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.config import Config, config as default_config
from agent.context import estimate_size
from agent.logging_utils import Tracer, cost_usd, new_id, preview
from agent.loop import Agent
from agent.tools.mcp_bridge import MCPBridge

# make_agent(config, tracer, mcp) -> Agent. Tests swap in one that uses a scripted FakeLLM.
AgentFactory = Callable[[Config, Tracer, MCPBridge | None], Agent]


def default_agent_factory(config: Config, tracer: Tracer, mcp: MCPBridge | None) -> Agent:
    return Agent(config, mcp=mcp, tracer=tracer)


class MessageIn(BaseModel):
    message: str = Field(min_length=1)


class DecisionIn(BaseModel):
    decision: str = Field(pattern=r"^(?i:y|yes|n|no|a|always)$")


@dataclass
class PendingApproval:
    name: str
    args: dict
    answered: threading.Event = field(default_factory=threading.Event)
    decision: str = "n"


class Session:
    """One Agent plus the plumbing between its worker thread and the HTTP side.

    Agent.run is blocking, so each turn runs in a thread. Its callbacks (tool calls, plan changes,
    approval requests) go through emit() onto the asyncio queue the open SSE response is reading.
    """

    def __init__(self, agent: Agent, tracer: Tracer, approval_timeout: float):
        self.id = tracer.session_id
        self.agent, self.tracer = agent, tracer
        self.approval_timeout = approval_timeout
        self.busy = threading.Lock()  # one turn at a time: the history can't take two at once
        self.pending: dict[str, PendingApproval] = {}
        self._emit: Callable[[str, dict], None] | None = None  # set while a turn is streaming
        agent.on_tool_call = self._on_tool_call
        agent.todos.on_change = self._on_todos
        agent.approver.ask = self._ask

    def emit(self, event: str, data: dict) -> None:
        if self._emit:
            self._emit(event, data)

    def _on_tool_call(self, name: str, args: dict, result: str) -> None:
        self.emit("tool_call", {"name": name, "args": args, "result": preview(result)})

    def _on_todos(self, todos) -> None:
        self.emit("todos", {"items": todos.items})

    def _ask(self, name: str, args: dict) -> str:
        """Called on the worker thread: announce the request, then block until it's answered.
        No answer in time (the client went away, say) counts as "no", so the safe default holds."""
        approval_id = new_id()
        pending = self.pending[approval_id] = PendingApproval(name, args)
        self.emit("approval_request", {"approval_id": approval_id, "name": name, "args": args})
        try:
            pending.answered.wait(self.approval_timeout)
            return pending.decision
        finally:
            del self.pending[approval_id]

    def answer(self, approval_id: str, decision: str) -> bool:
        pending = self.pending.get(approval_id)
        if pending is None:
            return False
        pending.decision = decision
        pending.answered.set()
        return True

    def run_turn(self, message: str) -> None:
        """The worker thread's job: run one turn, emit its answer or error, roll back on failure."""
        # A copy, not an index: context management may rewrite earlier messages during the turn.
        checkpoint = list(self.agent.messages)
        try:
            self.emit("answer", {"text": self.agent.run(message)})
        except Exception as e:
            # Roll back the half-finished turn so the history stays valid for the next message.
            self.agent.messages[:] = checkpoint
            self.emit("error", {"error": f"{type(e).__name__}: {e}"})

    def summary(self) -> dict:
        agent, usage = self.agent, self.agent.usage
        return {
            "session_id": self.id,
            "busy": self.busy.locked(),
            "messages": len(agent.messages),
            "context_chars": estimate_size(agent.messages),
            "context_budget": agent.context.budget,
            "todos": agent.todos.items,
            "usage": {"input_tokens": usage.input_tokens, "cache_read_tokens": usage.cache_read_tokens,
                      "output_tokens": usage.output_tokens, "by_model": usage.by_model},
            "cost_usd": round(cost_usd(usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, agent.config), 6),
            "pending_approvals": [{"approval_id": i, "name": p.name, "args": p.args} for i, p in self.pending.items()],
        }

    def close(self) -> None:
        for pending in list(self.pending.values()):
            pending.answered.set()  # wake a waiting turn; the default decision is "n"
        self.tracer.close()


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def create_app(config: Config = default_config, make_agent: AgentFactory = default_agent_factory,
               start_mcp: bool = True) -> FastAPI:
    sessions: dict[str, Session] = {}
    state: dict = {"mcp": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_mcp:
            try:
                state["mcp"] = MCPBridge.from_config(config)
                state["mcp"].start()
            except ValueError as e:  # a broken mcp_servers.json: run without MCP, like the CLI
                print(f"Warning: MCP disabled: {e}")
        yield
        for session in list(sessions.values()):
            session.close()
        sessions.clear()
        if state["mcp"]:
            state["mcp"].close()

    def check_token(request: Request) -> None:
        if not config.api_token:
            return
        header = request.headers.get("authorization", "")
        # compare_digest: a plain == leaks how many leading characters matched through its timing.
        if not secrets.compare_digest(header.encode(), f"Bearer {config.api_token}".encode()):
            raise HTTPException(401, "Missing or wrong bearer token", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="Atlas", lifespan=lifespan, dependencies=[Depends(check_token)])

    def get_session(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, f"No session '{session_id}'")
        return session

    @app.get("/health")
    def health():
        mcp = state["mcp"]
        return {"status": "ok", "model": config.model, "sessions": len(sessions),
                "mcp": dict(mcp.status()) if mcp else {}}

    @app.get("/tools")
    def tools():
        # A throwaway agent, without a trace, just to read the tool list the sessions get.
        try:
            agent = make_agent(config, None, state["mcp"])
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        return [{"name": t["name"], "description": t.get("description", ""), "risky": agent.registry.is_risky(t["name"])}
                for t in agent.tool_schemas()]

    @app.post("/sessions", status_code=201)
    def create_session():
        if len(sessions) >= config.max_sessions:
            raise HTTPException(429, f"Session limit reached ({config.max_sessions}); delete one first")
        tracer = Tracer(config.log_dir, secrets=(config.api_key, config.api_token))
        try:
            agent = make_agent(config, tracer, state["mcp"])
        except RuntimeError as e:  # e.g. no API key configured
            tracer.close()
            raise HTTPException(500, str(e))
        session = Session(agent, tracer, config.approval_timeout)
        sessions[session.id] = session
        return {"session_id": session.id}

    @app.get("/sessions/{session_id}")
    def session_info(session_id: str):
        return get_session(session_id).summary()

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        session = get_session(session_id)
        if session.busy.locked():
            raise HTTPException(409, "A turn is still running")
        sessions.pop(session_id).close()

    @app.post("/sessions/{session_id}/reset")
    def reset_session(session_id: str):
        session = get_session(session_id)
        if not session.busy.acquire(blocking=False):
            raise HTTPException(409, "A turn is still running")
        try:
            session.agent.reset()
        finally:
            session.busy.release()
        return {"status": "reset"}

    @app.post("/sessions/{session_id}/approvals/{approval_id}")
    def answer_approval(session_id: str, approval_id: str, body: DecisionIn):
        if not get_session(session_id).answer(approval_id, body.decision.lower()):
            raise HTTPException(404, f"No pending approval '{approval_id}'")
        return {"status": "answered"}

    @app.post("/sessions/{session_id}/messages")
    async def send_message(session_id: str, body: MessageIn):
        session = get_session(session_id)
        # Taken here, not in the stream, so a second message gets a clean 409 instead of a broken stream.
        if not session.busy.acquire(blocking=False):
            raise HTTPException(409, "A turn is already running in this session")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        session._emit = lambda event, data: loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def work():
            try:
                session.run_turn(body.message)
            finally:
                session._emit = None
                session.busy.release()
                loop.call_soon_threadsafe(queue.put_nowait, None)  # end of stream

        threading.Thread(target=work, name=f"atlas-{session.id}", daemon=True).start()

        async def stream():
            # If the client disconnects, the turn still finishes in its thread (so the history stays
            # whole); its approval requests just time out as "no".
            while (item := await queue.get()) is not None:
                yield sse(*item)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=default_config.server_host, port=default_config.server_port)
