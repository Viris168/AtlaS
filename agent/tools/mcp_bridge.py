"""agent/tools/mcp_bridge.py — Load tools from MCP servers (Milestone 8).

MCP (Model Context Protocol) lets other programs offer tools to Atlas. Each server listed in
mcp_servers.json is started (or, for a URL, connected to) once, its tools are listed, and each
one is wrapped as an ordinary Tool, so the loop, the registry and the guardrails treat it like
any of our own tools.

The MCP client library is async, but our tools are plain functions. So the bridge runs its own
asyncio event loop on a background thread, keeps one long-lived session per server there, and
each tool call hops onto that loop and waits for the answer.

mcp_servers.json:
    {"servers": [
        {"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]},
        {"name": "github", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
         "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}},
        {"name": "docs", "url": "http://localhost:8000/mcp", "trusted": true}
    ]}

"env" values may use ${VAR} to pull secrets from the environment (or .env) instead of the file.
Every MCP tool asks for approval before running, unless its server is marked "trusted": true.
"""
import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent.tools.registry import Tool

# The API only accepts tool names made of these characters, at most 64 of them.
_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_NAME = 64


@dataclass
class ServerConfig:
    name: str
    command: str = ""  # stdio: the program to start...
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""  # ...or Streamable HTTP: the server's endpoint
    trusted: bool = False  # True: run this server's tools without asking first


def load_server_configs(path: Path) -> list[ServerConfig]:
    """Read mcp_servers.json. A missing file means no servers; a broken one is an error worth seeing."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text() or "{}")
    configs, seen = [], set()
    for entry in data.get("servers", []):
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"{path}: every server needs a 'name'")
        if name in seen:
            raise ValueError(f"{path}: two servers are named '{name}'")
        if bool(entry.get("command")) == bool(entry.get("url")):
            raise ValueError(f"{path}: server '{name}' needs either 'command' or 'url', not both")
        seen.add(name)
        configs.append(ServerConfig(
            name=name,
            command=entry.get("command", ""),
            args=[str(a) for a in entry.get("args", [])],
            env={k: os.path.expandvars(str(v)) for k, v in entry.get("env", {}).items()},
            url=entry.get("url", ""),
            trusted=bool(entry.get("trusted", False)),
        ))
    return configs


def tool_name(server: str, tool: str) -> str:
    """mcp__<server>__<tool>: tells the model (and the user) where a tool comes from,
    and keeps two servers' 'search' tools from clashing with each other or with ours."""
    return _NAME_CHARS.sub("_", f"mcp__{server}__{tool}")[:_MAX_NAME]


def render_result(result) -> str:
    """Turn a CallToolResult into the text the model reads."""
    parts = []
    for block in result.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "image":
            parts.append(f"[image: {block.mime_type}, not shown]")
        elif block.type == "audio":
            parts.append(f"[audio: {block.mime_type}, not shown]")
        elif block.type == "resource":
            text = getattr(block.resource, "text", None)
            parts.append(text if text is not None else f"[binary resource: {block.resource.uri}]")
        elif block.type == "resource_link":
            parts.append(f"[resource: {block.uri}]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, indent=2))
    return "\n".join(parts) or "(the tool returned nothing)"


class MCPToolError(Exception):
    """The server ran the tool and reported that it failed."""


@dataclass
class _Server:
    config: ServerConfig
    client: Client | None = None
    tools: list = field(default_factory=list)  # mcp_types.Tool, as the server listed them
    error: str = ""
    ready: threading.Event = field(default_factory=threading.Event)  # connected, or gave up
    done: threading.Event = field(default_factory=threading.Event)  # session closed and process stopped
    stop: asyncio.Event | None = None


class MCPBridge:
    """Owns the background event loop and one session per server. Start once, share with
    every agent (subagents included), close on exit."""

    def __init__(self, configs: list[ServerConfig], log_dir: Path, startup_timeout: float = 30, call_timeout: float = 120):
        self.servers = [_Server(c) for c in configs]
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._tasks: list[asyncio.Future] = []

    @classmethod
    def from_config(cls, config) -> "MCPBridge":
        return cls(load_server_configs(config.mcp_config_path), config.log_dir,
                   config.mcp_startup_timeout, config.mcp_call_timeout)

    def start(self) -> "MCPBridge":
        """Connect to every server, in parallel. A server that fails is skipped and its error
        kept for status(), so one broken server doesn't stop Atlas from starting."""
        if not self.servers or self._loop is not None:
            return self
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-bridge", daemon=True)
        self._thread.start()
        self._tasks = [asyncio.run_coroutine_threadsafe(self._serve(s), self._loop) for s in self.servers]
        for server, task in zip(self.servers, self._tasks):
            if not server.ready.wait(self.startup_timeout):
                server.error = f"no answer within {self.startup_timeout:g}s"
                task.cancel()  # stops the half-started server too
        return self

    async def _serve(self, server: _Server) -> None:
        """Hold one server's session open until close(). The session must be opened and closed
        in the same task, which is why this lives as one coroutine instead of open/close methods."""
        server.stop = asyncio.Event()
        cfg = server.config
        errlog = None
        try:
            if cfg.url:
                target = cfg.url
            else:
                # The server's stderr goes to a log file, so its chatter doesn't mix into the CLI.
                errlog = open(self.log_dir / f"mcp-{cfg.name}.log", "a")
                params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
                target = stdio_client(params, errlog=errlog)
            async with Client(target, read_timeout_seconds=self.call_timeout) as client:
                server.tools = await self._list_tools(client)
                server.client = client
                server.ready.set()
                await server.stop.wait()
        except BaseException as e:  # ExceptionGroup from anyio, CancelledError on shutdown
            if not server.error:
                server.error = _describe(e)
        finally:
            server.client = None
            server.ready.set()
            if errlog:
                errlog.close()
            server.done.set()

    @staticmethod
    async def _list_tools(client: Client) -> list:
        tools, cursor = [], None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                return tools

    def call(self, server_name: str, tool: str, args: dict) -> str:
        """Run one tool on its server and wait for the result (called from the agent's thread)."""
        server = next(s for s in self.servers if s.config.name == server_name)
        if server.client is None:
            raise RuntimeError(f"MCP server '{server_name}' is not connected" + (f": {server.error}" if server.error else ""))
        future = asyncio.run_coroutine_threadsafe(server.client.call_tool(tool, args), self._loop)
        try:
            result = future.result(self.call_timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(f"'{tool}' on MCP server '{server_name}' took longer than {self.call_timeout:g}s")
        text = render_result(result)
        if result.is_error:
            raise MCPToolError(text)
        return text

    def tools(self) -> list[Tool]:
        """Every connected server's tools, wrapped for the ToolRegistry."""
        wrapped, taken = [], set()
        for server in self.servers:
            if server.client is None:
                continue
            for t in server.tools:
                name = tool_name(server.config.name, t.name)
                if name in taken:
                    continue  # two names that only differ after cleaning or past 64 chars: keep the first
                taken.add(name)
                wrapped.append(self._wrap(server.config, t, name))
        return wrapped

    def _wrap(self, cfg: ServerConfig, mcp_tool, name: str) -> Tool:
        def fn(**args) -> str:
            return self.call(cfg.name, mcp_tool.name, args)

        description = (mcp_tool.description or mcp_tool.title or mcp_tool.name).strip()
        return Tool(
            name=name,
            description=f"[from MCP server '{cfg.name}'] {description}",
            input_schema=mcp_tool.input_schema or {"type": "object", "properties": {}},
            fn=fn,
            # Tool annotations like readOnlyHint are the server's own claims, so they don't count:
            # only the user can vouch for a server, by marking it trusted.
            risky=not cfg.trusted,
        )

    def status(self) -> list[tuple[str, str]]:
        """(server name, one-line state) for the /mcp command."""
        rows = []
        for s in self.servers:
            if s.client is not None:
                rows.append((s.config.name, f"connected, {len(s.tools)} tools" + ("" if s.config.trusted else ", approval needed")))
            else:
                rows.append((s.config.name, f"failed: {s.error or 'not started'}"))
        return rows

    def close(self) -> None:
        """Stop every session (which also stops the server processes) and the background loop."""
        if self._loop is None:
            return
        for server in self.servers:
            if server.stop is not None:
                self._loop.call_soon_threadsafe(server.stop.set)
        # Wait on `done`, not the task's future: a cancelled future reports back at once, while
        # the session is still stopping the server process. Stopping the loop then would orphan it.
        for server, task in zip(self.servers, self._tasks):
            if not server.done.wait(10):
                task.cancel()
                server.done.wait(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
        self._loop.close()
        self._loop = None


def _describe(error: BaseException) -> str:
    """anyio wraps failures in ExceptionGroups; dig out the first real cause for a readable message."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
