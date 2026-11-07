"""A tiny MCP server for tests/test_mcp_bridge.py, started over stdio like a real one."""
import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("atlas-test")


@server.tool()
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@server.tool()
def get_env(name: str) -> str:
    """Return an environment variable, to check that 'env' from the config reaches the server."""
    return os.environ.get(name, "(unset)")


@server.tool()
def explode() -> str:
    """Always fails."""
    raise ValueError("boom")


if __name__ == "__main__":
    server.run("stdio")
