# MCP Documentation

The Model Context Protocol (MCP) connects AI assistants to external tools and data.

## Installation

Install the SDK with pip:

```bash
pip install mcp-sdk
## this line is a shell comment, not a heading
```

Python 3.10 or newer is required.

## Server Setup

The server step is described here in detail so that readers understand sentence 0. The server step is described here in detail so that readers understand sentence 1. The server step is described here in detail so that readers understand sentence 2. The server step is described here in detail so that readers understand sentence 3. The server step is described here in detail so that readers understand sentence 4. The server step is described here in detail so that readers understand sentence 5. The server step is described here in detail so that readers understand sentence 6. The server step is described here in detail so that readers understand sentence 7.

### Transports

MCP servers support two transports: stdio for local processes and streamable HTTP for remote servers.
The transport step is described here in detail so that readers understand sentence 0. The transport step is described here in detail so that readers understand sentence 1. The transport step is described here in detail so that readers understand sentence 2. The transport step is described here in detail so that readers understand sentence 3. The transport step is described here in detail so that readers understand sentence 4. The transport step is described here in detail so that readers understand sentence 5. The transport step is described here in detail so that readers understand sentence 6. The transport step is described here in detail so that readers understand sentence 7. The transport step is described here in detail so that readers understand sentence 8. The transport step is described here in detail so that readers understand sentence 9.

### Authentication

Remote servers authenticate clients with OAuth 2.1 bearer tokens. Set `MCP_AUTH_TOKEN` in the environment.
The auth step is described here in detail so that readers understand sentence 0. The auth step is described here in detail so that readers understand sentence 1. The auth step is described here in detail so that readers understand sentence 2. The auth step is described here in detail so that readers understand sentence 3. The auth step is described here in detail so that readers understand sentence 4. The auth step is described here in detail so that readers understand sentence 5. The auth step is described here in detail so that readers understand sentence 6. The auth step is described here in detail so that readers understand sentence 7. The auth step is described here in detail so that readers understand sentence 8. The auth step is described here in detail so that readers understand sentence 9.

### Tool Registration

Register a tool with the `@server.tool()` decorator. Each tool needs a name, description and JSON input schema.
The registration step is described here in detail so that readers understand sentence 0. The registration step is described here in detail so that readers understand sentence 1. The registration step is described here in detail so that readers understand sentence 2. The registration step is described here in detail so that readers understand sentence 3. The registration step is described here in detail so that readers understand sentence 4. The registration step is described here in detail so that readers understand sentence 5. The registration step is described here in detail so that readers understand sentence 6. The registration step is described here in detail so that readers understand sentence 7. The registration step is described here in detail so that readers understand sentence 8. The registration step is described here in detail so that readers understand sentence 9.

## Troubleshooting

If the client reports error -32601 (Method not found), the server does not implement that method. Check the tool name spelling.
