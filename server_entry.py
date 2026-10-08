"""Server entry point used by demo_client.py (stdio MCP server)."""

import anyio

from pg_mcp.server import mcp

if __name__ == "__main__":
    anyio.run(mcp.run_stdio_async)
