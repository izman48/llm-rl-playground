#!/usr/bin/env python
"""Launch the MCP server over stdio (drive the gym from Claude Desktop / Inspector).

Needs `uv run --extra mcp rl-mcp` (or `pip install -e ".[mcp]"`).
"""
import _bootstrap  # noqa: F401
from playground.mcp.server import main

if __name__ == "__main__":
    main()
