"""Opt-in interop fixture: an MCP server built with the official ``mcp`` SDK.

Used only by ``tests/test_mcp_interop.py`` on Python >= 3.10 with the
``mcp`` package installed. It is a standalone subprocess script (never
imported by the normal test run), so it may safely depend on the SDK that
the light core deliberately avoids.

The SDK renamed FastMCP -> MCPServer in v2; this targets v2 and is skipped
whenever the official package is unavailable.
"""

import sys


def main() -> None:
    from mcp.server.mcpserver import MCPServer  # type: ignore

    server = MCPServer("interop-demo")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return "Hello, %s!" % name

    if len(sys.argv) > 1:
        server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=int(sys.argv[1]),
            stateless_http=True,
            json_response=len(sys.argv) > 2,
        )
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
    sys.exit(0)
