"""Use Model Context Protocol (MCP) tools with a local model.

This example runs fully offline: it starts a tiny MCP server (defined inline
below) over stdio, discovers the tools it advertises, and registers them on an
``LLM`` backed by the deterministic ``echo`` backend. Swap in a real model and a
real server command to do something useful.

Run it::

    python examples/mcp_tools.py
    python examples/mcp_tools.py --server "npx -y @modelcontextprotocol/server-filesystem /tmp"

A real server needs to be installed (npx/uvx/pip) and may talk to the network;
the demo server below never does.
"""

import argparse
import shlex
import sys

from local_llm_kit import LLM
from local_llm_kit.mcp import MCPServer

# A minimal JSON-RPC MCP server on stdin/stdout. Real servers implement far
# more, but this is enough to show discovery + invocation end to end.
DEMO_SERVER = r"""
import datetime, json, sys

TOOLS = [
    {
        "name": "utc_now",
        "description": "Return the current UTC time (ISO-8601).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "echo_text",
        "description": "Echo the given text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def handle(msg):
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if mid is None:
        return None  # notification
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "demo", "version": "1.0"},
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "utc_now":
            text = datetime.datetime.utcnow().isoformat() + "Z"
        elif name == "echo_text":
            text = "echo: %s" % args.get("text", "")
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "unknown tool"}}
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}]}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "no method"}}


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        message = json.loads(line)
    except ValueError:
        continue
    response = handle(message)
    if response is not None:
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default=None,
        help="Full MCP server command (stdio). Defaults to the built-in demo server.",
    )
    args = parser.parse_args(argv)

    if args.server:
        parts = shlex.split(args.server, posix=(sys.platform != "win32"))
        server = MCPServer("cli-server", command=parts[0], args=parts[1:])
    else:
        server = MCPServer("demo", command=sys.executable, args=["-c", DEMO_SERVER])

    with server:
        tools = server.tools()
        print("Discovered %d MCP tool(s) from %r:" % (len(tools), server.name))
        for tool in tools:
            print("  - %s: %s" % (tool.name, tool.description))

        # Call one directly (no model involved).
        print("\nDirect call -> utc_now: %s" % server.call_tool("utc_now", {}))

        # Register them on an LLM; `chat` will include and (auto-)execute them.
        llm = LLM(model_path="echo", backend="echo")
        try:
            names = llm.add_mcp_server(server)
            print("\nRegistered on LLM: %s" % ", ".join(names))
            response = llm.chat(
                [{"role": "user", "content": "What time is it? Use the utc_now tool."}]
            )
            print("\nModel said:\n%s" % response["choices"][0]["message"]["content"])
            print(
                "\n(With a real tool-calling model the loop above would call the MCP "
                "tool and answer with its result. Try: --server 'ollama' or point "
                "--backend at a local model.)"
            )
        finally:
            llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
