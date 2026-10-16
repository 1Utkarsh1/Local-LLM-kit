"""A tiny Model Context Protocol server over stdio, for tests only.

Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout. Stdlib only; it is
launched as a subprocess by ``tests/test_mcp.py``.

Modes (argv):
    --paginate   paginate ``tools/list`` (2 tools per page) with nextCursor
    --sleep      never answer ``tools/list`` (used to test timeouts)
"""

import json
import sys
import time

TOOLS = [
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "boom",
        "description": "Always reports an error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "weird/name",
        "description": "Tool whose name is not a valid OpenAI function name.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _respond(mid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": mid}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _handle(msg, paginate, sleep):
    method = msg.get("method")
    mid = msg.get("id")
    params = msg.get("params") or {}
    if mid is None:
        return  # notification (e.g. notifications/initialized)
    if method == "initialize":
        _respond(
            mid,
            result={
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-stdio", "version": "1.0"},
            },
        )
    elif method == "tools/list":
        if sleep:
            time.sleep(3600)
            return
        if not paginate:
            _respond(mid, result={"tools": TOOLS})
            return
        cursor = params.get("cursor")
        if cursor in (None, "0"):
            _respond(mid, result={"tools": TOOLS[:2], "nextCursor": "1"})
        else:
            _respond(mid, result={"tools": TOOLS[2:]})
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "add":
            _respond(
                mid,
                result={
                    "content": [
                        {"type": "text", "text": str(int(args.get("a", 0)) + int(args.get("b", 0)))}
                    ]
                },
            )
        elif name == "boom":
            _respond(mid, result={"content": [{"type": "text", "text": "kaboom"}], "isError": True})
        elif name == "weird/name":
            _respond(mid, result={"content": [{"type": "text", "text": "weird ok"}]})
        else:
            _respond(mid, error={"code": -32602, "message": "unknown tool: %s" % name})
    else:
        _respond(mid, error={"code": -32601, "message": "method not found: %s" % method})


def main(argv):
    paginate = "--paginate" in argv
    sleep = "--sleep" in argv
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        _handle(msg, paginate, sleep)


if __name__ == "__main__":
    main(sys.argv[1:])
