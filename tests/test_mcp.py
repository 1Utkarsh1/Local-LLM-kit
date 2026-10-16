"""Offline tests for the Model Context Protocol (MCP) client.

Covers schema conversion, the stdio transport (subprocess JSON-RPC),
the Streamable HTTP transport (JSON + SSE), error/timeout/cleanup paths,
and end-to-end registration + tool execution through ``LLM``.

Everything here is offline: the stdio server is a local Python script and
the HTTP server is an in-process ``http.server``. No network, no models,
no optional dependencies. Python 3.9 compatible.

Run: python -m pytest tests/test_mcp.py -q
"""

import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm_kit import LLM
from local_llm_kit.backends.base import BaseBackend
from local_llm_kit.mcp import (
    MCPError,
    MCPProtocolError,
    MCPServer,
    MCPTool,
    MCPToolError,
    mcp_tools_to_openai,
    sanitize_tool_name,
)

FAKE_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fake_mcp_server.py")


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stdio_server(*extra, name="fake", timeout=10.0):
    return MCPServer(
        name, command=sys.executable, args=[FAKE_SERVER] + list(extra), timeout=timeout
    )


# ---------------------------------------------------------------------------
# Fake HTTP MCP server
# ---------------------------------------------------------------------------
def _http_rpc(msg):
    """Mirror of the stdio fake server, returning a response dict or None."""
    method = msg.get("method")
    mid = msg.get("id")
    params = msg.get("params") or {}
    if mid is None:
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-http", "version": "1.0"},
            },
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            },
        }
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {"content": [{"type": "text", "text": str(args.get("text", ""))}]},
            }
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "unknown tool"}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "no method"}}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        pass

    def _read(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self):  # noqa: N802 (http.server API)
        msg = self._read()
        self.server.seen.append((dict(self.headers), msg))
        resp = _http_rpc(msg)
        if resp is None:
            self.send_response(202)
            self.end_headers()
            return
        if self.server.mode == "sse":
            payload = ("event: message\ndata: " + json.dumps(resp) + "\n\n").encode()
            ctype = "text/event-stream"
        else:
            payload = json.dumps(resp).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        if self.server.session_id:
            self.send_header("Mcp-Session-Id", self.server.session_id)
        self.end_headers()
        self.wfile.write(payload)

    def do_DELETE(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()


class HttpServerFixture:
    def __init__(self, mode="json", session_id=None):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.mode = mode
        self.httpd.seen = []
        self.httpd.session_id = session_id
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        host, port = self.httpd.server_address
        return "http://%s:%d/mcp" % (host, port)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
# Fake backend that emits a tool call on its first turn
# ---------------------------------------------------------------------------
class _ToolBackend(BaseBackend):
    backend_name = "fake-tools"

    def __init__(self, tool_name, arguments=None):
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.turns = 0

    def generate(self, prompt, **kwargs):
        self.turns += 1
        if self.turns == 1:
            return {
                "text": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps(self.arguments),
                        },
                    }
                ],
            }
        return {"text": "final answer", "finish_reason": "stop"}

    def generate_stream(self, prompt, **kwargs):
        yield {"text": "final answer"}

    def get_context_window(self):
        return 4096

    def count_tokens(self, text):
        return max(1, len(str(text)) // 4)


class TestNameConversion(unittest.TestCase):
    def test_sanitize_tool_name(self):
        self.assertEqual(sanitize_tool_name("weird/name"), "weird_name")
        self.assertEqual(sanitize_tool_name("ok-name_1"), "ok-name_1")
        self.assertEqual(sanitize_tool_name(""), "tool")
        self.assertEqual(sanitize_tool_name("a b.c"), "a_b_c")

    def test_from_wire_requires_name(self):
        with self.assertRaises(MCPProtocolError):
            MCPTool.from_wire({"description": "no name"})
        with self.assertRaises(MCPProtocolError):
            MCPTool.from_wire("not a dict")

    def test_to_openai_defaults_missing_schema(self):
        tool = MCPTool.from_wire({"name": "noop", "description": "d"})
        spec = tool.to_openai()
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "noop")
        self.assertEqual(spec["function"]["parameters"], {"type": "object", "properties": {}})

    def test_prefix_namespacing(self):
        tools = [MCPTool(name="add"), MCPTool(name="weird/name")]
        specs = mcp_tools_to_openai(tools, prefix="srv")
        self.assertEqual([s["function"]["name"] for s in specs], ["srv__add", "srv__weird_name"])


class TestStdioTransport(unittest.TestCase):
    def test_open_list_and_call(self):
        with stdio_server() as srv:
            tools = srv.list_tools()
            self.assertEqual(sorted(t.name for t in tools), ["add", "boom", "weird/name"])
            self.assertEqual(srv.server_info.get("name"), "fake-stdio")
            self.assertEqual(srv.call_tool("add", {"a": 2, "b": 3}), "5")

    def test_to_openai_tools_shape(self):
        with stdio_server() as srv:
            specs = srv.to_openai_tools()
            by_name = {s["function"]["name"]: s for s in specs}
            self.assertIn("add", by_name)
            self.assertIn("weird_name", by_name)  # sanitized
            self.assertEqual(by_name["add"]["function"]["parameters"]["required"], ["a", "b"])

    def test_tool_error_is_raised(self):
        with stdio_server() as srv:
            with self.assertRaises(MCPToolError):
                srv.call_tool("boom", {})

    def test_jsonrpc_error_is_raised(self):
        with stdio_server() as srv:
            with self.assertRaises(MCPProtocolError) as ctx:
                srv.call_tool("does-not-exist", {})
            self.assertEqual(ctx.exception.code, -32602)

    def test_pagination_collects_all_pages(self):
        with stdio_server("--paginate") as srv:
            tools = srv.list_tools()
            self.assertEqual(sorted(t.name for t in tools), ["add", "boom", "weird/name"])

    def test_refresh_tools_refetches(self):
        with stdio_server() as srv:
            first = srv.list_tools()
            second = srv.refresh_tools()
            self.assertEqual(len(first), len(second))

    def test_close_terminates_and_blocks_further_use(self):
        srv = stdio_server()
        srv.open()
        self.assertFalse(srv.closed)
        srv.close()
        self.assertTrue(srv.closed)
        with self.assertRaises(MCPProtocolError):
            srv.request("tools/list", {})

    def test_missing_command_raises_clean_error(self):
        srv = MCPServer("bad", command="/nonexistent/mcp-binary-xyz", timeout=2.0)
        with self.assertRaises(MCPError):
            srv.open()

    def test_timeout_is_reported(self):
        srv = stdio_server("--sleep", timeout=0.5)
        try:
            srv.open()
            with self.assertRaises(MCPProtocolError) as ctx:
                srv.list_tools()
            self.assertIn("Timed out", str(ctx.exception))
        finally:
            srv.close()


class TestHttpTransport(unittest.TestCase):
    def test_json_body_transport(self):
        fx = HttpServerFixture(mode="json")
        try:
            with MCPServer("http-json", url=fx.url, timeout=5.0) as srv:
                self.assertEqual([t.name for t in srv.tools()], ["echo"])
                self.assertEqual(srv.call_tool("echo", {"text": "hi"}), "hi")
        finally:
            fx.close()

    def test_sse_transport_and_session_header(self):
        fx = HttpServerFixture(mode="sse", session_id="sess-123")
        try:
            with MCPServer("http-sse", url=fx.url, timeout=5.0) as srv:
                self.assertEqual([t.name for t in srv.tools()], ["echo"])
                self.assertEqual(srv.call_tool("echo", {"text": "yo"}), "yo")
            # The initialize response carried a session id; later requests echoed it.
            init_headers = fx.httpd.seen[0][0]
            self.assertEqual(init_headers.get("Mcp-Session-Id"), None)
            later_headers = fx.httpd.seen[-1][0]
            self.assertEqual(later_headers.get("Mcp-Session-Id"), "sess-123")
        finally:
            fx.close()

    def test_http_error_is_wrapped(self):
        srv = MCPServer("dead", url="http://127.0.0.1:1/mcp", timeout=2.0)
        with self.assertRaises(MCPError):
            srv.open()


class TestConfigValidation(unittest.TestCase):
    def test_requires_exactly_one_transport(self):
        with self.assertRaises(ValueError):
            MCPServer("x")
        with self.assertRaises(ValueError):
            MCPServer("x", command="python", url="http://localhost:1/mcp")
        with self.assertRaises(ValueError):
            MCPServer("", command="python")
        with self.assertRaises(ValueError):
            MCPServer("x", url="ftp://nope")


class TestLLMIntegration(unittest.TestCase):
    def test_tool_loop_executes_mcp_tool(self):
        srv = stdio_server()
        backend = _ToolBackend("fake__add", {"a": 2, "b": 3})
        llm = LLM("test-model", backend_instance=backend, use_cache=False)
        try:
            names = llm.add_mcp_server(srv)
            self.assertIn("fake__add", names)
            response = llm.chat([{"role": "user", "content": "add 2 and 3"}])
            self.assertEqual(response["choices"][0]["message"]["content"], "final answer")
            # The server is tracked, and its tool output was fed back to the model.
            self.assertIn("fake", llm.mcp_servers)
            tool_texts = [m.get("content") for m in llm.memory.messages if m.get("role") == "tool"]
            self.assertIn("5", tool_texts)
        finally:
            llm.close()
        self.assertTrue(srv.closed)

    def test_include_excludes_tools(self):
        srv = stdio_server()
        llm = LLM(
            "test-model",
            backend_instance=_ToolBackend("fake__add", {"a": 1, "b": 1}),
            use_cache=False,
        )
        try:
            names = llm.add_mcp_server(srv, include=["add"])
            self.assertEqual(names, ["fake__add"])
        finally:
            llm.close()

    def test_no_prefix_keeps_raw_names(self):
        srv = stdio_server()
        llm = LLM("test-model", backend_instance=_ToolBackend("x"), use_cache=False)
        try:
            names = llm.add_mcp_server(srv, prefix=False)
            self.assertIn("add", names)
            self.assertIn("weird_name", names)
        finally:
            llm.close()

    def test_repr_smoke(self):
        srv = stdio_server()
        try:
            srv.open()
            self.assertIn("fake", repr(srv))
        finally:
            srv.close()


class TestCli(unittest.TestCase):
    def _run(self, argv):
        import contextlib
        import io

        from local_llm_kit import cli

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(argv)
        return code, out.getvalue()

    def test_mcp_tools_json(self):
        cmd = "%s %s" % (sys.executable, FAKE_SERVER)
        code, out = self._run(["mcp", "tools", "--command", cmd, "--json"])
        self.assertEqual(code, 0, msg=out)
        specs = json.loads(out)
        names = sorted(s["function"]["name"] for s in specs)
        self.assertEqual(names, ["add", "boom", "weird_name"])

    def test_mcp_tools_table(self):
        cmd = "%s %s" % (sys.executable, FAKE_SERVER)
        code, out = self._run(["mcp", "tools", "--command", cmd])
        self.assertEqual(code, 0, msg=out)
        self.assertIn("add", out)
        self.assertIn("required", out)

    def test_mcp_call(self):
        cmd = "%s %s" % (sys.executable, FAKE_SERVER)
        code, out = self._run(
            ["mcp", "call", "--command", cmd, "--tool", "add", "--args", '{"a": 4, "b": 5}']
        )
        self.assertEqual(code, 0, msg=out)
        self.assertEqual(out.strip(), "9")

    def test_mcp_tools_over_http_url(self):
        fx = HttpServerFixture(mode="json")
        try:
            code, out = self._run(["mcp", "tools", "--url", fx.url, "--json"])
            self.assertEqual(code, 0, msg=out)
            specs = json.loads(out)
            self.assertEqual([s["function"]["name"] for s in specs], ["echo"])
        finally:
            fx.close()

    def test_mcp_call_error_exit_code(self):
        cmd = "%s %s" % (sys.executable, FAKE_SERVER)
        with self.assertRaises(SystemExit) as ctx:
            self._run(["mcp", "call", "--command", cmd, "--tool", "boom"])
        self.assertEqual(ctx.exception.code, 1)


class TestCoreStaysLight(unittest.TestCase):
    def test_import_mcp_pulls_no_heavy_deps(self):
        code = (
            "import sys; import local_llm_kit.mcp; "
            "bad = [m for m in ('torch', 'transformers', 'llama_cpp') if m in sys.modules]; "
            "print(','.join(bad))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=_repo_root()
        )
        self.assertEqual(out.returncode, 0, msg=out.stderr[-2000:])
        self.assertEqual(out.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
