"""Opt-in interop test: our stdlib MCP client vs. the official ``mcp`` SDK.

This is the *real runtime* counterpart to the mocked contract tests in
``tests/test_mcp.py``. It starts a server built with the official Python SDK
(``mcp`` >= 2.0, which requires Python >= 3.10) as a subprocess and drives it
through :class:`local_llm_kit.mcp.MCPServer`.

It is skipped in the default suite and enabled in the Python 3.12 Linux CI job.
Enable it locally::

    pip install "mcp"                       # Python >= 3.10
    LOCAL_LLM_KIT_MCP_INTEROP=1 pytest tests/test_mcp_interop.py -v

A skip means "not run here", not "passed".
"""

import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

from local_llm_kit.mcp import MCPServer

RUN = os.environ.get("LOCAL_LLM_KIT_MCP_INTEROP") == "1"

HAVE_MCP = False
if RUN and sys.version_info >= (3, 10):
    try:
        import mcp  # noqa: F401

        HAVE_MCP = True
    except Exception:  # pragma: no cover - depends on the environment
        HAVE_MCP = False

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_interop_mcp_server.py")

_SKIP_REASON = (
    "opt-in interop test: set LOCAL_LLM_KIT_MCP_INTEROP=1 with the official "
    "'mcp' SDK installed on Python >= 3.10"
)


@unittest.skipUnless(RUN and HAVE_MCP, _SKIP_REASON)
class TestOfficialSdkInterop(unittest.TestCase):
    def test_list_and_call_tools_from_official_sdk_server(self):
        with MCPServer("interop", command=sys.executable, args=[SERVER], timeout=30.0) as srv:
            names = sorted(t.name for t in srv.tools())
            self.assertIn("add", names)
            self.assertIn("greet", names)

            specs = {s["function"]["name"]: s for s in srv.to_openai_tools()}
            self.assertEqual(specs["add"]["function"]["parameters"]["type"], "object")
            self.assertIn("a", specs["add"]["function"]["parameters"]["properties"])

            self.assertIn("42", srv.call_tool("add", {"a": 2, "b": 40}))
            self.assertIn("Ada", srv.call_tool("greet", {"name": "Ada"}))

    def test_stateless_2026_protocol_with_official_sdk(self):
        with MCPServer(
            "modern",
            command=sys.executable,
            args=[SERVER],
            timeout=10,
            protocol_version="2026-07-28",
        ) as server:
            self.assertIn("2026-07-28", server.initialize()["supportedVersions"])
            self.assertIn("add", [tool.name for tool in server.list_tools()])
            self.assertIn("42", server.call_tool("add", {"a": 2, "b": 40}))

    def test_stateless_http_json_and_sse_with_official_sdk(self):
        for json_mode in (False, True):
            with self.subTest(json_mode=json_mode):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                args = [sys.executable, SERVER, str(port)] + (["json"] if json_mode else [])
                process = subprocess.Popen(
                    args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                url = "http://127.0.0.1:%s/mcp" % port
                try:
                    deadline = time.monotonic() + 10
                    while True:
                        try:
                            with urllib.request.urlopen(url, timeout=0.1):
                                break
                        except urllib.error.HTTPError as error:
                            error.close()
                            break  # HTTP 405 is enough to establish readiness.
                        except OSError:
                            if process.poll() is not None or time.monotonic() >= deadline:
                                self.fail("Official MCP HTTP fixture did not start.")
                            time.sleep(0.02)
                    with MCPServer(
                        "http-interop", url=url, timeout=3, protocol_version="2026-07-28"
                    ) as server:
                        self.assertIn("add", [tool.name for tool in server.list_tools()])
                        self.assertIn("42", server.call_tool("add", {"a": 2, "b": 40}))
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
