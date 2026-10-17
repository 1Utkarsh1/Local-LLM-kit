"""Opt-in interop test: our stdlib MCP client vs. the official ``mcp`` SDK.

This is the *real runtime* counterpart to the mocked contract tests in
``tests/test_mcp.py``. It starts a server built with the official Python SDK
(``mcp`` >= 2.0, which requires Python >= 3.10) as a subprocess and drives it
through :class:`local_llm_kit.mcp.MCPServer`.

It is skipped by default so CI stays offline and light. Enable it explicitly::

    pip install "mcp"                       # Python >= 3.10
    LOCAL_LLM_KIT_MCP_INTEROP=1 pytest tests/test_mcp_interop.py -v

A skip means "not run here", not "passed".
"""

import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
