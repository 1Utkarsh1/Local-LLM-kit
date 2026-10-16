"""Offline contract tests for Model Context Protocol (MCP) client."""

import json
import unittest
from unittest.mock import MagicMock, patch

from local_llm_kit.mcp import MCPClient, MCPError


class TestMCPClient(unittest.TestCase):
    def test_init_validation(self):
        with self.assertRaises(ValueError):
            MCPClient([])

        client = MCPClient(["python", "-m", "mock_server"])
        self.assertEqual(client.command, ["python", "-m", "mock_server"])
        self.assertFalse(client.is_connected)

    def test_mock_stdio_handshake_and_tools(self):
        client = MCPClient(["echo", "mock"], auto_connect=False)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()

        # Responses sequence for stdout:
        # 1. initialize response
        # 2. tools/list response
        # 3. tools/call response
        responses = [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test-server", "version": "1.0"},
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "tools": [
                            {
                                "name": "lookup_user",
                                "description": "Fetch user by ID",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"user_id": {"type": "integer"}},
                                    "required": ["user_id"],
                                },
                            }
                        ]
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "result": {
                        "content": [{"type": "text", "text": '{"name": "Alice", "role": "admin"}'}],
                        "isError": False,
                    },
                }
            )
            + "\n",
        ]

        mock_proc.stdout.readline.side_effect = responses

        with patch("subprocess.Popen", return_value=mock_proc):
            client.connect()
            self.assertTrue(client.is_connected)

            # Check tool listing and OpenAI-format conversion
            tools = client.list_tools()
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0]["type"], "function")
            self.assertEqual(tools[0]["function"]["name"], "lookup_user")
            self.assertEqual(tools[0]["function"]["description"], "Fetch user by ID")
            self.assertIn("user_id", tools[0]["function"]["parameters"]["properties"])

            # Call tool
            result = client.call_tool("lookup_user", {"user_id": 42})
            self.assertIn("Alice", result)

            client.close()
            self.assertFalse(client.is_connected)

    def test_tool_call_error_propagation(self):
        client = MCPClient(["echo", "mock"], auto_connect=False)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()

        responses = [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2024-11-05"},
                }
            )
            + "\n",
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "content": [{"type": "text", "text": "Database connection refused"}],
                        "isError": True,
                    },
                }
            )
            + "\n",
        ]
        mock_proc.stdout.readline.side_effect = responses

        with patch("subprocess.Popen", return_value=mock_proc):
            client.connect()
            with self.assertRaises(MCPError) as ctx:
                client.call_tool("query_db", {"q": "SELECT 1"})
            self.assertIn("reported execution failure", str(ctx.exception))
            self.assertIn("Database connection refused", str(ctx.exception))

    def test_json_rpc_error_handling(self):
        client = MCPClient(["echo", "mock"], auto_connect=False)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()

        responses = [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
            + "\n"
        ]
        mock_proc.stdout.readline.side_effect = responses

        with patch("subprocess.Popen", return_value=mock_proc):
            with self.assertRaises(MCPError) as ctx:
                client.connect()
            self.assertIn("Method not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
