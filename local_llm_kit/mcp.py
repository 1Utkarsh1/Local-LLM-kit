"""
Model Context Protocol (MCP) client for local-llm-kit.

Enables local LLMs to connect directly to standard MCP tool and resource servers
(e.g., SQLite, Filesystem, Git, Brave search, memory, or custom scripts)
over JSON-RPC 2.0 stdio transport with zero mandatory external dependencies.

Example::

    from local_llm_kit import LLM
    from local_llm_kit.mcp import MCPClient

    with MCPClient.from_command("python", "my_mcp_server.py") as mcp:
        tools = mcp.list_tools()
        # Connect to any local backend (Ollama, llama.cpp, etc.)
        llm = LLM("llama3.2:3b", backend="ollama")
        response = llm.chat(
            messages=[{"role": "user", "content": "List tables in the database"}],
            tools=tools,
        )

Python 3.9+ compatible.
"""

import json
import logging
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    """Raised when an MCP operation fails or returns an error."""

    def __init__(self, message: str, code: Optional[int] = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class MCPClient:
    """Client for Model Context Protocol (MCP) servers communicating over stdio.

    Spawns the MCP server as a subprocess, manages JSON-RPC 2.0 framing over
    stdin/stdout, and provides normalized tool discovery and execution.

    Args:
        command: Command and arguments to launch the server subprocess.
        env: Optional environment variable dict to pass to the subprocess.
        cwd: Optional working directory for the subprocess.
        protocol_version: MCP protocol version string. Defaults to ``"2024-11-05"``.
        auto_connect: Automatically initialize on first call if not connected.
    """

    def __init__(
        self,
        command: Sequence[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        protocol_version: str = DEFAULT_MCP_PROTOCOL_VERSION,
        auto_connect: bool = True,
    ):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._req_id = 0
        self._initialized = False
        self._server_capabilities: Dict[str, Any] = {}
        self._server_info: Dict[str, Any] = {}

        if not command:
            raise ValueError("command must be a non-empty list or tuple of strings.")
        self.command = list(command)
        self.env = env
        self.cwd = cwd
        self.protocol_version = protocol_version
        self.auto_connect = auto_connect

    @classmethod
    def from_command(cls, *cmd: str, **kwargs: Any) -> "MCPClient":
        """Convenience factory: ``MCPClient.from_command("python", "server.py")``."""
        return cls(cmd, **kwargs)

    @property
    def is_connected(self) -> bool:
        """True if the subprocess is running and initialized."""
        return self._process is not None and self._process.poll() is None and self._initialized

    def connect(self) -> None:
        """Start the MCP server subprocess and complete the handshake."""
        with self._lock:
            if self.is_connected:
                return

            full_env = os.environ.copy()
            if self.env:
                full_env.update(self.env)
            # Ensure unbuffered python output when executing python scripts
            full_env["PYTHONUNBUFFERED"] = "1"

            try:
                self._process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    cwd=self.cwd,
                    env=full_env,
                )
            except Exception as exc:
                raise MCPError(
                    f"Failed to spawn MCP server process {self.command!r}: {exc}"
                ) from exc

            # Step 1: initialize request
            req = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {
                        "roots": {"listChanged": False},
                        "sampling": {},
                    },
                    "clientInfo": {
                        "name": "local-llm-kit",
                        "version": "0.2.0",
                    },
                },
            }
            resp = self._send_and_receive_unlocked(req)
            if "error" in resp:
                err = resp["error"]
                raise MCPError(
                    f"MCP initialization error: {err.get('message', 'unknown')}",
                    code=err.get("code"),
                    data=err.get("data"),
                )

            result = resp.get("result", {})
            self._server_capabilities = result.get("capabilities", {})
            self._server_info = result.get("serverInfo", {})

            # Step 2: initialized notification
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
            self._send_notification_unlocked(notification)
            self._initialized = True
            logger.debug(
                "MCP connected to %s (capabilities: %s)",
                self._server_info,
                self._server_capabilities,
            )

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            if self.auto_connect:
                self.connect()
            else:
                raise MCPError("MCPClient is not connected. Call connect() first.")

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send_notification_unlocked(self, payload: Dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise MCPError("Server process stdin is not available.")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(f"Failed to write to MCP server stdin: {exc}") from exc

    def _send_and_receive_unlocked(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise MCPError("Server process pipes are not open.")

        req_id = payload.get("id")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            stderr_out = ""
            if self._process.stderr:
                try:
                    stderr_out = self._process.stderr.read()
                except Exception:
                    pass
            raise MCPError(f"MCP server pipe broken: {exc}. Stderr: {stderr_out}") from exc

        while True:
            resp_line = self._process.stdout.readline()
            if not resp_line:
                poll = self._process.poll()
                stderr_out = ""
                if self._process.stderr:
                    try:
                        stderr_out = self._process.stderr.read()
                    except Exception:
                        pass
                raise MCPError(
                    f"MCP server closed stdout unexpectedly (exit code: {poll}). Stderr: {stderr_out}"
                )

            resp_line = resp_line.strip()
            if not resp_line:
                continue

            try:
                data = json.loads(resp_line)
            except json.JSONDecodeError:
                # Some servers print logging messages to stdout; ignore non-json lines
                logger.warning("MCP client received non-JSON line from server: %r", resp_line)
                continue

            # Check if this matches our request ID
            if data.get("id") == req_id:
                return data
            # If server sent notification or unprompted message, log and keep reading
            if "method" in data and "id" not in data:
                logger.debug("Received MCP server notification: %s", data.get("method"))
                continue

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a JSON-RPC request to the MCP server and return the 'result'."""
        self._ensure_connected()
        with self._lock:
            req = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": method,
                "params": params or {},
            }
            resp = self._send_and_receive_unlocked(req)
            if "error" in resp:
                err = resp["error"]
                raise MCPError(
                    f"MCP error for {method}: {err.get('message', 'unknown')}",
                    code=err.get("code"),
                    data=err.get("data"),
                )
            return resp.get("result")

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools and return them in OpenAI-compatible tool spec format.

        Returns:
            List of dicts formatted as ``{"type": "function", "function": {...}}``,
            ready to be passed directly to ``LLM.chat(..., tools=tools)``.
        """
        res = self.request("tools/list", {}) or {}
        raw_tools = res.get("tools", [])
        normalized = []
        for t in raw_tools:
            name = t.get("name")
            if not name:
                continue
            desc = t.get("description", "")
            schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": schema,
                    },
                }
            )
        return normalized

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Execute a tool on the MCP server and return the string content result.

        Args:
            name: Tool name as exposed by the server.
            arguments: Dictionary of input arguments.

        Returns:
            Extracted text result from the tool execution.

        Raises:
            MCPError: If the tool execution failed or returned an error status.
        """
        params = {"name": name, "arguments": arguments or {}}
        res = self.request("tools/call", params) or {}

        is_error = res.get("isError", False)
        content_items = res.get("content", [])

        texts = []
        for item in content_items:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    texts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
                elif item.get("type") == "resource":
                    res_obj = item.get("resource", {})
                    texts.append(str(res_obj.get("text", "")))
                else:
                    texts.append(str(item))
            else:
                texts.append(str(item))

        output_str = "\n".join(texts)
        if is_error:
            raise MCPError(f"Tool {name!r} reported execution failure: {output_str}")
        return output_str

    def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources provided by the MCP server."""
        res = self.request("resources/list", {}) or {}
        return res.get("resources", [])

    def read_resource(self, uri: str) -> str:
        """Read the content of a resource URI from the MCP server."""
        res = self.request("resources/read", {"uri": uri}) or {}
        contents = res.get("contents", [])
        texts = []
        for c in contents:
            if isinstance(c, dict):
                texts.append(c.get("text") or c.get("blob") or "")
            else:
                texts.append(str(c))
        return "\n".join(texts)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List prompt templates exposed by the MCP server."""
        res = self.request("prompts/list", {}) or {}
        return res.get("prompts", [])

    def close(self) -> None:
        """Cleanly terminate the MCP server subprocess."""
        if not hasattr(self, "_lock") or self._lock is None:
            return
        with self._lock:
            if self._process is not None:
                try:
                    if self._process.stdin:
                        self._process.stdin.close()
                except Exception:
                    pass
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None
                self._initialized = False

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
