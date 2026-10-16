"""
Model Context Protocol (MCP) client for local-llm-kit.

MCP lets a host discover and call tools exposed by *servers* (filesystem,
git, sqlite, fetch, browsers, company APIs, ...).  This module is a small
**stdlib-only** client so ``local-llm-kit`` can act as an MCP host without
depending on the official ``mcp`` SDK or pulling anything into the light
core.  Discovered tools are converted to OpenAI ``tools`` specs, so they
drop straight into :meth:`local_llm_kit.LLM.chat` /
:meth:`local_llm_kit.LLM.add_mcp_server`.

Two transports are supported:

``stdio``
    Launch a local server as a subprocess (``command`` + ``args``) and speak
    newline-delimited JSON-RPC over its stdin/stdout.  This is the transport
    used by virtually every local MCP server (``npx @modelcontextprotocol/...``,
    ``uvx mcp-server-...``, ``python -m my_server``), and it keeps inference
    fully on-device — nothing leaves the machine unless *you* configure an
    HTTP server.

``http``
    POST JSON-RPC to a Streamable HTTP endpoint (``url``).  Responses may be
    a plain JSON body or a request-scoped ``text/event-stream``.

Example::

    from local_llm_kit import LLM
    from local_llm_kit.mcp import MCPServer

    with MCPServer("fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as srv:
        print([t.name for t in srv.tools()])
        llm = LLM(model_path="...")
        llm.add_mcp_server(srv)          # registers srv's tools on the LLM
        llm.chat([{"role": "user", "content": "List the files in /tmp"}])

Notes / compatibility:

  - The default protocol revision is :data:`DEFAULT_PROTOCOL_VERSION`; pass
    ``protocol_version=`` to override.  We accept whatever revision the server
    reports and never hard-fail on a mismatch.
  - ``initialize`` is sent best-effort: the 2026 revision is stateless and
    servers that do not implement it are tolerated.
  - Everything is synchronous and thread-safe (one request in flight at a
    time per server, guarded by a lock).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MCPError",
    "MCPProtocolError",
    "MCPToolError",
    "MCPTool",
    "MCPServer",
    "DEFAULT_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "mcp_tools_to_openai",
    "sanitize_tool_name",
]

#: Protocol revision we ask for by default.  Servers reply with the revision
#: they intend to use (older servers are widely deployed, so we request a
#: broadly supported revision and adapt to the answer).  Override per server
#: with ``MCPServer(..., protocol_version="...")``.
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

#: Revisions this client understands (used only for a friendly log line).
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28")

#: Sent as ``clientInfo`` during initialize.
CLIENT_NAME = "local-llm-kit"

#: Default request timeout in seconds (per JSON-RPC round trip).
DEFAULT_TIMEOUT = 30.0

#: OpenAI function names must match this charset; MCP names are freer.
_NAME_ALLOWED = re.compile(r"[^A-Za-z0-9_-]")


class MCPError(Exception):
    """Base class for all MCP client errors."""


class MCPProtocolError(MCPError):
    """Transport failed, the peer died, or a JSON-RPC error was returned."""

    def __init__(self, message: str, code: Optional[int] = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class MCPToolError(MCPError):
    """A ``tools/call`` returned ``isError: true`` or produced unusable output."""


def sanitize_tool_name(name: str) -> str:
    """Return *name* using only characters OpenAI function names allow.

    Invalid characters become ``_``; an empty result becomes ``"tool"``.
    """
    cleaned = _NAME_ALLOWED.sub("_", str(name or "").strip())
    return cleaned or "tool"


class MCPTool:
    """A tool advertised by an MCP server (already schema-normalized)."""

    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        server: Optional[str] = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP tool 'name' must be a non-empty string.")
        self.name = name
        self.description = description or ""
        self.input_schema = input_schema if isinstance(input_schema, dict) else {}
        self.server = server

    @classmethod
    def from_wire(cls, data: Dict[str, Any], server: Optional[str] = None) -> "MCPTool":
        """Build a tool from a ``tools/list`` entry."""
        if not isinstance(data, dict):
            raise MCPProtocolError(
                "tools/list entry must be an object, got %r" % type(data).__name__
            )
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MCPProtocolError("tools/list entry is missing a valid 'name'.")
        schema = data.get("inputSchema") or data.get("input_schema") or {}
        description = data.get("description") or data.get("title") or ""
        return cls(
            name=name,
            description=str(description),
            input_schema=schema if isinstance(schema, dict) else {},
            server=server,
        )

    def to_openai(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Convert to an OpenAI ``tools`` entry.

        ``parameters`` is the MCP ``inputSchema`` verbatim; when a server omits
        it we emit an empty object schema so the model still sees the tool.
        """
        params = self.input_schema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": sanitize_tool_name(name or self.name),
                "description": self.description,
                "parameters": params,
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "MCPTool(name=%r, server=%r)" % (self.name, self.server)


def mcp_tools_to_openai(
    tools: Iterable[MCPTool], prefix: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Convert MCP tools to OpenAI specs, optionally namespacing ``prefix__name``."""
    specs: List[Dict[str, Any]] = []
    for t in tools:
        qualified = "%s__%s" % (prefix, t.name) if prefix else t.name
        specs.append(t.to_openai(qualified))
    return specs


class _StdioTransport:
    """Newline-delimited JSON-RPC over a subprocess' stdin/stdout."""

    def __init__(
        self,
        command: str,
        args: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("MCP stdio command must be a non-empty string.")
        self.command = command
        self.args = [str(a) for a in (args or [])]
        self.cwd = cwd
        merged_env = dict(os.environ)
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        self.env = merged_env
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Any]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._stderr_tail: List[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        try:
            self._proc = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                cwd=self.cwd,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise MCPError("MCP server command not found: %r" % self.command) from exc
        except OSError as exc:
            raise MCPError("could not start MCP server %r: %s" % (self.command, exc)) from exc
        # Ignore SIGPIPE-style BrokenPipeError on Windows/POSIX alike.
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._lines.put(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("MCP server emitted non-JSON stdout: %r", line[:200])
        except (ValueError, OSError):  # pipe closed
            pass
        finally:
            self._lines.put(_EOF)

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr_tail.append(line.rstrip("\n"))
                if len(self._stderr_tail) > 50:
                    del self._stderr_tail[: len(self._stderr_tail) - 50]
        except (ValueError, OSError):
            pass

    def send(self, message: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPProtocolError(
                "MCP server is not running (command=%r).%s" % (self.command, self._stderr_hint())
            )
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise MCPProtocolError(
                "MCP server closed its input (command=%r).%s" % (self.command, self._stderr_hint())
            ) from exc

    def recv(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Return the next message, or ``None`` on timeout."""
        try:
            item = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _EOF:
            raise MCPProtocolError(
                "MCP server exited or closed stdout (command=%r).%s"
                % (self.command, self._stderr_hint())
            )
        return item if isinstance(item, dict) else None

    def _stderr_hint(self) -> str:
        tail = [ln for ln in self._stderr_tail if ln.strip()]
        if not tail:
            return ""
        return " Server stderr (last %d lines):\n%s" % (len(tail), "\n".join(tail[-10:]))

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        # Order matters: close stdin and stop the child first, then let the
        # reader threads observe EOF before closing their pipes. Closing a
        # pipe while another thread is blocked reading it can deadlock.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                    pass
        for thread in (self._reader, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    @property
    def session_id(self) -> Optional[str]:
        return None

    def set_session_id(self, value: Optional[str]) -> None:  # pragma: no cover - stdio ignores
        pass


class _HttpTransport:
    """Streamable HTTP JSON-RPC (JSON body or request-scoped SSE)."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("MCP http url must be a non-empty string.")
        if not url.startswith(("http://", "https://")):
            raise ValueError("MCP http url must start with http:// or https://, got %r" % url)
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._session_id: Optional[str] = None

    def start(self) -> None:  # nothing to spawn
        return None

    def send(self, message: Dict[str, Any]) -> None:
        raise MCPProtocolError("HTTP transport does not use send(); use post().")

    def recv(self, timeout: float) -> Optional[Dict[str, Any]]:  # pragma: no cover - unused
        return None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def set_session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    def post(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST *message* and return the matched JSON-RPC response (or None)."""
        body = json.dumps(message).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "%s/%s" % (CLIENT_NAME, _version()),
        }
        headers.update(self.headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
                if sid:
                    self._session_id = sid
                content_type = (resp.headers.get("Content-Type") or "").lower()
                payload = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001 - best-effort body
                pass
            raise MCPProtocolError(
                "MCP HTTP request to %s failed: HTTP %s %s %s"
                % (self.url, exc.code, exc.reason, detail)
            ) from exc
        except urllib.error.URLError as exc:
            raise MCPProtocolError(
                "MCP HTTP request to %s failed: %s" % (self.url, exc.reason)
            ) from exc
        if not payload.strip():
            return None
        if "text/event-stream" in content_type:
            return self._parse_sse(payload, message.get("id"))
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MCPProtocolError(
                "MCP HTTP server returned non-JSON body: %r" % payload[:200]
            ) from exc
        return self._select(parsed, message.get("id"))

    @staticmethod
    def _select(parsed: Any, want_id: Any) -> Optional[Dict[str, Any]]:
        if isinstance(parsed, list):  # batch
            for item in parsed:
                if isinstance(item, dict) and item.get("id") == want_id:
                    return item
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _parse_sse(payload: str, want_id: Any) -> Optional[Dict[str, Any]]:
        """Extract the JSON-RPC response with *want_id* from an SSE stream."""
        data_lines: List[str] = []
        found: Optional[Dict[str, Any]] = None
        for raw in payload.splitlines():
            line = raw.rstrip("\r")
            if not line.strip():
                if data_lines:
                    found = _sse_match("\n".join(data_lines), want_id) or found
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if data_lines:
            found = _sse_match("\n".join(data_lines), want_id) or found
        return found

    def close(self) -> None:
        # Best-effort session teardown; servers treat a missing session as gone.
        if not self._session_id:
            return
        headers = {
            "Accept": "application/json, text/event-stream",
            "User-Agent": "%s/%s" % (CLIENT_NAME, _version()),
            "Mcp-Session-Id": self._session_id,
        }
        headers.update(self.headers)
        req = urllib.request.Request(self.url, headers=headers, method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=5).close()
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass


def _sse_match(blob: str, want_id: Any) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and parsed.get("id") == want_id:
        return parsed
    return None


class _EofSentinel:
    """Sentinel pushed onto the stdio queue when the server's stdout closes."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "<MCP EOF>"


_EOF = _EofSentinel()


def _version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - always importable in practice
        return "0"


class MCPServer:
    """A connection to one MCP server plus the tools it advertises.

    Provide exactly one of ``command`` (stdio) or ``url`` (Streamable HTTP).
    """

    def __init__(
        self,
        name: str,
        command: Optional[str] = None,
        args: Optional[Sequence[str]] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP server 'name' must be a non-empty string.")
        if bool(command) == bool(url):
            raise ValueError("Provide exactly one of 'command' (stdio) or 'url' (http).")
        self.name = name.strip()
        self.timeout = float(timeout)
        self.protocol_version = protocol_version
        self.server_info: Dict[str, Any] = {}
        self.instructions: str = ""
        self._tools: Optional[List[MCPTool]] = None
        self._lock = threading.RLock()
        self._opened = False
        self._next_id = 0
        self._server_request_handler = None
        if command:
            self.transport: Any = _StdioTransport(command, args=args, env=env, cwd=cwd)
            self.transport_kind = "stdio"
        else:
            self.transport = _HttpTransport(url or "", headers=headers, timeout=self.timeout)
            self.transport_kind = "http"

    # -- lifecycle --------------------------------------------------------
    def open(self) -> "MCPServer":
        """Start the transport and perform the (best-effort) initialize."""
        if self._opened:
            return self
        self.transport.start()
        self._opened = True
        try:
            self.initialize()
        except MCPProtocolError:
            self.close()
            raise
        return self

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        try:
            self.transport.close()
        finally:
            self._tools = None

    def __enter__(self) -> "MCPServer":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return not self._opened

    # -- JSON-RPC ---------------------------------------------------------
    def _rpc_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a JSON-RPC request and return its ``result`` (raises on error)."""
        if not self._opened:
            raise MCPProtocolError("MCP server %r is not open; call open() first." % self.name)
        message: Dict[str, Any] = {"jsonrpc": "2.0", "id": self._rpc_id(), "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            if self.transport_kind == "http":
                response = self._http_roundtrip(message)
            else:
                self.transport.send(message)
                response = self._await_response(message["id"])
        return self._result_of(method, response)

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._opened:
            raise MCPProtocolError("MCP server %r is not open; call open() first." % self.name)
        message: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            if self.transport_kind == "http":
                self.transport.post(message)
            else:
                self.transport.send(message)

    def _await_response(self, want_id: Any) -> Optional[Dict[str, Any]]:
        while True:
            message = self.transport.recv(self.timeout)
            if message is None:
                raise MCPProtocolError(
                    "Timed out after %ss waiting for MCP server %r (id=%s)."
                    % (self.timeout, self.name, want_id)
                )
            if message.get("id") == want_id and ("result" in message or "error" in message):
                return message
            self._handle_unsolicited(message)

    def _http_roundtrip(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if message.get("id") is None:
            self.transport.post(message)
            return None
        response = self.transport.post(message)
        if response is None:
            raise MCPProtocolError(
                "MCP HTTP server %r returned no response for %s."
                % (self.name, message.get("method"))
            )
        return response

    def _handle_unsolicited(self, message: Dict[str, Any]) -> None:
        """Handle server->client requests / notifications we didn't ask for."""
        if "id" in message and "method" in message:
            # A server-initiated request (e.g. sampling). Reply so it is not left hanging.
            try:
                self.transport.send(
                    {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {"code": -32601, "message": "Method not supported by client"},
                    }
                )
            except MCPProtocolError:
                pass
        # Notifications are ignored.

    @staticmethod
    def _result_of(method: str, response: Optional[Dict[str, Any]]) -> Any:
        if response is None:
            return None
        error = response.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message", "unknown error")
                raise MCPProtocolError(
                    "MCP %s failed: %s" % (method, message),
                    code=error.get("code"),
                    data=error.get("data"),
                )
            raise MCPProtocolError("MCP %s failed: %s" % (method, error))
        return response.get("result")

    # -- protocol operations ----------------------------------------------
    def initialize(self) -> Dict[str, Any]:
        """Best-effort initialize; tolerates servers that do not implement it."""
        params = {
            "protocolVersion": self.protocol_version,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": _version()},
        }
        try:
            result = self.request("initialize", params) or {}
        except MCPProtocolError as exc:
            # Method-not-found (-32601) or a stateless server that answers with
            # a bare error: continue, since tools/* may still work.
            if exc.code == -32601 or "not found" in str(exc).lower():
                logger.debug("MCP server %r has no initialize; continuing.", self.name)
                return {}
            raise
        if isinstance(result, dict):
            self.server_info = result.get("serverInfo") or {}
            self.instructions = str(result.get("instructions") or "")
            agreed = result.get("protocolVersion")
            if agreed and agreed not in SUPPORTED_PROTOCOL_VERSIONS:
                logger.info(
                    "MCP server %r negotiated protocol %s (client knows %s).",
                    self.name,
                    agreed,
                    ", ".join(SUPPORTED_PROTOCOL_VERSIONS),
                )
        try:
            self.notify("notifications/initialized")
        except MCPProtocolError:
            pass
        return result if isinstance(result, dict) else {}

    def list_tools(self) -> List[MCPTool]:
        """Return the server's tools (cached until :meth:`refresh_tools`)."""
        with self._lock:
            if self._tools is not None:
                return list(self._tools)
        tools: List[MCPTool] = []
        cursor: Optional[str] = None
        seen_cursors = set()
        while True:
            params: Dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            result = self.request("tools/list", params or None) or {}
            raw = result.get("tools") if isinstance(result, dict) else None
            if not isinstance(raw, list):
                raise MCPProtocolError(
                    "MCP server %r returned an invalid tools/list result." % self.name
                )
            for entry in raw:
                tools.append(MCPTool.from_wire(entry, server=self.name))
            cursor = result.get("nextCursor") if isinstance(result, dict) else None
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
        with self._lock:
            self._tools = tools
        return list(tools)

    def refresh_tools(self) -> List[MCPTool]:
        """Drop the cache and re-fetch ``tools/list``."""
        with self._lock:
            self._tools = None
        return self.list_tools()

    def tools(self) -> List[MCPTool]:
        """Alias for :meth:`list_tools`."""
        return self.list_tools()

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Call one tool and flatten its MCP content blocks to text.

        Raises:
            MCPToolError: when the server reports ``isError`` or no usable
                content is returned.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool name must be a non-empty string.")
        args = arguments if isinstance(arguments, dict) else {}
        result = self.request("tools/call", {"name": name, "arguments": args}) or {}
        if not isinstance(result, dict):
            raise MCPToolError("MCP tool %r returned an unexpected result." % name)
        text = _flatten_content(result.get("content"), result.get("structuredContent"))
        if result.get("isError"):
            raise MCPToolError(text or ("MCP tool %r reported an error." % name))
        if not text:
            raise MCPToolError("MCP tool %r returned no content." % name)
        return text

    def to_openai_tools(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """All of this server's tools as OpenAI ``tools`` specs."""
        return mcp_tools_to_openai(self.list_tools(), prefix=prefix)

    def to_dict(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {"name": self.name, "transport": self.transport_kind}
        if self.transport_kind == "stdio":
            base["command"] = self.transport.command
            base["args"] = list(self.transport.args)
        else:
            base["url"] = self.transport.url
        base["server_info"] = dict(self.server_info)
        base["tools"] = [t.name for t in (self._tools or [])]
        return base

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "MCPServer(name=%r, transport=%s)" % (self.name, self.transport_kind)


def _flatten_content(content: Any, structured: Any = None) -> str:
    """Flatten MCP ``content`` blocks into a single string."""
    parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif btype in ("image", "audio"):
                mime = block.get("mimeType") or block.get("mime_type") or "unknown"
                parts.append("[%s: %s]" % (btype, mime))
            elif btype in ("resource", "resource_link"):
                uri = block.get("uri") or (block.get("resource") or {}).get("uri")
                parts.append("[resource: %s]" % (uri or "unknown"))
            else:
                # Unknown block type: keep something rather than dropping it.
                parts.append(json.dumps(block, default=str))
    elif isinstance(content, str):
        parts.append(content)
    if not parts and structured is not None:
        try:
            return json.dumps(structured, default=str)
        except (TypeError, ValueError):
            return str(structured)
    return "\n".join(part for part in parts if part)
