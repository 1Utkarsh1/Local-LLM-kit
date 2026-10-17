"""Opt-in SQLite conversation history with atomic writes and lossless message export.

One locked connection works across threads, including an in-memory database.
Nothing is written until a ConversationLog is explicitly constructed.
"""

import json
import os
import threading
import uuid
from functools import wraps
from typing import Any, Dict, List, Optional

__all__ = ["ConversationLog", "DEFAULT_LOG_PATH", "get_conversation_log"]


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


def _atomic(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock, self._conn:
            self._batch = True
            try:
                return method(self, *args, **kwargs)
            finally:
                self._batch = False

    return call


# ---------------------------------------------------------------------------
# Default on-disk path (XDG_DATA_HOME or ~/.local/share/local-llm-kit/)
# ---------------------------------------------------------------------------


def _default_log_path() -> str:
    """Resolve the default SQLite log path (XDG_DATA_HOME or platform default)."""
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return os.path.join(xdg, "local-llm-kit", "conversation_logs.db")
    return os.path.join(
        os.path.expanduser("~"),
        ".local",
        "share",
        "local-llm-kit",
        "conversation_logs.db",
    )


DEFAULT_LOG_PATH = _default_log_path()


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not exist."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _now_ts() -> int:
    import time

    return int(time.time())


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    created_at    INTEGER NOT NULL,
    model         TEXT,
    system_prompt TEXT,
    metadata      TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id   TEXT NOT NULL,
    created_at        INTEGER NOT NULL,
    role              TEXT NOT NULL,
    content           TEXT,
    model             TEXT,
    finish_reason     TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    payload           TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    name      TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result    TEXT,
    call_id   TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);
"""


# ---------------------------------------------------------------------------
# ConversationLog
# ---------------------------------------------------------------------------


class ConversationLog:
    """SQLite-backed persistent log of LLM conversations.

    Stores conversations, their messages (user / assistant / tool / system),
    and any tool-call records produced during a turn.  Connections are
    serialized with a reentrant lock, including complete interaction transactions.

    :param path: filesystem path to the SQLite database file.  When
        ``None`` the default XDG/standard-path location is used
        (see :data:`DEFAULT_LOG_PATH`).
    """

    def __init__(self, path: Optional[str] = None) -> None:
        import sqlite3

        self._path = str(path) if path is not None else DEFAULT_LOG_PATH
        self._lock = threading.RLock()
        self._batch = False
        _ensure_parent_dir(self._path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            # Preserve databases made by the original prototype.
            for table, column in (("messages", "payload"), ("tool_calls", "call_id")):
                columns = {row[1] for row in self._conn.execute("PRAGMA table_info(%s)" % table)}
                if column not in columns:
                    self._conn.execute("ALTER TABLE %s ADD COLUMN %s TEXT" % (table, column))

    @_locked
    def close(self) -> None:
        """Close the store; it cannot be used again after closing."""
        self._conn.close()

    def __enter__(self) -> "ConversationLog":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _execute(
        self,
        query: str,
        params: Optional[tuple] = None,
        commit: bool = False,
    ) -> Any:
        """Execute a statement while the caller holds the connection lock."""
        conn = self._conn
        cur = conn.execute(query, params or ())
        if commit and not self._batch:
            conn.commit()
        return cur

    # -- CRUD -----------------------------------------------------------------

    @_locked
    def create(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new conversation and return its ID."""
        conv_id = uuid.uuid4().hex
        ts = _now_ts()
        meta_json = json.dumps(metadata) if metadata is not None else None
        self._execute(
            "INSERT INTO conversations (id, created_at, model, system_prompt, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, ts, model, system_prompt, meta_json),
            commit=True,
        )
        return conv_id

    @_locked
    def add_message(
        self,
        conv_id: str,
        role: str,
        content: Optional[str],
        model: Optional[str] = None,
        finish_reason: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a message to *conv_id* and return the new message row id."""
        ts = _now_ts()
        cur = self._execute(
            "INSERT INTO messages "
            "(conversation_id, created_at, role, content, model, finish_reason, "
            " prompt_tokens, completion_tokens, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                ts,
                role,
                content,
                model,
                finish_reason,
                prompt_tokens,
                completion_tokens,
                json.dumps(payload) if payload is not None else None,
            ),
            commit=True,
        )
        return int(cur.lastrowid)

    @_locked
    def add_tool_call(
        self,
        message_id: int,
        name: str,
        arguments: Any,
        result: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> int:
        """Record a tool call associated with *message_id*.

        *arguments* may be a ``dict`` (JSON-serialised automatically) or a
        JSON string.  Returns the new tool-call row id.
        """
        if isinstance(arguments, dict):
            args_json = json.dumps(arguments)
        elif isinstance(arguments, str):
            args_json = arguments
        else:
            args_json = json.dumps(arguments)
        cur = self._execute(
            "INSERT INTO tool_calls (message_id, name, arguments, result, call_id) VALUES (?, ?, ?, ?, ?)",
            (message_id, name, args_json, result, call_id),
            commit=True,
        )
        return int(cur.lastrowid)

    @_locked
    def get_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent conversations (newest first)."""
        conn = self._conn
        rows = conn.execute(
            "SELECT id, created_at, model, system_prompt, metadata "
            "FROM conversations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_conv(row) for row in rows]

    @_locked
    def get_conversation(self, conv_id: str) -> Dict[str, Any]:
        """Return a single conversation's metadata (raises ``KeyError`` if absent)."""
        conn = self._conn
        row = conn.execute(
            "SELECT id, created_at, model, system_prompt, metadata "
            "FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if row is None:
            raise KeyError("Conversation not found: %s" % conv_id)
        return _row_to_conv(row)

    @_locked
    def get_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """Return all messages for *conv_id* in insertion order."""
        conn = self._conn
        rows = conn.execute(
            "SELECT id, conversation_id, created_at, role, content, model, "
            "finish_reason, prompt_tokens, completion_tokens, payload "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

    @_locked
    def get_tool_calls(self, message_id: int) -> List[Dict[str, Any]]:
        """Return tool-call records associated with *message_id*."""
        conn = self._conn
        rows = conn.execute(
            "SELECT id, message_id, name, arguments, result, call_id "
            "FROM tool_calls WHERE message_id = ? ORDER BY id ASC",
            (message_id,),
        ).fetchall()
        return [_row_to_tool_call(row) for row in rows]

    @_locked
    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-message search (LIKE) across all conversations.

        Returns matching message dicts enriched with the parent
        conversation's ``model``.
        """
        pattern = "%%%s%%" % query
        conn = self._conn
        rows = conn.execute(
            "SELECT m.id, m.conversation_id, m.created_at, m.role, m.content, "
            "       m.model, m.finish_reason, m.prompt_tokens, m.completion_tokens, m.payload "
            "FROM messages m "
            "WHERE m.content LIKE ? "
            "ORDER BY m.created_at DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

    @_locked
    def delete(self, conv_id: str) -> int:
        """Delete a conversation and all of its messages/tool calls.

        Returns the number of conversations removed (0 or 1).
        """
        cur = self._execute(
            "DELETE FROM conversations WHERE id = ?",
            (conv_id,),
            commit=True,
        )
        return cur.rowcount

    @_locked
    def clear(self) -> None:
        """Delete *all* conversations, messages, and tool calls."""
        self._execute("DELETE FROM conversations", commit=True)

    # -- High-level convenience API ------------------------------------------

    @_atomic
    def log_interaction(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response: Any,
        tools_used: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Atomically append supplied messages and a response.

        With an existing ID, supply only new messages or a complete history
        beginning with previously stored messages; that prefix is deduplicated.
        Full message payloads preserve images, tool IDs and reasoning fields.
        """
        if conversation_id is None:
            conversation_id = self.create(model, system_prompt, metadata)
        else:
            self.get_conversation(conversation_id)
            if metadata is not None:
                self._update_conversation_metadata(conversation_id, metadata)
        if not isinstance(messages, list) or any(not isinstance(msg, dict) for msg in messages):
            raise TypeError("messages must be a list of message objects.")
        previous = [row["payload"] for row in self.get_messages(conversation_id)]
        if previous and messages[: len(previous)] == previous:
            messages = messages[len(previous) :]
        if isinstance(response, str):
            assistant, usage, finish = {"role": "assistant", "content": response}, {}, None
        elif isinstance(response, dict):
            choice = (response.get("choices") or [{}])[0]
            assistant = dict(
                choice.get("message") or {"content": response.get("text", response.get("content"))},
                role="assistant",
            )
            usage, finish = response.get("usage") or {}, choice.get("finish_reason")
        else:
            raise TypeError("response must be text or a chat response object.")
        if tools_used:
            existing = list(assistant.get("tool_calls") or [])
            keys = {
                call.get("id") or json.dumps(call.get("function", call), sort_keys=True)
                for call in existing
            }
            for call in tools_used:
                key = call.get("id") or json.dumps(call.get("function", call), sort_keys=True)
                if key not in keys:
                    existing.append(call)
                    keys.add(key)
            assistant["tool_calls"] = existing
        for index, message in enumerate(messages + [assistant]):
            if not isinstance(message.get("role"), str) or not message["role"]:
                raise ValueError("Every message must have a non-empty role.")
            final = index == len(messages)
            content = message.get("content")
            mid = self.add_message(
                conversation_id,
                message["role"],
                content if isinstance(content, str) or content is None else json.dumps(content),
                model=model,
                finish_reason=finish if final else None,
                prompt_tokens=usage.get("prompt_tokens") if final else None,
                completion_tokens=usage.get("completion_tokens") if final else None,
                payload=message,
            )
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                self.add_tool_call(
                    mid,
                    function["name"],
                    function.get("arguments", "{}"),
                    call.get("result", call.get("output")),
                    call.get("id"),
                )
        return conversation_id

    @_locked
    def get_conversation_history(
        self,
        conversation_id: str,
        include_tool_details: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return the full message + tool-call history for a conversation.

        Each message dict is enriched with a ``tool_calls`` key (list of
        parsed tool-call records) when ``include_tool_details`` is True.

        :raises KeyError: if the conversation does not exist.
        """
        self.get_conversation(conversation_id)  # validate existence
        messages = self.get_messages(conversation_id)
        if not include_tool_details:
            return messages
        result: List[Dict[str, Any]] = []
        for msg in messages:
            enriched = dict(msg)
            enriched["tool_calls"] = self.get_tool_calls(msg["id"])
            result.append(enriched)
        return result

    @_locked
    def export_conversation(
        self,
        conversation_id: str,
        format: str = "json",
    ) -> str:
        """Export a conversation's full history in the requested format.

        Supported formats:

        * ``"json"``  -- structured JSON (conversation + messages + tool calls).
        * ``"text"``  -- human-readable transcript.
        * ``"openai"`` -- flattened OpenAI-style message list suitable for
          ``llm.chat(messages=...)``.

        :raises KeyError: if the conversation does not exist.
        :raises ValueError: for unsupported formats.
        """
        fmt = format.lower().strip()
        if fmt not in ("json", "text", "openai"):
            raise ValueError("Unsupported export format: {0!r}".format(format))

        conv = self.get_conversation(conversation_id)
        history = self.get_conversation_history(conversation_id)

        if fmt == "json":
            return json.dumps(
                {
                    "conversation": conv,
                    "history": history,
                },
                indent=2,
                default=str,
            )

        if fmt == "text":
            lines: List[str] = []
            lines.append("=== Conversation {0} ===".format(conversation_id))
            lines.append("Model: {0}".format(conv.get("model") or "(unknown)"))
            ts = conv.get("created_at")
            if ts is not None:
                lines.append("Created: {0}".format(ts))
            sys = conv.get("system_prompt")
            if sys:
                lines.append("System: {0}".format(sys))
            lines.append("")
            for msg in history:
                role = msg.get("role", "?")
                content = msg.get("content")
                if role == "assistant" and not content:
                    content = "(no text content)"
                if content:
                    lines.append("{0}: {1}".format(role.capitalize(), content))
                for tc in msg.get("tool_calls", []):
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (ValueError, TypeError):
                            pass
                    lines.append(
                        "Assistant called: {0}({1})".format(
                            tc.get("name"), json.dumps(args) if isinstance(args, dict) else args
                        )
                    )
                    res = tc.get("result")
                    if res:
                        lines.append("  Result: {0}".format(res))
            lines.append("")
            lines.append("=== End of conversation ===")
            return "\n".join(lines)

        # fmt == "openai": flatten to OpenAI-style message list
        openai_msgs: List[Dict[str, Any]] = []
        for msg in history:
            if msg.get("payload") is not None:
                openai_msgs.append(msg["payload"])
                continue
            role = msg.get("role")
            entry: Dict[str, Any] = {"role": role, "content": msg.get("content")}
            if role == "assistant":
                tcs = msg.get("tool_calls", [])
                if tcs:
                    openai_calls = []
                    for tc in tcs:
                        args = tc.get("arguments", "{}")
                        if isinstance(args, (dict, list)):
                            args = json.dumps(args)
                        openai_calls.append(
                            {
                                "id": tc.get("call_id") or "call_{0}".format(tc.get("id", "")),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name"),
                                    "arguments": args,
                                },
                            }
                        )
                    entry["tool_calls"] = openai_calls
            elif role == "tool":
                raise ValueError(
                    "Legacy tool message has no original tool_call_id; export JSON/text instead of inventing replay linkage."
                )
            openai_msgs.append(entry)
        return json.dumps({"model": conv.get("model"), "messages": openai_msgs}, indent=2)

    # -- internal: metadata update -------------------------------------------

    def _update_conversation_metadata(self, conv_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata JSON on an existing conversation (best-effort)."""
        meta_json = json.dumps(metadata)
        self._execute(
            "UPDATE conversations SET metadata = ? WHERE id = ?",
            (meta_json, conv_id),
            commit=True,
        )


# ---------------------------------------------------------------------------
# Row deserializers
# ---------------------------------------------------------------------------


def _decode_json(value: Any) -> Any:
    """Best-effort JSON decode; returns the raw value when it fails."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _row_to_conv(row: Any) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "model": row["model"],
        "system_prompt": row["system_prompt"],
        "metadata": _decode_json(row["metadata"]),
    }


def _row_to_message(row: Any) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "created_at": row["created_at"],
        "role": row["role"],
        "content": row["content"],
        "model": row["model"],
        "finish_reason": row["finish_reason"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "payload": _decode_json(row["payload"]),
    }


def _row_to_tool_call(row: Any) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "message_id": row["message_id"],
        "name": row["name"],
        "arguments": _decode_json(row["arguments"]),
        "result": row["result"],
        "call_id": row["call_id"],
    }


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def get_conversation_log(path: Optional[str] = None) -> "ConversationLog":
    """Return a :class:`ConversationLog` (default path if *path* is ``None``).

    A simple process-level cache is used so repeated calls with no *path*
    return the same instance, avoiding repeated schema initialization.
    """
    global _DEFAULT_LOG_INSTANCE
    if path is None:
        if _DEFAULT_LOG_INSTANCE is None:
            _DEFAULT_LOG_INSTANCE = ConversationLog(path)
        return _DEFAULT_LOG_INSTANCE
    return ConversationLog(path)


_DEFAULT_LOG_INSTANCE: Optional["ConversationLog"] = None
