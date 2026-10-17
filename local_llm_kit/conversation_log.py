"""
SQLite-backed conversation logging for local_llm_kit.

This module provides :class:`ConversationLog` -- a thread-safe (thread-local
SQLite connections) persistent store for LLM conversations, messages, and
tool-call records.  Uses only the Python standard library (``sqlite3``),
which is imported lazily so the light core install is never disturbed.

Thread safety
-------------
Each thread receives its own ``sqlite3.Connection`` via ``threading.local``
so connections are never shared across threads (the default SQLite behaviour
that would otherwise raise ``ProgrammingError``).  Write operations are
serialized by SQLite's own file locking with a generous busy timeout.

Python 3.9 compatible (stdlib only; ``typing.Dict``/``List``/``Optional``
house style).

Backward compatibility
----------------------
``PromptTemplate``, ``TemplateRegistry`` and the template convenience
helpers have moved to :mod:`local_llm_kit.prompt_templates`.  They are
re-exported here so existing ``from local_llm_kit.conversation_log import
PromptTemplate`` imports keep working.
"""

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

# Re-export prompt-template symbols for backward compatibility.  Importing
# prompt_templates imports this module (circular), so guard with a flag.
_PROMPT_TEMPLATES_LOADED = False
_PromptTemplate = None  # type: ignore[assignment, misc]
_TemplateRegistry = None  # type: ignore[assignment, misc]


def _ensure_prompt_template_exports() -> None:
    """Lazy-load prompt template exports on first access to avoid a cycle."""
    global _PromptTemplate, _TemplateRegistry, _PROMPT_TEMPLATES_LOADED
    if _PROMPT_TEMPLATES_LOADED:
        return
    from .prompt_templates import PromptTemplate, TemplateRegistry

    _PromptTemplate = PromptTemplate
    _TemplateRegistry = TemplateRegistry
    _PROMPT_TEMPLATES_LOADED = True


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for backward-compatible re-exports."""
    if name in (
        "PromptTemplate",
        "TemplateRegistry",
        "DEFAULT_REGISTRY",
        "register_template",
        "get_template",
        "render_template",
        "get_conversation_log",
    ):
        _ensure_prompt_template_exports()
        if name in ("PromptTemplate",):
            return _PromptTemplate
        if name in ("TemplateRegistry",):
            return _TemplateRegistry
        if name == "DEFAULT_REGISTRY":
            from .prompt_templates import DEFAULT_REGISTRY

            return DEFAULT_REGISTRY
        if name == "register_template":
            from .prompt_templates import register_template

            return register_template
        if name == "get_template":
            from .prompt_templates import get_template

            return get_template
        if name == "render_template":
            from .prompt_templates import render_template

            return render_template
        if name == "get_conversation_log":
            return get_conversation_log
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__() -> List[str]:
    base = [n for n in globals() if not n.startswith("_")]
    return base + [
        "PromptTemplate",
        "TemplateRegistry",
        "DEFAULT_REGISTRY",
        "register_template",
        "get_template",
        "render_template",
        "get_conversation_log",
    ]


__all__ = [  # noqa: F822
    "ConversationLog",
    "DEFAULT_LOG_PATH",
    "get_conversation_log",
    "PromptTemplate",
    "TemplateRegistry",
    "DEFAULT_REGISTRY",
    "register_template",
    "get_template",
    "render_template",
]

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
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    name      TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result    TEXT,
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
    thread-local so the object is safe to share across threads (each thread
    gets its own connection).

    :param path: filesystem path to the SQLite database file.  When
        ``None`` the default XDG/standard-path location is used
        (see :data:`DEFAULT_LOG_PATH`).
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or DEFAULT_LOG_PATH
        self._local = threading.local()
        self._init_lock = threading.Lock()
        _ensure_parent_dir(self._path)
        # Initialise the schema in the creator thread.
        self._get_conn()

    # -- connection management ------------------------------------------------

    def _get_conn(self) -> Any:
        """Return the thread-local SQLite connection, creating + initialising it if needed."""
        import sqlite3

        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.connection = conn
            with self._init_lock:
                conn.executescript(_SCHEMA)
                conn.commit()
        return conn

    @property
    def _conn(self) -> Any:
        """Thread-local connection property (lazily initialised)."""
        return self._get_conn()

    def close(self) -> None:
        """Close the connection for the *current* thread (if any)."""
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.connection = None

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
        """Execute a SQL statement on the thread-local connection."""
        conn = self._conn
        cur = conn.execute(query, params or ())
        if commit:
            conn.commit()
        return cur

    # -- CRUD -----------------------------------------------------------------

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

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: Optional[str],
        model: Optional[str] = None,
        finish_reason: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> int:
        """Append a message to *conv_id* and return the new message row id."""
        ts = _now_ts()
        cur = self._execute(
            "INSERT INTO messages "
            "(conversation_id, created_at, role, content, model, finish_reason, "
            " prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                ts,
                role,
                content,
                model,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            ),
            commit=True,
        )
        return int(cur.lastrowid)

    def add_tool_call(
        self,
        message_id: int,
        name: str,
        arguments: Any,
        result: Optional[str] = None,
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
            "INSERT INTO tool_calls (message_id, name, arguments, result) " "VALUES (?, ?, ?, ?)",
            (message_id, name, args_json, result),
            commit=True,
        )
        return int(cur.lastrowid)

    def get_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent conversations (newest first)."""
        conn = self._conn
        rows = conn.execute(
            "SELECT id, created_at, model, system_prompt, metadata "
            "FROM conversations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_conv(row) for row in rows]

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

    def get_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """Return all messages for *conv_id* in insertion order."""
        conn = self._conn
        rows = conn.execute(
            "SELECT id, conversation_id, created_at, role, content, model, "
            "finish_reason, prompt_tokens, completion_tokens "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

    def get_tool_calls(self, message_id: int) -> List[Dict[str, Any]]:
        """Return tool-call records associated with *message_id*."""
        conn = self._conn
        rows = conn.execute(
            "SELECT id, message_id, name, arguments, result "
            "FROM tool_calls WHERE message_id = ? ORDER BY id ASC",
            (message_id,),
        ).fetchall()
        return [_row_to_tool_call(row) for row in rows]

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-message search (LIKE) across all conversations.

        Returns matching message dicts enriched with the parent
        conversation's ``model``.
        """
        pattern = "%%%s%%" % query
        conn = self._conn
        rows = conn.execute(
            "SELECT m.id, m.conversation_id, m.created_at, m.role, m.content, "
            "       m.model, m.finish_reason, m.prompt_tokens, m.completion_tokens "
            "FROM messages m "
            "WHERE m.content LIKE ? "
            "ORDER BY m.created_at DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

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

    def clear(self) -> None:
        """Delete *all* conversations, messages, and tool calls."""
        self._execute("DELETE FROM conversations", commit=True)

    # -- High-level convenience API ------------------------------------------

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
        """Log a complete LLM interaction in a single call.

        This is the primary high-level entry point.  It creates (or reuses)
        a conversation, records every message in *messages*, the assistant
        response, and optionally the tool calls that were emitted.

        :param model: model identifier (e.g. ``"gpt-4o"`` or ``"echo"``).
        :param messages: the full message list sent to the model.  Each entry
            should be an OpenAI-style dict (``{"role": ..., "content": ...}``).
        :param response: the model's response dict (OpenAI ``chat`` shape),
            a plain string, or ``None``.
        :param tools_used: optional list of tool-call records (OpenAI-style
            ``tool_calls`` entries or ``{"name": ..., "arguments": ...}``).
        :param metadata: optional free-form metadata stored on the
            conversation.
        :param conversation_id: if provided, append to this existing
            conversation instead of creating a new one.
        :param system_prompt: optional system instruction stored on the
            conversation (only used when *conversation_id* is not given).

        :returns: the conversation ID.
        """
        if conversation_id is None:
            conversation_id = self.create(
                model=model,
                system_prompt=system_prompt,
                metadata=metadata,
            )
        elif metadata is not None:
            # Merge metadata into an existing conversation (best-effort).
            try:
                existing = self.get_conversation(conversation_id)
                existing_meta = existing.get("metadata") or {}
                if isinstance(existing_meta, dict):
                    merged = dict(existing_meta)
                    merged.update(metadata)
                    self._update_conversation_metadata(conversation_id, merged)
            except KeyError:
                pass

        # --- Record incoming messages ---
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if not role:
                continue
            content = msg.get("content")
            if isinstance(content, list):
                # Vision parts: flatten to text for logging.
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", part.get("type", "")) or "")
                    else:
                        parts.append(str(part))
                content = "\n".join(p for p in parts if p)
            elif content is not None:
                content = str(content)
            self.add_message(conv_id=conversation_id, role=role, content=content)

        # --- Record the assistant response ---
        assistant_content = None
        finish_reason = None
        tool_calls_list = []

        if isinstance(response, str):
            assistant_content = response
        elif isinstance(response, dict):
            # OpenAI-style: {"choices": [{"message": {"content": ...}}]}
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            else:
                msg = {}
            assistant_content = response.get("content") or response.get("text")
            if assistant_content is None and isinstance(msg, dict):
                assistant_content = msg.get("content")
            if isinstance(assistant_content, list):
                parts = []
                for part in assistant_content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", part.get("type", "")) or "")
                    else:
                        parts.append(str(part))
                assistant_content = "\n".join(p for p in parts if p)
            elif assistant_content is not None:
                assistant_content = str(assistant_content)
            usage = response.get("usage") or {}
            if isinstance(usage, dict):
                finish_reason = msg.get("finish_reason") if isinstance(msg, dict) else None
            tool_calls_list = response.get("tool_calls") or []
            if not tool_calls_list and isinstance(msg, dict):
                tool_calls_list = msg.get("tool_calls") or []

        msg_id = self.add_message(
            conv_id=conversation_id,
            role="assistant",
            content=assistant_content,
            model=model,
            finish_reason=finish_reason,
        )

        # --- Record tool calls (if any) ---
        all_tools = list(tools_used or []) + list(tool_calls_list or [])
        for tc in all_tools:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            name = fn.get("name", "") if isinstance(fn, dict) else ""
            if not name:
                continue
            arguments = fn.get("arguments", "{}")
            result = tc.get("result") or tc.get("output")
            try:
                self.add_tool_call(message_id=msg_id, name=name, arguments=arguments, result=result)
            except Exception:
                pass

        return conversation_id

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
                                "id": "call_{0}".format(tc.get("id", "")),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name"),
                                    "arguments": args,
                                },
                            }
                        )
                    entry["tool_calls"] = openai_calls
            elif role == "tool":
                entry["tool_call_id"] = msg.get("id")
                entry["name"] = msg.get("model") or msg.get("finish_reason")
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
    }


def _row_to_tool_call(row: Any) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "message_id": row["message_id"],
        "name": row["name"],
        "arguments": _decode_json(row["arguments"]),
        "result": row["result"],
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
