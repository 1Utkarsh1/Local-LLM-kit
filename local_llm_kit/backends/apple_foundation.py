"""
Apple Foundation Models (Apple Intelligence) backend.

Uses the ``apple-fm-sdk`` package to access Apple's on-device Apple Intelligence
language model shipped with macOS 26.0+ (and iOS 26+ / visionOS 26+).

This is a *native* backend: it talks to the on-device model directly through
``apple_fm_sdk.LanguageModelSession`` (not via an OpenAI-compatible proxy such
as apfel). The optional OpenAI-bridge route is already covered by the
``openai-compat`` backend.

Install (optional, macOS only)::

    pip install "local-llm-kit[apple-fm]"   # or: pip install apple-fm-sdk

Capabilities:
  * ``chat()`` and ``complete()`` / ``generate()`` (sync, bridged via asyncio)
  * Native function calling: OpenAI-style ``tools`` are converted to
    ``apple_fm_sdk.Tool`` instances and executed on-device by the session
  * ``response_format`` (JSON) via Apple's ``json_schema`` guided generation
  * No streaming: ``generate_stream`` / ``chat_stream`` raise
    ``NotImplementedError`` (the SDK's ``stream_response`` yields full
    snapshots, not deltas, which conflicts with this client's framing)

Notes:
  * The Apple SDK runs the on-device model; it is only usable on macOS 26.0+
    with Apple Intelligence enabled. The backend raises a clear error
    otherwise and never imports ``apple_fm_sdk`` at module import time.
  * A fresh ``LanguageModelSession`` is created per request, so conversation
    history passed to ``chat()`` is flattened into a single prompt. For
    persistent multi-turn sessions keep the same backend instance and call
    ``chat`` per turn with the accumulated message list.
  * On-device tool execution runs inside the Apple session. Pass
    ``execute_tools=False`` (as ``local-llm-kit serve`` does by default) to
    omit tool registration so the model responds as text-only; this keeps the
    kit's server-safe tool policy intact for this backend.
"""

import asyncio
import concurrent.futures
import json
import platform
import sys
from dataclasses import field, make_dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .base import BaseBackend

_APPLE_FM_IMPORT_ERROR = (
    "Apple Foundation Models backend requires 'pip install local-llm-kit[apple-fm]' "
    "(or 'pip install apple-fm-sdk'). This feature is only available on "
    "macOS 26.0+ with Apple Intelligence enabled."
)

_DEFAULT_CONTEXT_WINDOW = 4096


def _import_apple_fm() -> Any:
    """Lazily import ``apple_fm_sdk`` (an optional, macOS-only dependency).

    Raises ImportError with a clear, actionable message when the package is
    absent so callers learn exactly what to install.
    """
    try:
        import apple_fm_sdk as fm  # noqa: F401  (import-time side effects)
    except ImportError as exc:
        raise ImportError(_APPLE_FM_IMPORT_ERROR) from exc
    return fm


def apple_fm_available() -> bool:
    """Return True iff ``apple-fm-sdk`` is installed AND the on-device model is usable.

    Never raises: used by the ``auto`` backend resolver and CLI discovery so the
    light core install is never disturbed by a missing platform-specific extra.
    """
    if sys.platform != "darwin":
        return False
    try:
        fm = _import_apple_fm()
    except ImportError:
        return False
    SystemLanguageModel = getattr(fm, "SystemLanguageModel", None)
    if SystemLanguageModel is None:
        return False
    try:
        result = SystemLanguageModel.is_available()
    except TypeError:
        # is_available() is an instance method on the real SDK.
        try:
            result = SystemLanguageModel().is_available()
        except Exception:
            return False
    except Exception:
        return False
    if isinstance(result, (tuple, list)) and result:
        return bool(result[0])
    return bool(result)


class AppleFoundationBackend(BaseBackend):
    """Native backend for Apple's on-device Foundation Models (Apple Intelligence).

    Requires ``apple-fm-sdk`` and macOS 26.0+ with Apple Intelligence enabled.
    Heavy dependencies are imported lazily inside the constructor and request
    helpers, so importing this module is always safe.
    """

    backend_name = "apple-foundation"
    #: The kit drives tool execution for this backend (see _accepts_tool_impls).
    supports_native_chat = True
    supports_vision = False
    supports_embeddings = False
    #: Opt-in marker: the LLM forwards registered tool implementations here so
    #: the on-device session can execute them. Gated so other native backends
    #: (ollama, openai-compat) receive no extra kwargs.
    _accepts_tool_impls = True

    def __init__(
        self,
        model_name: str = "system",
        instructions: Optional[str] = None,
        context_window: int = _DEFAULT_CONTEXT_WINDOW,
        **kwargs: Any,
    ) -> None:
        """Create an on-device Apple Foundation Models backend.

        Args:
            model_name: Logical model id reported by model_info() (the SDK
                exposes only the system model, so "system" is the default).
            instructions: Optional system instructions seeded into every
                session. System messages in ``chat()`` prepend to this.
            context_window: Reported context window size for token accounting.
            model: Optional alias of ``model_name`` (when passed as a
                keyword argument; ``LLM`` forwards ``model_path`` positionally).
        """
        fm = _import_apple_fm()
        self._fm = fm
        self._SystemLanguageModel = fm.SystemLanguageModel
        self._LanguageModelSession = fm.LanguageModelSession
        self._GenerationOptions = fm.GenerationOptions
        self._Tool = fm.Tool
        self._generable = fm.generable
        self._GenerationSchema = fm.GenerationSchema

        available, reason = self._check_available(self._SystemLanguageModel)
        if not available:
            raise RuntimeError(
                "Apple Foundation Models are not available on this system (%s). "
                "Requires macOS 26.0+ with Apple Intelligence enabled."
                % (reason or "unknown reason")
            )

        self._model = self._SystemLanguageModel()
        self.model = kwargs.pop("model", model_name)
        self.model_name = self.model
        self.model_path = self.model
        self.instructions = instructions
        self.context_window = int(context_window or _DEFAULT_CONTEXT_WINDOW)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @staticmethod
    def _check_available(system_language_model: Any) -> Tuple[bool, str]:
        """Resolve ``SystemLanguageModel.is_available()`` to a ``(bool, str)`` tuple.

        The SDK documents the check as a classmethod in some references and as an
        instance method in others; support both forms so the backend is robust.
        """
        try:
            result = system_language_model.is_available()
        except TypeError:
            try:
                result = system_language_model().is_available()
            except Exception as exc:  # pragma: no cover - defensive
                return False, "is_available() raised %s" % type(exc).__name__
        except Exception as exc:  # pragma: no cover - defensive
            return False, "is_available() raised %s" % type(exc).__name__
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return bool(result[0]), str(result[1])
        if isinstance(result, bool):
            return result, ""
        return bool(result), ""

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    def _run_async(self, coro: Any) -> Any:
        """Run an Apple SDK coroutine to completion from synchronous code.

        ``apple_fm_sdk`` exposes an async API only. When called from a thread
        with no running loop (the normal sync path, including the kit's
        ``achat`` which uses ``asyncio.to_thread``) we simply ``asyncio.run``
        it. When a loop *is* already running we execute the coroutine on a
        private loop in a dedicated thread so we never nest event loops.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    # ------------------------------------------------------------------
    # Prompt / message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _message_text(content: Any) -> str:
        """Flatten an OpenAI message ``content`` (str or vision part list) to text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type", "text")
                    if ptype == "text":
                        parts.append(str(part.get("text", "")))
                    elif ptype in ("image_url", "image"):
                        parts.append("[image]")
                    else:
                        parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)

    @classmethod
    def _convert_messages(
        cls, messages: List[Dict[str, Any]], instructions: Optional[str]
    ) -> Tuple[str, str]:
        """Split OpenAI messages into ``(system_instruction, apple_prompt)``.

        The Apple session accepts a single prompt string (``respond(prompt)``);
        conversation history is flattened so a fresh session has context.
        """
        sys_parts: List[str] = [instructions] if instructions else []
        body: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = cls._message_text(message.get("content"))
            if role == "system":
                sys_parts.append(content)
                continue
            if role == "user":
                body.append("User: %s" % content)
            elif role == "assistant":
                body.append("Assistant: %s" % content)
                for call in message.get("tool_calls") or []:
                    func = call.get("function", call) if isinstance(call, dict) else {}
                    call_name = func.get("name", "")
                    call_args = func.get("arguments", "")
                    body.append("Called: %s(%s)" % (call_name, call_args))
            elif role == "tool":
                body.append("Tool result (%s): %s" % (message.get("name", ""), content))
            else:
                body.append("%s: %s" % (role.capitalize(), content))
        body.append("Assistant:")
        instruction = "\n".join(p for p in sys_parts if p).strip()
        prompt = "\n".join(body)
        return instruction, prompt

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response(raw: Any) -> Tuple[str, Optional[Dict[str, int]], str]:
        """Map an Apple SDK response to ``(content, usage, finish_reason)``.

        Handles the real SDK's plain ``str`` return as well as ``GeneratedContent``
        (guided generation) and any ``{"content"/"usage"}`` shaped object.
        """
        usage: Optional[Dict[str, int]] = None
        content: Any = None

        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, dict):
            content = raw.get("content") or raw.get("text") or ""
            usage = raw.get("usage")
        else:
            # Guided / structured responses: generated content objects. Prefer
            # the SDK's explicit ``content``/``text`` attrs, then ``to_json()``
            # (GeneratedContent) and finally a ``value()`` fall-through.
            content = getattr(raw, "content", None)
            if content is None:
                content = getattr(raw, "text", None)
            if content is None:
                to_json = getattr(raw, "to_json", None)
                if callable(to_json):
                    try:
                        content = to_json()
                    except Exception:
                        content = None
            if content is None:
                value = getattr(raw, "value", None)
                if value is not None:
                    try:
                        resolved = value() if callable(value) else value
                    except Exception:
                        resolved = None
                    if isinstance(resolved, str):
                        content = resolved
                    elif resolved is None:
                        content = ""
                    else:
                        try:
                            content = json.dumps(resolved)
                        except (TypeError, ValueError):
                            content = str(resolved)
            usage = getattr(raw, "usage", None)
            if not isinstance(usage, dict):
                usage = None
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        return content, usage, "stop"

    # ------------------------------------------------------------------
    # Options / tools
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_schema(
        response_format: Optional[Union[str, Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Convert an OpenAI ``response_format`` into an Apple ``json_schema`` dict."""
        if not response_format or not isinstance(response_format, dict):
            return None
        rtype = response_format.get("type")
        if rtype == "json_object":
            return {"type": "object"}
        if rtype == "json_schema":
            nested = response_format.get("json_schema") or response_format.get("schema")
            if isinstance(nested, dict):
                schema = nested.get("schema")
                if isinstance(schema, dict):
                    return schema
                if nested.get("type"):
                    return nested
            return {"type": "object"}
        return None

    def _build_options(
        self,
        temperature: Optional[float],
        max_new_tokens: Optional[int],
    ) -> Optional[Any]:
        """Build a ``GenerationOptions`` (or None for SDK defaults)."""
        opts: Dict[str, Any] = {}
        if temperature is not None:
            opts["temperature"] = float(temperature)
        if max_new_tokens is not None:
            opts["maximum_response_tokens"] = int(max_new_tokens)
        if not opts:
            return None
        return self._GenerationOptions(**opts)

    @staticmethod
    def _normalize_tool_spec(tool: Any) -> Optional[Dict[str, Any]]:
        """Normalize a tool entry (OpenAI spec, bare spec, or callable) to a function dict."""
        if isinstance(tool, dict):
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                return tool["function"]
            if "name" in tool:
                return tool
            return None
        spec = getattr(tool, "spec", None)
        if isinstance(spec, dict):
            fn = spec.get("function", spec) if isinstance(spec, dict) else {}
            if isinstance(fn, dict):
                return fn
        return None

    @staticmethod
    def _json_type_to_python(schema: Any) -> Any:
        """Map a JSON Schema fragment to a Python type for the generable Arguments class."""
        if not isinstance(schema, dict):
            return str
        jtype = schema.get("type", "string")
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List[str],
            "object": str,
        }
        return mapping.get(jtype, str)

    def _build_arguments_class(self, parameters: Any, tool_name: str) -> Any:
        """Build a ``@generable`` Arguments class from an OpenAI tool parameters schema.

        Required properties are emitted first (dataclass rule: fields with
        defaults must follow non-default fields). Unknown/nested JSON types
        fall back to ``str`` so the model can still pass a JSON blob.
        """
        props: Dict[str, Any] = {}
        required: List[str] = []
        if isinstance(parameters, dict):
            props = parameters.get("properties", {}) or {}
            required = list(parameters.get("required", []) or [])
        required_set = set(required)

        required_fields: List[Any] = []
        optional_fields: List[Any] = []
        for pname, pschema in props.items():
            ptype = self._json_type_to_python(pschema)
            if pname in required_set:
                required_fields.append((pname, ptype))
            else:
                optional_fields.append((pname, ptype, field(default=None)))
        fields = required_fields + optional_fields
        try:
            args_cls = make_dataclass("Arguments", fields)
            decorated = self._generable(tool_name + " arguments")(args_cls)
        except Exception:
            # Fallback: a single optional ``arguments`` string field so the
            # model can still hand us a JSON blob to parse in call().
            fallback = make_dataclass("Arguments", [("arguments", str)])
            try:
                decorated = self._generable(tool_name + " arguments")(fallback)
            except Exception:
                decorated = fallback
        return decorated

    def _build_apple_tools(
        self,
        tools: Optional[List[Any]],
        implementations: Optional[Dict[str, Any]],
        execute_tools: bool,
    ) -> List[Any]:
        """Convert OpenAI-style tool specs into ``apple_fm_sdk.Tool`` instances."""
        if not execute_tools:
            return []
        impls = implementations or {}
        apple_tools: List[Any] = []
        for tool in tools or []:
            spec = self._normalize_tool_spec(tool)
            if spec is None:
                continue
            name = spec.get("name", "")
            description = spec.get("description", "")
            parameters = spec.get("parameters", {"type": "object", "properties": {}})
            apple_tools.append(
                self._make_apple_tool(name, description, parameters, impls.get(name))
            )
        return apple_tools

    def _make_apple_tool(
        self, name: str, description: str, parameters: Any, implementation: Optional[Any]
    ) -> Any:
        """Dynamically build an ``apple_fm_sdk.Tool`` subclass for one function."""
        args_cls = self._build_arguments_class(parameters, name)
        tool_name = name
        tool_description = description
        field_types = dict(getattr(args_cls, "__dataclass_fields__", {}) or {})

        tool_base = self._Tool

        class _AppleTool(tool_base):  # type: ignore[misc, valid-type]

            @property
            def name(self_) -> str:
                return tool_name

            @property
            def description(self_) -> str:
                return tool_description

            @property
            def arguments_schema(self_) -> Any:
                schema_fn = getattr(args_cls, "generation_schema", None)
                if schema_fn is None:
                    return self._GenerationSchema()  # type: ignore[attr-defined]
                try:
                    return schema_fn()
                except Exception:
                    return self._GenerationSchema()  # type: ignore[attr-defined]

            async def call(self_, args: Any) -> str:
                extracted = _extract_apple_args(args, field_types)
                # NOTE: read `implementation` from the closure (not a class
                # attribute): a plain function stored as a class attr becomes a
                # descriptor and would bind the tool instance as `self`.
                impl = implementation
                if impl is None:
                    return (
                        "Error: tool '%s' has no registered implementation in this backend."
                        % tool_name
                    )
                try:
                    result = impl(**extracted)
                except Exception as exc:  # noqa: BLE001 - surface as tool output, not raise
                    return "Error: tool '%s' failed (%s: %s)" % (
                        tool_name,
                        type(exc).__name__,
                        str(exc)[:200],
                    )
                if isinstance(result, str):
                    return result
                try:
                    return json.dumps(result)
                except (TypeError, ValueError):
                    return str(result)

        return _AppleTool()

    # ------------------------------------------------------------------
    # Prompt API (used by LLM.complete -> backend.generate)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if stream:
            raise NotImplementedError("Apple Foundation Models backend does not support streaming.")
        options = self._build_options(temperature, max_new_tokens)
        session = self._LanguageModelSession(instructions=self.instructions, model=self._model)
        content, usage, finish = self._run_respond(session, prompt, options=options)
        return {
            "text": content,
            "finish_reason": finish,
            "model": self.model,
            "usage": usage,
        }

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError("Apple Foundation Models backend does not support streaming.")

    # ------------------------------------------------------------------
    # Chat API (used by LLM.chat when supports_native_chat is True)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if stream:
            raise NotImplementedError("Apple Foundation Models backend does not support streaming.")
        implementations = kwargs.pop("implementations", None)
        execute_tools = kwargs.pop("execute_tools", True)
        instructions, prompt = self._convert_messages(messages, self.instructions)
        apple_tools = self._build_apple_tools(tools, implementations, execute_tools)
        options = self._build_options(temperature, max_new_tokens)
        json_schema = self._extract_json_schema(response_format)
        session = self._LanguageModelSession(
            instructions=instructions or None,
            model=self._model,
            tools=apple_tools or None,
        )
        content, usage, finish = self._run_respond(
            session, prompt, options=options, json_schema=json_schema
        )
        return {
            "text": content,
            "finish_reason": finish,
            "model": self.model,
            "usage": usage,
        }

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError("Apple Foundation Models backend does not support streaming.")

    # ------------------------------------------------------------------
    # Async respond
    # ------------------------------------------------------------------

    def _run_respond(
        self,
        session: Any,
        prompt: str,
        options: Optional[Any] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[Dict[str, int]], str]:
        return self._run_async(self._respond_async(session, prompt, options, json_schema))

    async def _respond_async(
        self,
        session: Any,
        prompt: str,
        options: Optional[Any] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[Dict[str, int]], str]:
        call_kwargs: Dict[str, Any] = {}
        if options is not None:
            call_kwargs["options"] = options
        if json_schema is not None:
            call_kwargs["json_schema"] = json_schema
        raw = await session.respond(prompt, **call_kwargs)
        return self._extract_response(raw)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_context_window(self) -> int:
        return self.context_window

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def model_info(self) -> Dict[str, Any]:
        info = super().model_info()
        info.update(
            {
                "model": self.model,
                "apple_foundation": True,
                "streaming": False,
                "platform": platform.platform(),
            }
        )
        return info

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": self.model,
                "object": "model",
                "owned_by": "apple-foundation",
            }
        ]


def _field_python_type(field_obj: Any) -> Any:
    """Resolve a ``dataclasses.Field`` type to a concrete Python type for SDK queries.

    ``args.value(type, for_property=name)`` expects a plain Python type; typing
    generics (e.g. ``List[str]``) resolve to their origin (e.g. ``list``).
    """
    ftype = getattr(field_obj, "type", None) or str
    if isinstance(ftype, type):
        return ftype
    origin = getattr(ftype, "__origin__", None)
    if origin is not None and isinstance(origin, type):
        return origin
    return str


def _extract_apple_args(args: Any, field_types: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parsed arguments from an Apple ``GeneratedContent`` object.

    Uses the SDK's ``args.value(type, for_property=name)`` API with type
    fallbacks so primitive schemas work on-device while unknown/nested types
    degrade to ``None`` rather than crashing the tool call.
    """
    extracted: Dict[str, Any] = {}
    if args is None:
        return extracted
    value_fn = getattr(args, "value", None)
    if not callable(value_fn):
        # Some SDK variants expose the dict directly.
        raw = getattr(args, "content_dict", None) or getattr(args, "to_dict", None)
        if callable(raw):
            raw = raw()
        if isinstance(raw, dict):
            return dict(raw)
        return extracted
    for name, field_obj in field_types.items():
        python_type = _field_python_type(field_obj)
        try:
            extracted[name] = value_fn(python_type, for_property=name)
        except Exception:
            try:
                extracted[name] = value_fn(str, for_property=name)
            except Exception:
                extracted[name] = None
    return extracted
