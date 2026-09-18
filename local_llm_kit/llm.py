"""
Core LLM class with an OpenAI-compatible API over local backends.

Backends: "auto" | "transformers" | "llamacpp" | "ollama" | "openai-compat" | "echo".
Only the stdlib is imported at module load; heavy backends (transformers,
llama.cpp, ...) are imported lazily inside ``_init_backend``.

Backwards compatible with the v0.1.x API (``functions``/``function_call``,
``format="json"``, ``logprobs``/``top_logprobs``, sync streaming) while adding:
modern ``tools``/``tool_choice``, ``response_format`` (json_object/json_schema),
vision message passthrough, ``seed``/``stop``, an auto tool-execution loop with
``tool`` role messages, ``embed()``, ``achat()``/``acomplete()``,
``add_tool()``/``add_function()``, and a module-level backend cache so
repeated ``LLM(model_path)`` construction reuses the loaded backend.

Security notes (see SECURITY review):
  - Tool execution is scoped to the tools advertised for the request
    (``tools=``/``functions=`` or the instance registry) and validated
    against each tool's ``required`` parameters before execution.
  - ``execute_tools=False`` returns tool calls without running them — the
    HTTP server uses this by default so remote callers cannot trigger
    local function execution unless explicitly enabled.
  - Tool implementations run with a configurable timeout
    (``tool_timeout``); failures become tool-output strings, never
    tracebacks, and server logs should redact secrets.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from .backends.base import BaseBackend
from .function_calling import FunctionCall  # noqa: F401 (re-exported for compat)
from .memory import MessageHistory
from .prompt_formatting import get_prompt_formatter, has_images
from .tools import (
    ToolCall,
    ToolRegistry,
    normalize_tools,
    parse_tool_calls,
    should_call_tools,
)

logger = logging.getLogger(__name__)

__all__ = ["LLM", "get_client", "clear_backend_cache", "clear_client_cache"]

_SUPPORTED_BACKENDS = ("auto", "transformers", "llamacpp", "ollama", "openai-compat", "echo")

#: Maximum characters of model output fed to the tool-call parser.
#: Bounds the regex work per request (ReDoS mitigation).
_PARSE_MAX_CHARS = 20000

# ---------------------------------------------------------------------------
# Module-level caches: repeated LLM(model_path) reuses loaded backends and
# client objects. Only the backend object is shared; memory/registries stay
# per-LLM unless the client itself is reused via get_client().
# ---------------------------------------------------------------------------
_BACKEND_CACHE: Dict[str, BaseBackend] = {}
_CLIENT_CACHE: Dict[str, "LLM"] = {}
_CACHE_LOCK = threading.Lock()


def _freeze_kwargs(kwargs: Dict[str, Any]) -> str:
    try:
        return json.dumps(kwargs, sort_keys=True, default=repr)
    except Exception:
        return repr(sorted(kwargs.items()))


def _cache_key(model_path: str, backend_name: Optional[str], backend_kwargs: Dict[str, Any]) -> str:
    return "%s|%s|%s" % (
        model_path,
        (backend_name or "auto").lower(),
        _freeze_kwargs(backend_kwargs),
    )


def clear_backend_cache() -> None:
    """Drop all cached backend instances (useful in tests)."""
    with _CACHE_LOCK:
        _BACKEND_CACHE.clear()


def clear_client_cache() -> None:
    """Drop all cached LLM client instances (useful in tests)."""
    with _CACHE_LOCK:
        _CLIENT_CACHE.clear()


def get_client(
    model_path: str,
    backend: Optional[str] = None,
    backend_instance: Optional[BaseBackend] = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> "LLM":
    """Return a cached shared :class:`LLM` for ``(model_path, backend, kwargs)``.

    This is what the module-level :func:`chat` / :func:`complete` helpers
    use so they don't reload the model on every call (the biggest v0.1.x
    performance flaw). Pass ``use_cache=False`` when you need isolation
    (e.g. in tests).
    """
    key = _cache_key(model_path, backend, kwargs)
    if use_cache:
        with _CACHE_LOCK:
            hit = _CLIENT_CACHE.get(key)
        if hit is not None:
            return hit
    client = LLM(
        model_path=model_path,
        backend=backend,
        backend_instance=backend_instance,
        use_cache=use_cache,
        **kwargs,
    )
    if use_cache and backend_instance is None:
        with _CACHE_LOCK:
            _CLIENT_CACHE.setdefault(key, client)
    return client


# ---------------------------------------------------------------------------
# Inline echo backend: guarantees backend="echo" works offline / in tests even
# if local_llm_kit/backends/echo.py is unavailable for any reason.
# ---------------------------------------------------------------------------
class _EchoBackend(BaseBackend):
    """Deterministic offline backend: echoes the prompt back."""

    backend_name = "echo"

    def __init__(self, model_path: str = "echo", echo_response: str = "echo", **kwargs: Any):
        self.model_path = model_path
        self.echo_response = echo_response

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        text = kwargs.get("echo_response", self.echo_response)
        return {"text": str(text), "finish_reason": "stop"}

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        text = str(kwargs.get("echo_response", self.echo_response))
        for word in text.split(" "):
            yield {"text": word + " "}
        # No trailing empty chunk; LLM adds the finish chunk.

    def get_context_window(self) -> int:
        return 4096

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    def embed(self, texts: Any, **kwargs: Any) -> List[List[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        vecs = []
        for t in items:
            h = abs(hash(t)) % 1000
            vecs.append([float((h + i * 37) % 100) / 100.0 for i in range(8)])
        return vecs


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _new_id(prefix: str) -> str:
    return "%s-%s" % (prefix, uuid.uuid4().hex[:24])


def _extract_text(result: Any) -> str:
    """Pull generated text out of the various shapes backends may return."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"]
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            msg = first.get("message", {}) if isinstance(first, dict) else {}
            if isinstance(msg.get("content"), str):
                return msg["content"] or ""
            if isinstance(first.get("text"), str):
                return first["text"]
    return str(result)


def _normalize_response_format(
    response_format: Optional[Union[str, Dict[str, Any]]],
    format: Optional[str],  # noqa: A002 (legacy name, kept for compat)
) -> Optional[Dict[str, Any]]:
    """Merge legacy ``format="json"`` and modern ``response_format``."""
    if response_format is None and format == "json":
        return {"type": "json_object"}
    if response_format is None:
        return None
    if isinstance(response_format, str):
        if response_format in ("json_object", "json"):
            return {"type": "json_object"}
        raise ValueError("Unsupported response_format string: %r" % (response_format,))
    rtype = response_format.get("type")
    if rtype in ("json_object", "json_schema"):
        return response_format
    if rtype == "text":
        return None
    raise ValueError("Unsupported response_format type: %r" % (rtype,))


def _forced_tool_name(
    tool_choice: Union[str, Dict[str, Any], None],
    function_call: Union[str, Dict[str, str]],
) -> Optional[str]:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", tool_choice)
        if isinstance(fn, dict) and fn.get("name"):
            return fn["name"]
    if isinstance(function_call, dict) and function_call.get("name"):
        return function_call["name"]
    return None


def _tool_names(tools_norm: List[Dict[str, Any]]) -> List[str]:
    names = []
    for t in tools_norm:
        try:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.append(name)
        except AttributeError:
            continue
    return names


def _required_params(tool_spec: Dict[str, Any]) -> List[str]:
    try:
        fn = tool_spec.get("function", tool_spec) if isinstance(tool_spec, dict) else {}
        params = fn.get("parameters", {}) if isinstance(fn, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []
        return [r for r in required if isinstance(r, str)]
    except AttributeError:
        return []


class LLM:
    """
    OpenAI-like interface for local language models.

    :param model_path: path, HF id, Ollama tag, URL, or "echo".
    :param backend: "auto" (default), "transformers", "llamacpp", "ollama",
        "openai-compat", or "echo".
    :param backend_instance: pre-built backend object (tests / DI). Skips the
        cache and auto-detection entirely.
    :param use_cache: reuse module-level cached backends for identical
        ``(model_path, backend, backend_kwargs)``. Set False in tests that
        need isolation.
    """

    def __init__(
        self,
        model_path: str,
        backend: Optional[str] = None,
        context_window: Optional[int] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        backend_kwargs: Optional[Dict[str, Any]] = None,
        backend_instance: Optional[BaseBackend] = None,
        use_cache: bool = True,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.backend_kwargs = dict(backend_kwargs or {})

        if backend_instance is not None:
            self.backend = backend_instance
            self._backend_name = getattr(backend_instance, "backend_name", "injected")
        else:
            key = _cache_key(model_path, backend, self.backend_kwargs)
            cached = None
            if use_cache:
                with _CACHE_LOCK:
                    cached = _BACKEND_CACHE.get(key)
            if cached is not None:
                self.backend = cached
                self._backend_name = (backend or "auto").lower()
            else:
                self.backend = self._init_backend(backend)
                self._backend_name = (backend or "auto").lower()
                if use_cache:
                    with _CACHE_LOCK:
                        _BACKEND_CACHE.setdefault(key, self.backend)

        if context_window:
            self.context_window = context_window
        else:
            try:
                self.context_window = self.backend.get_context_window()
            except Exception:
                self.context_window = 4096
            logger.info("Context window: %s", self.context_window)

        self.prompt_formatter = get_prompt_formatter(model_path)
        self.memory = MessageHistory(context_window=self.context_window)
        # ToolRegistry subclasses FunctionRegistry, so both APIs share one store.
        self.tool_registry = ToolRegistry()
        self.function_registry = self.tool_registry  # backwards-compat alias

    # -- backend setup ----------------------------------------------------
    def _init_backend(self, backend_name: Optional[str]) -> BaseBackend:
        name = (backend_name or "auto").lower()
        if name == "auto":
            lp = self.model_path.lower()
            if self.model_path.startswith(("http://", "https://")):
                name = "openai-compat"
            elif lp.startswith("ollama:"):
                name = "ollama"
            elif ".gguf" in lp or "ggml" in lp:
                name = "llamacpp"
            else:
                name = "transformers"
        if name not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "Unsupported backend %r; choose from %s" % (backend_name, list(_SUPPORTED_BACKENDS))
            )

        if name == "echo":
            try:
                from .backends.echo import EchoBackend  # type: ignore

                return EchoBackend(self.model_path, **self.backend_kwargs)
            except Exception:
                return _EchoBackend(self.model_path, **self.backend_kwargs)
        if name == "transformers":
            try:
                from .backends.transformers import TransformersBackend

                return TransformersBackend(self.model_path, **self.backend_kwargs)
            except ImportError as exc:
                raise RuntimeError(
                    "transformers backend needs torch+transformers "
                    "(pip install 'local-llm-kit[transformers]')."
                ) from exc
        if name == "llamacpp":
            try:
                from .backends.llamacpp import LlamaCppBackend

                return LlamaCppBackend(self.model_path, **self.backend_kwargs)
            except ImportError as exc:
                raise RuntimeError(
                    "llamacpp backend needs llama-cpp-python "
                    "(pip install 'local-llm-kit[llamacpp]')."
                ) from exc
        if name == "ollama":
            try:
                from .backends.ollama import OllamaBackend  # type: ignore

                return OllamaBackend(self.model_path, **self.backend_kwargs)
            except ImportError as exc:
                raise RuntimeError(
                    "ollama backend module missing; ensure a running "
                    "`ollama serve` (no extra package required)."
                ) from exc
        # openai-compat: stdlib-only backend module.
        try:
            from .backends.openai_compat import OpenAICompatBackend  # type: ignore

            return OpenAICompatBackend(self.model_path, **self.backend_kwargs)
        except ImportError as exc:
            raise RuntimeError(
                "openai-compat backend module missing " "(local_llm_kit/backends/openai_compat.py)."
            ) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    # -- registration -----------------------------------------------------
    def add_function(self, name: str, schema: Dict[str, Any], implementation: Callable) -> None:
        """Register a legacy function (also usable as a modern tool)."""
        self.tool_registry.add_function(name, schema, implementation)

    def add_tool(self, func: Callable, name: Optional[str] = None) -> Dict[str, Any]:
        """Register a Python callable as a modern tool; returns its tool spec."""
        return self.tool_registry.add_tool(func, name=name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return registered tools in OpenAI ``tools`` format."""
        return self.tool_registry.get_tool_list()

    # -- internal plumbing ------------------------------------------------
    def _now(self) -> int:
        try:
            return self.backend.get_timestamp()
        except Exception:
            return int(time.time())

    def _count(self, text: Any) -> int:
        try:
            return self.backend.count_tokens("" if text is None else str(text))
        except Exception:
            return max(1, len(str(text or "")) // 4)

    def _sampling_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_new_tokens": self.max_new_tokens if max_tokens is None else max_tokens,
            "top_p": self.top_p if top_p is None else top_p,
            "top_k": self.top_k if top_k is None else top_k,
            "repetition_penalty": (
                self.repetition_penalty if repetition_penalty is None else repetition_penalty
            ),
        }
        if seed is not None:
            params["seed"] = seed
        if stop is not None:
            params["stop"] = stop
        if extra:
            params.update(extra)
        return params

    def _format_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools_norm: Optional[List[Dict[str, Any]]],
        function_call: Union[str, Dict[str, str]],
        tool_choice: Union[str, Dict[str, Any], None],
        json_mode: bool,
    ) -> str:
        legacy_funcs = None
        if tools_norm:
            legacy_funcs = [
                t["function"] if isinstance(t, dict) and "function" in t else t for t in tools_norm
            ]
        fmt = self.prompt_formatter.format_messages
        try:
            return fmt(messages, functions=legacy_funcs, function_call=function_call, json_mode=json_mode, tools=tools_norm)  # type: ignore[call-arg]
        except TypeError:
            return fmt(
                messages, functions=legacy_funcs, function_call=function_call, json_mode=json_mode
            )

    def _backend_generate(
        self,
        full_messages: List[Dict[str, Any]],
        prompt: str,
        gen_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Vision passthrough: native ``backend.chat(messages)`` wins when the
        conversation carries images; otherwise plain prompt generation."""
        chat_fn = getattr(self.backend, "chat", None)
        if has_images(full_messages) and callable(chat_fn):
            try:
                params = {k: v for k, v in gen_params.items() if k != "stream"}
                return chat_fn(messages=full_messages, **params)
            except TypeError:
                pass
        flat = {k: v for k, v in gen_params.items() if k != "stream"}
        result = self.backend.generate(prompt, **flat)
        if isinstance(result, dict):
            return result
        return {"text": _extract_text(result)}

    def _parse_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls with a length cap (ReDoS mitigation)."""
        if not text:
            return []
        snippet = text if len(text) <= _PARSE_MAX_CHARS else text[:_PARSE_MAX_CHARS]
        try:
            return parse_tool_calls(snippet)
        except ValueError:
            return []

    def _execute_tool_call(
        self,
        call: ToolCall,
        allowed_names: Optional[List[str]] = None,
        spec: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = 10.0,
    ) -> str:
        """Execute one tool call with scope check, arg validation and timeout.

        Failures are returned as tool-output strings (generic, truncated —
        never tracebacks) so they can't leak secrets into chat history.
        """
        if allowed_names is not None and call.name not in allowed_names:
            return "Error: tool '%s' is not available for this request" % call.name
        if not self.tool_registry.has_function(call.name):
            return "Error: tool '%s' is not registered" % call.name
        if not isinstance(call.arguments, dict):
            return "Error: tool '%s' received invalid arguments (must be an object)" % call.name
        if spec is not None:
            missing = [r for r in _required_params(spec) if r not in call.arguments]
            if missing:
                return "Error: tool '%s' missing required argument(s): %s" % (
                    call.name,
                    ", ".join(missing),
                )
        try:
            impl = self.tool_registry.implementations[call.name]
        except (AttributeError, KeyError):
            return "Error: tool '%s' has no implementation" % call.name
        try:
            if timeout is None:
                result = impl(**call.arguments)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(impl, **call.arguments)
                    result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "Error: tool '%s' timed out after %s seconds" % (call.name, timeout)
        except Exception as exc:  # noqa: BLE001 (tool errors become tool output)
            logger.warning("Tool '%s' failed: %s", call.name, exc)
            return "Error: tool '%s' failed (%s: %s)" % (
                call.name,
                type(exc).__name__,
                str(exc)[:200],
            )
        return result if isinstance(result, str) else json.dumps(result)

    def _ensure_json_output(self, text: str) -> str:
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        match = re.search(r"(\{|\[).*?(\}|\])", text, re.DOTALL)
        if match:
            try:
                candidate = match.group(0)
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        correction = (
            "The previous response was not valid JSON. "
            "Respond with only a valid JSON object or array.\n\nPrevious invalid response:\n"
            + text
            + "\n\nValid JSON response:"
        )
        try:
            retry = self.backend.generate(
                correction, temperature=0.2, max_new_tokens=self.max_new_tokens
            )
            retry_text = _extract_text(retry)
            json.loads(retry_text)
            return retry_text
        except Exception:
            return (
                '{"error": "Failed to generate valid JSON", "attempted_response": '
                + json.dumps(text)
                + "}"
            )

    def _usage(self, prompt: str, text: Any) -> Dict[str, int]:
        pt = self._count(prompt)
        ct = self._count(text)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

    # -- chat -------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        format: Optional[str] = None,  # noqa: A002 (legacy name)
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Union[str, Dict[str, Any], None] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_iterations: int = 5,
        execute_tools: bool = True,
        tool_timeout: Optional[float] = 10.0,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Generate a chat completion.

        Modern parameters (``tools``/``tool_choice``/``response_format``)
        are preferred; legacy ``functions``/``function_call``/``format``
        keep working and are normalized internally.

        :param execute_tools: when True (default, v0.1.x behaviour), parsed
            tool calls are executed (scoped + validated + timed out) and the
            model is re-queried with the results. When False, tool calls are
            returned unexecuted with ``finish_reason="tool_calls"`` — the
            HTTP server uses this so remote callers can't trigger local
            function execution unless explicitly enabled.
        :param tool_timeout: per-tool execution timeout in seconds
            (None = no timeout).
        :param max_iterations: maximum tool-execution rounds.
        """
        # Register per-call callables so the loop can execute them.
        for item in tools or []:
            if callable(item) and not isinstance(item, dict):
                try:
                    self.tool_registry.add_tool(item)
                except Exception:
                    logger.debug("Could not auto-register tool %r", item, exc_info=True)
        tools_norm = normalize_tools(tools, functions)
        if functions is None and not tools and self.tool_registry.has_functions():
            tools_norm = self.tool_registry.get_tool_list()

        allow_tools = bool(tools_norm) and should_call_tools(tool_choice, function_call)
        active_tools = tools_norm if allow_tools else []
        forced_name = _forced_tool_name(tool_choice, function_call)
        rf = _normalize_response_format(response_format, format)
        json_mode = rf is not None
        legacy_path = (
            tools is None and tool_choice is None
        )  # old-style call -> legacy finish_reason/keys

        history = list(messages)  # vision parts (lists) preserved as-is
        self.memory.add_messages(history)

        gen_params = self._sampling_params(
            temperature, max_tokens, top_p, top_k, repetition_penalty, seed, stop, kwargs
        )
        gen_params["logprobs"] = logprobs
        gen_params["top_logprobs"] = (
            (top_logprobs if top_logprobs is not None else 5) if logprobs else None
        )

        loop_kwargs = {
            "forced_name": forced_name,
            "function_call": function_call,
            "tool_choice": tool_choice,
            "json_mode": json_mode,
            "legacy_path": legacy_path,
            "max_iterations": max(0, max_iterations),
            "execute_tools": execute_tools,
            "tool_timeout": tool_timeout,
        }
        if stream:
            return self._chat_streaming(history, active_tools, gen_params, **loop_kwargs)
        return self._chat_once(history, active_tools, gen_params, **loop_kwargs)

    def _tool_specs_by_name(
        self, active_tools: List[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        names = _tool_names(active_tools)
        specs = {}
        for t in active_tools:
            try:
                fn = t.get("function", t) if isinstance(t, dict) else {}
                if isinstance(fn, dict) and fn.get("name"):
                    specs[fn["name"]] = t
            except AttributeError:
                continue
        return names, specs

    def _chat_once(
        self,
        history: List[Dict[str, Any]],
        active_tools: List[Dict[str, Any]],
        gen_params: Dict[str, Any],
        forced_name: Optional[str] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        tool_choice: Union[str, Dict[str, Any], None] = None,
        json_mode: bool = False,
        legacy_path: bool = True,
        max_iterations: int = 5,
        execute_tools: bool = True,
        tool_timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        working = list(history)
        created = self._now()
        allowed_names, specs_by_name = self._tool_specs_by_name(active_tools)
        prompt = ""
        text = ""
        pending: List[ToolCall] = []
        completion_parts: List[str] = []
        finish = "stop"

        for turn in range(max_iterations + 1):
            prompt = self._format_prompt(
                working, active_tools or None, function_call, tool_choice, json_mode
            )
            result = self._backend_generate(working, prompt, gen_params)
            text = _extract_text(result)
            if json_mode:
                text = self._ensure_json_output(text)
            completion_parts.append(text or "")
            finish = result.get("finish_reason", "stop") if isinstance(result, dict) else "stop"

            calls: List[ToolCall] = []
            if active_tools:
                calls = self._parse_calls(text or "")
                if forced_name:
                    calls = [c for c in calls if c.name == forced_name]
            if not calls:
                pending = []
                finish = "stop" if finish not in ("length",) else finish
                break
            pending = calls
            if not execute_tools or turn >= max_iterations:
                # Report calls without executing (server-safe mode or budget spent).
                finish = "function_call" if (legacy_path and len(calls) == 1) else "tool_calls"
                break
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [c.to_dict() for c in calls],
            }
            if len(calls) == 1:
                assistant_msg["function_call"] = {
                    "name": calls[0].name,
                    "arguments": json.dumps(calls[0].arguments),
                }
            working.append(assistant_msg)
            self.memory.add_messages([assistant_msg])
            for call in calls:
                content = self._execute_tool_call(
                    call,
                    allowed_names,
                    specs_by_name.get(call.name),
                    tool_timeout,
                )
                tool_msg: Dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": content,
                }
                working.append(tool_msg)
                self.memory.add_messages([tool_msg])
            pending = []
            text = ""

        if pending:
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [c.to_dict() for c in pending],
            }
            if len(pending) == 1:
                message["function_call"] = {
                    "name": pending[0].name,
                    "arguments": json.dumps(pending[0].arguments),
                }
            self.memory.add_messages([message])
        else:
            message = {"role": "assistant", "content": text}
            self.memory.add_messages([message])

        response: Dict[str, Any] = {
            "id": _new_id("chatcmpl"),
            "object": "chat.completion",
            "created": created,
            "model": self.model_path,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        if gen_params.get("logprobs"):
            response["choices"][0]["logprobs"] = {}
        response["usage"] = self._usage(prompt, "".join(completion_parts) if pending else text)
        return response

    def _chat_streaming(
        self,
        history: List[Dict[str, Any]],
        active_tools: List[Dict[str, Any]],
        gen_params: Dict[str, Any],
        forced_name: Optional[str] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        tool_choice: Union[str, Dict[str, Any], None] = None,
        json_mode: bool = False,
        legacy_path: bool = True,
        max_iterations: int = 5,
        execute_tools: bool = True,
        tool_timeout: Optional[float] = 10.0,
    ) -> Iterator[Dict[str, Any]]:
        """Stream OpenAI-style chunks. Multi-turn: each tool-loop iteration
        streams its own deltas (single-pass when no tools are involved)."""
        response_id = _new_id("chatcmpl")
        created = self._now()
        working = list(history)
        allowed_names, specs_by_name = self._tool_specs_by_name(active_tools)

        for turn in range(max_iterations + 1):
            prompt = self._format_prompt(
                working, active_tools or None, function_call, tool_choice, json_mode
            )
            stream_params = dict(gen_params)
            accumulated = ""
            first = True
            try:
                chunk_iter = self.backend.generate_stream(prompt, **stream_params)
            except Exception:
                result = self._backend_generate(working, prompt, gen_params)
                chunk_iter = iter([{"text": _extract_text(result)}])
            for chunk in chunk_iter:
                piece = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                accumulated += piece
                delta: Dict[str, Any] = {"content": piece}
                if first:
                    delta["role"] = "assistant"
                    first = False
                out: Dict[str, Any] = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_path,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                if gen_params.get("logprobs") and isinstance(chunk, dict) and "logprobs" in chunk:
                    out["choices"][0]["logprobs"] = chunk["logprobs"]
                yield out

            raw_text = accumulated
            calls: List[ToolCall] = []
            if active_tools:
                check_text = self._ensure_json_output(raw_text) if json_mode else raw_text
                calls = self._parse_calls(check_text)
                if forced_name:
                    calls = [c for c in calls if c.name == forced_name]
            if not calls or not execute_tools or turn >= max_iterations:
                if calls:  # surface the calls (unexecuted in server-safe mode)
                    delta2: Dict[str, Any] = {
                        "content": None,
                        "tool_calls": [c.to_dict() for c in calls],
                    }
                    if legacy_path and len(calls) == 1:
                        delta2["function_call"] = {
                            "name": calls[0].name,
                            "arguments": json.dumps(calls[0].arguments),
                        }
                    yield {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_path,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta2,
                                "finish_reason": (
                                    "function_call"
                                    if (legacy_path and len(calls) == 1)
                                    else "tool_calls"
                                ),
                            }
                        ],
                    }
                    return
                yield {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_path,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                return
            # Execute and continue streaming the follow-up turn.
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [c.to_dict() for c in calls],
            }
            working.append(assistant_msg)
            self.memory.add_messages([assistant_msg])
            for call in calls:
                content = self._execute_tool_call(
                    call,
                    allowed_names,
                    specs_by_name.get(call.name),
                    tool_timeout,
                )
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": content,
                }
                working.append(tool_msg)
                self.memory.add_messages([tool_msg])

    # -- completion -------------------------------------------------------
    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        format: Optional[str] = None,  # noqa: A002 (legacy name)
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        rf = _normalize_response_format(response_format, format)
        if rf is not None:
            prompt = "You must respond with valid JSON only, no other text.\n\n" + prompt
            if rf.get("type") == "json_schema":
                schema = rf.get("json_schema", rf.get("schema", {}))
                prompt += "\n\nJSON Schema:\n" + json.dumps(schema)
        gen_params = self._sampling_params(
            temperature, max_tokens, top_p, top_k, repetition_penalty, seed, stop, kwargs
        )
        gen_params["logprobs"] = logprobs
        gen_params["top_logprobs"] = (
            (top_logprobs if top_logprobs is not None else 5) if logprobs else None
        )
        if stream:
            return self._complete_streaming(prompt, gen_params)
        result = self.backend.generate(
            prompt, **{k: v for k, v in gen_params.items() if k != "stream"}
        )
        text = _extract_text(result)
        if rf is not None:
            text = self._ensure_json_output(text)
        response: Dict[str, Any] = {
            "id": _new_id("cmpl"),
            "object": "text_completion",
            "created": self._now(),
            "model": self.model_path,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": (
                        result.get("finish_reason", "stop") if isinstance(result, dict) else "stop"
                    ),
                }
            ],
        }
        if logprobs:
            response["choices"][0]["logprobs"] = (
                result.get("logprobs", {}) if isinstance(result, dict) else {}
            )
        response["usage"] = self._usage(prompt, text)
        return response

    def _complete_streaming(
        self, prompt: str, gen_params: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        response_id = _new_id("cmpl")
        created = self._now()
        for chunk in self.backend.generate_stream(prompt, **gen_params):
            piece = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            out: Dict[str, Any] = {
                "id": response_id,
                "object": "text_completion.chunk",
                "created": created,
                "model": self.model_path,
                "choices": [{"text": piece, "index": 0, "finish_reason": None}],
            }
            if gen_params.get("logprobs") and isinstance(chunk, dict) and "logprobs" in chunk:
                out["choices"][0]["logprobs"] = chunk["logprobs"]
            yield out
        yield {
            "id": response_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": self.model_path,
            "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
        }

    # -- embeddings -------------------------------------------------------
    def embed(
        self, input: Union[str, List[str]], **kwargs: Any
    ) -> Dict[str, Any]:  # noqa: A002 (OpenAI field name)
        """OpenAI-shaped embeddings. Delegates to ``backend.embed`` (ollama /
        openai-compat / echo); raises a helpful error otherwise."""
        texts = [input] if isinstance(input, str) else list(input)
        vectors: Any = None
        for attr in ("embed", "embeddings", "get_embeddings"):
            fn = getattr(self.backend, attr, None)
            if callable(fn):
                try:
                    vectors = fn(texts, **kwargs)
                except TypeError:
                    vectors = fn(texts)
                break
        if vectors is None:
            raise RuntimeError(
                "Backend %r does not support embeddings. Use backend='ollama' "
                "or backend='openai-compat' (or inject a backend with .embed)."
                % type(self.backend).__name__
            )
        if isinstance(vectors, dict):  # already OpenAI-shaped
            data = vectors.get("data", [])
            items = [d.get("embedding", []) for d in data] if data else []
        elif isinstance(vectors, list) and vectors and isinstance(vectors[0], dict):
            items = [v.get("embedding", v) for v in vectors]
        else:
            items = list(vectors)
        return {
            "object": "list",
            "model": self.model_path,
            "data": [
                {"object": "embedding", "index": i, "embedding": list(vec)}
                for i, vec in enumerate(items)
            ],
            "usage": {
                "prompt_tokens": sum(self._count(t) for t in texts),
                "total_tokens": sum(self._count(t) for t in texts),
            },
        }

    # -- async (via asyncio.to_thread; Py3.9+) ------------------------------
    def achat(self, *args: Any, **kwargs: Any):
        """Async chat. ``await llm.achat(...)``; ``async for`` when stream=True."""
        if kwargs.get("stream"):
            return self._achat_stream(*args, **kwargs)
        return self._achat_once(*args, **kwargs)

    async def _achat_once(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("stream", None)
        return await asyncio.to_thread(self.chat, *args, **kwargs)

    async def _achat_stream(self, *args: Any, **kwargs: Any):
        gen = self.chat(*args, **kwargs)
        for chunk in gen:
            await asyncio.sleep(0)
            yield chunk

    def acomplete(self, *args: Any, **kwargs: Any):
        """Async completion. ``await llm.acomplete(...)``; ``async for`` when stream=True."""
        if kwargs.get("stream"):
            return self._acomplete_stream(*args, **kwargs)
        return self._acomplete_once(*args, **kwargs)

    async def _acomplete_once(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("stream", None)
        return await asyncio.to_thread(self.complete, *args, **kwargs)

    async def _acomplete_stream(self, *args: Any, **kwargs: Any):
        gen = self.complete(*args, **kwargs)
        for chunk in gen:
            await asyncio.sleep(0)
            yield chunk

    def __repr__(self) -> str:
        return "LLM(model_path=%r, backend=%s)" % (self.model_path, type(self.backend).__name__)
