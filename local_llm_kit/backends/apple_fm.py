"""
Apple Foundation Models backend (on-device system model).

Uses Apple's official ``apple-fm-sdk`` (Python bindings for the Foundation
Models framework) to run inference on the on-device model at the core of
Apple Intelligence. Prompts and responses stay on the Mac; no API key, no
network, no model downloads.

Requirements (verified against ``apple/apple-fm-sdk`` and PyPI 2026-10-16):

* macOS 26.0+
* Apple Intelligence-capable Mac with Apple Intelligence turned on
  (https://support.apple.com/en-us/121115)
* Xcode 26.0+ with the Xcode/SDK agreement accepted (the SDK builds native
  bindings against the macOS SDK)
* Python 3.10+
* ``pip install "local-llm-kit[apple-fm]"`` (installs ``apple-fm-sdk>=0.2.1``)

Design notes:

* Stdlib-only at import time. ``apple_fm_sdk`` is imported lazily inside
  :meth:`AppleFMBackend.__init__` so the core install stays light and
  non-Mac platforms import this module cleanly.
* One fresh :class:`LanguageModelSession` per request. Backend instances are
  shared through the module-level cache in :mod:`local_llm_kit.llm`, so a
  long-lived session would leak transcripts across unrelated ``LLM`` calls.
  Per-request sessions keep calls isolated; the system model weights stay
  loaded by the OS either way.
* ``supports_native_chat`` is True because chat history is passed through
  (system messages become session instructions), but tool execution stays in
  the client-side loop: tools are described in the prompt text and parsed
  from the reply, exactly like the other prompt-based backends. The SDK's
  automatic in-framework tool execution is deliberately not used so the
  security model (scoped, validated, time-bounded, opt-in on the server)
  stays uniform across backends.
* Structured output uses the SDK's native ``json_schema`` guided generation
  when ``response_format={"type": "json_schema", ...}`` is given. Plain
  ``json_object`` requests use the prompt/repair path shared with the other
  backends.
* Streaming yields text deltas. The SDK's ``stream_response`` yields
  full-text snapshots (not deltas), so this backend diffs consecutive
  snapshots. Guided (schema) responses do not stream server-side either;
  they are returned as one completed chunk.
* Embeddings are not available on-device (``apfel --serve`` likewise
  answers ``POST /v1/embeddings`` with 501); :meth:`embed` is not
  implemented and raises :exc:`NotImplementedError`.
* Vision parts are degraded to ``[image]`` placeholders. The SDK takes image
  *files* (``ImageAttachment(path)``), not the base64 data URLs used by the
  OpenAI/HTTP adapters, so wiring them would require materialising uploads
  to disk first. Use ``backend="apfel"`` on macOS 27 (or an HTTP vision
  model) for image input.
"""

import asyncio
import queue
import sys
import threading
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseBackend

#: Model id reported in responses. Matches the id served by
#: ``apfel --serve`` (https://github.com/Arthur-Ficial/apfel), so configs
#: can switch between the native backend and the bridge without edits.
APPLE_MODEL_ID = "apple-foundationmodel"

#: Fallback window when the runtime value cannot be read. The documented
#: live sizes are 4096 tokens (macOS 26, or M1/M2 Macs) and 8192 tokens
#: (macOS 27 on M3+ with 12GB+), read at runtime via
#: ``SystemLanguageModel.context_size``.
_FALLBACK_CONTEXT_WINDOW = 4096

_MISSING_SDK_HINT = (
    "The apple-fm backend needs Apple's Foundation Models SDK for Python: "
    'pip install "local-llm-kit[apple-fm]". Requires macOS 26+, Xcode 26+ '
    "(SDK agreement accepted), Python 3.10+, and Apple Intelligence turned on: "
    "https://support.apple.com/en-us/121115"
)

#: Actionable hints per ``SystemLanguageModelUnavailableReason``.
_REASON_HINTS = {
    "APPLE_INTELLIGENCE_NOT_ENABLED": (
        "Apple Intelligence is not enabled. Turn it on in System Settings and "
        "wait for the on-device model to finish downloading "
        "(https://support.apple.com/en-us/121115)."
    ),
    "DEVICE_NOT_ELIGIBLE": (
        "This Mac is not eligible for Apple Foundation Models (requires an "
        "Apple Intelligence-capable Mac)."
    ),
    "MODEL_NOT_READY": (
        "The on-device model is still downloading or preparing. Try again "
        "after Apple Intelligence finishes setting up."
    ),
}


def _reason_text(reason: Any) -> str:
    name = getattr(reason, "name", None) or str(reason)
    hint = _REASON_HINTS.get(str(name), "")
    if hint:
        return "%s %s" % (name, hint)
    return str(name)


class AppleFMBackend(BaseBackend):
    """On-device Apple Foundation Models backend (async SDK, sync facade).

    Import-safe everywhere: importing this module never requires macOS or
    the SDK. Instantiation validates the platform, the interpreter, the
    SDK presence, and model availability, in that order, with actionable
    errors.
    """

    backend_name = "apple-fm"
    supports_vision = False
    supports_embeddings = False
    supports_native_chat = True

    def __init__(
        self,
        model: str = APPLE_MODEL_ID,
        use_case: Any = "general",
        guardrails: Any = "default",
        context_window: Optional[int] = None,
        instructions: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Args:
        model: Model id reported in responses (informational: the device
            exposes exactly one system model; defaults to
            ``"apple-foundationmodel"``).
        use_case: ``"general"`` (default) or ``"content_tagging"``; also
            accepts the SDK's ``SystemLanguageModelUseCase`` enum.
        guardrails: ``"default"`` or ``"permissive_content_transformations"``;
            also accepts the SDK's ``SystemLanguageModelGuardrails`` enum.
        context_window: Override the runtime ``context_size`` value.
        instructions: Persona instructions prepended to every session
            (per-call ``system`` messages are appended after these).
        timeout: Seconds to wait for a generation before raising
            :exc:`TimeoutError` (``None`` waits indefinitely, matching the
            in-process backends).
        """
        if sys.platform != "darwin":
            raise RuntimeError(
                "The apple-fm backend runs only on macOS (this platform is %r). "
                "Apple's SDK has no Linux/Windows build." % (sys.platform,)
            )
        if sys.version_info < (3, 10):
            raise RuntimeError(
                "The apple-fm backend needs Python 3.10+ (running %s); "
                "apple-fm-sdk declares requires-python >= 3.10." % (sys.version.split()[0],)
            )
        try:
            import apple_fm_sdk as fm  # type: ignore
        except ImportError as exc:
            raise ImportError(_MISSING_SDK_HINT) from exc
        self._fm = fm

        self.model = model or APPLE_MODEL_ID
        self.model_path = self.model
        self.timeout = timeout
        self.base_instructions = instructions
        self.use_case = self._normalize_use_case(use_case)
        self.guardrails = self._normalize_guardrails(guardrails)

        system_model = fm.SystemLanguageModel(use_case=self.use_case, guardrails=self.guardrails)
        available, reason = system_model.is_available()
        if not available:
            raise RuntimeError("Apple Foundation Models are unavailable: %s" % _reason_text(reason))
        self._system_model = system_model

        if context_window is not None:
            self._context_window = int(context_window)
        else:
            try:
                self._context_window = int(system_model.context_size)
            except Exception:
                self._context_window = _FALLBACK_CONTEXT_WINDOW

    # -- availability preflight ----------------------------------------

    @staticmethod
    def is_available() -> Any:
        """Return ``(True, None)`` when the system model is usable.

        Never raises: returns ``(False, reason-string)`` when the platform,
        interpreter, SDK, or model is unsuitable. Useful as a CLI/example
        preflight before constructing the backend (which raises).
        """
        if sys.platform != "darwin":
            return False, "requires macOS (running on %s)" % sys.platform
        if sys.version_info < (3, 10):
            return False, "requires Python 3.10+"
        try:
            import apple_fm_sdk as fm  # type: ignore
        except ImportError:
            return False, _MISSING_SDK_HINT
        try:
            available, reason = fm.SystemLanguageModel().is_available()
        except Exception as exc:  # e.g. bindings built without the macOS 26 SDK
            return False, "availability check failed: %s" % exc
        if available:
            return True, None
        return False, _reason_text(reason)

    # -- option normalisation ------------------------------------------

    def _normalize_use_case(self, use_case: Any) -> Any:
        fm = self._fm
        enum = fm.SystemLanguageModelUseCase
        if isinstance(use_case, enum):
            return use_case
        key = str(use_case).strip().lower().replace("-", "_")
        mapping = {"general": enum.GENERAL, "content_tagging": enum.CONTENT_TAGGING}
        if key not in mapping:
            raise ValueError(
                "Unknown use_case %r; choose 'general' or 'content_tagging'." % (use_case,)
            )
        return mapping[key]

    def _normalize_guardrails(self, guardrails: Any) -> Any:
        fm = self._fm
        enum = fm.SystemLanguageModelGuardrails
        if isinstance(guardrails, enum):
            return guardrails
        key = str(guardrails).strip().lower().replace("-", "_")
        mapping = {
            "default": enum.DEFAULT,
            "permissive_content_transformations": enum.PERMISSIVE_CONTENT_TRANSFORMATIONS,
            "permissive": enum.PERMISSIVE_CONTENT_TRANSFORMATIONS,
        }
        if key not in mapping:
            raise ValueError(
                "Unknown guardrails %r; choose 'default' or "
                "'permissive_content_transformations'." % (guardrails,)
            )
        return mapping[key]

    def _options(
        self,
        temperature: float,
        max_new_tokens: int,
        seed: Optional[int] = None,
    ) -> Any:
        """Map generic sampling controls onto ``GenerationOptions``.

        Only knobs the SDK exposes are set: ``temperature`` always,
        ``maximum_response_tokens`` always (the SDK warns strict caps can
        malform output, so callers should prefer generous values),
        greedy sampling for ``temperature == 0``, and seeded random
        sampling when ``seed`` is given. ``top_p``/``top_k``/penalties have
        no SDK equivalent and are documented as ignored.
        """
        fm = self._fm
        sampling = None
        if temperature == 0:
            sampling = fm.SamplingMode.greedy()
        elif seed is not None:
            sampling = fm.SamplingMode.random(seed=int(seed))
        return fm.GenerationOptions(
            temperature=temperature,
            sampling=sampling,
            maximum_response_tokens=int(max_new_tokens),
        )

    # -- sync bridge for the async-only SDK -----------------------------

    def _run(self, coro_factory: Any) -> Any:
        """Await a coroutine factory from synchronous code.

        Uses ``asyncio.run`` when no loop is running in this thread
        (the common path, including ``asyncio.to_thread`` workers). When
        called from inside a running loop, the work moves to a helper
        thread with its own loop so the caller's loop is never blocked
        or re-entered.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())

        box: Dict[str, Any] = {}

        def _target() -> None:
            try:
                box["result"] = asyncio.run(coro_factory())
            except BaseException as exc:  # re-raised in the caller thread
                box["error"] = exc

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise TimeoutError(
                "Apple Foundation Models request timed out after %s seconds." % (self.timeout,)
            )
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _new_session(self, instructions: Optional[str]) -> Any:
        combined = (
            "\n\n".join(part for part in (self.base_instructions, instructions) if part) or None
        )
        return self._fm.LanguageModelSession(instructions=combined, model=self._system_model)

    # -- prompt API -----------------------------------------------------

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
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if stream:
            text = "".join(
                chunk["text"]
                for chunk in self.generate_stream(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
            )
            return {"text": text, "finish_reason": "stop", "model": self.model}
        options = self._options(temperature, max_new_tokens, kwargs.get("seed"))
        stop = kwargs.get("stop")
        session = self._new_session(None)

        async def _call() -> Any:
            return await session.respond(prompt, options=options)

        text = self._run(_call)
        if not isinstance(text, str):
            text = str(text)
        text = _apply_stop(text, stop)
        return {"text": text, "finish_reason": "stop", "model": self.model}

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
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        options = self._options(temperature, max_new_tokens, kwargs.get("seed"))
        stop = kwargs.get("stop")
        for snapshot in self._stream_snapshots(prompt, options):
            yield {"text": snapshot}
        if stop:
            # Stop sequences truncate the assembled text; per-snapshot
            # truncation would corrupt the delta stream, so callers that
            # need early stopping should use generate().
            pass

    # -- chat API (native sessions) ------------------------------------

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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if stream:
            chunks = list(
                self.chat_stream(
                    messages,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    seed=seed,
                    **kwargs,
                )
            )
            text = "".join(chunk.get("text", "") for chunk in chunks)
            finish = next(
                (c["finish_reason"] for c in reversed(chunks) if c.get("finish_reason")),
                "stop",
            )
            return {"text": text, "finish_reason": finish, "model": self.model}
        prompt, instructions = self._chat_prompt(messages, tools)
        options = self._options(temperature, max_new_tokens, seed)
        session = self._new_session(instructions)
        schema = _json_schema_from_response_format(response_format)

        async def _call() -> Any:
            if schema is not None:
                return await session.respond(prompt, json_schema=schema, options=options)
            return await session.respond(prompt, options=options)

        result = self._run(_call)
        text = _generated_content_to_text(result)
        text = _apply_stop(text, stop)
        return {"text": text, "finish_reason": "stop", "model": self.model}

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        prompt, instructions = self._chat_prompt(messages, tools)
        options = self._options(temperature, max_new_tokens, seed)
        if _json_schema_from_response_format(response_format) is not None:
            # The SDK does not stream guided generation; emit one chunk.
            result = self.chat(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                seed=seed,
                **kwargs,
            )
            yield {"text": result["text"], "finish_reason": result["finish_reason"]}
            return
        session_instructions = instructions
        for snapshot in self._stream_snapshots(prompt, options, session_instructions):
            yield {"text": snapshot}

    def _chat_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """Split messages into (prompt text, session instructions).

        System messages become session instructions (the SDK's native slot
        for them). Everything else is flattened to ``Role: text`` lines so
        the stateless-per-request session still sees the full history.
        Tool specs are appended as a JSON block so the model can emit the
        text-format tool calls the client-side loop parses.
        """
        system_parts = []
        rest = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                text = self.message_text(message)
                if text:
                    system_parts.append(text)
            else:
                rest.append(message)
        prompt = self.messages_to_prompt(rest)
        if tools:
            prompt += "\n\nAvailable tools (JSON):\n" + _tools_documentation(tools)
        instructions = "\n\n".join(system_parts) or None
        return prompt, instructions

    def _stream_snapshots(
        self,
        prompt: str,
        options: Any,
        instructions: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield text *deltas* from the SDK's full-text snapshots.

        The async iteration runs on a worker thread (the SDK streams via a
        background callback); snapshots cross over a queue and the caller
        thread diffs them into deltas.
        """
        inbox: "queue.Queue[Any]" = queue.Queue()
        sentinel = object()

        def _target() -> None:
            async def _consume() -> None:
                session = self._new_session(instructions)
                async for snapshot in session.stream_response(prompt, options):
                    inbox.put(str(snapshot))

            try:
                asyncio.run(_consume())
            except BaseException as exc:  # re-raised in the consumer thread
                inbox.put(exc)
            finally:
                inbox.put(sentinel)

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        previous = ""
        while True:
            item = inbox.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                worker.join()
                raise item
            snapshot = str(item)
            if snapshot.startswith(previous):
                delta = snapshot[len(previous) :]
            else:
                # Snapshot revised earlier text; re-emit rather than drop.
                delta = snapshot
            previous = snapshot
            if delta:
                yield delta
        worker.join(self.timeout)
        if worker.is_alive():
            raise TimeoutError(
                "Apple Foundation Models stream timed out after %s seconds." % (self.timeout,)
            )

    # -- introspection --------------------------------------------------

    def get_context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        """~4 chars/token heuristic (the SDK's exact ``token_count`` is async
        and would add a round trip per call; use the SDK directly for exact
        budgeting against ``context_size``)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def model_info(self) -> Dict[str, Any]:
        info = super().model_info()
        info.update(
            {
                "on_device": True,
                "requires_network": False,
                "use_case": _enum_name(self.use_case),
                "guardrails": _enum_name(self.guardrails),
                "variant": self._variant_name(),
            }
        )
        return info

    def _variant_name(self) -> Optional[str]:
        try:
            variant = self._system_model.variant
        except Exception:
            return None
        if variant is None:
            return None
        return getattr(variant, "display_name", None) or str(variant)

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": self.model, "object": "model", "owned_by": "apple"}]


# -- helpers ------------------------------------------------------------


def _apply_stop(text: str, stop: Any) -> str:
    if not stop:
        return text
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = len(text)
    for marker in stops:
        if not isinstance(marker, str) or not marker:
            continue
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


def _json_schema_from_response_format(response_format: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    schema = response_format.get("json_schema", {})
    if isinstance(schema, dict):
        inner = schema.get("schema", schema)
        if isinstance(inner, dict):
            return inner
    return None


def _generated_content_to_text(result: Any) -> str:
    """Best-effort conversion of an SDK response to text.

    Plain ``respond()`` returns ``str``; ``json_schema`` guided generation
    returns ``GeneratedContent`` (verified: ``to_json() -> str`` in
    ``apple-fm-sdk`` 0.2.x ``generable.py``).
    """
    if isinstance(result, str):
        return result
    to_json = getattr(result, "to_json", None)
    if callable(to_json):
        try:
            text = to_json()
            if isinstance(text, str):
                return text
        except Exception:
            pass
    return str(result)


def _tools_documentation(tools: List[Dict[str, Any]]) -> str:
    import json as _json

    specs = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        specs.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    try:
        return _json.dumps(specs)
    except (TypeError, ValueError):
        return str(specs)


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    text = getattr(value, "value", value)
    return str(text).lower()
