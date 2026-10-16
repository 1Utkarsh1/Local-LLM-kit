"""
Apple Foundation Models backend for on-device Apple Intelligence.

Provides access to the built-in system foundation model (AFM-on-device 3B)
available on Apple Silicon (M-series / A17 Pro+ / A18 Pro+) running macOS 26+.

Inference is 100% on-device (zero network calls, zero model weights to download).

Dependencies:
    Requires the optional ``apple-fm`` extra (``pip install "local-llm-kit[apple-fm]"``)
    which installs ``apple-fm-sdk``. Heavy imports are lazy inside methods.
"""

from __future__ import annotations

import asyncio
import platform
import sys
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .base import BaseBackend

_APPLE_FM_MIN_PYTHON = (3, 10)
_DEFAULT_CONTEXT_FLOOR = 4096


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously, handling nested or existing event loops safely."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Running inside an active loop (e.g. Jupyter or async runner); run in worker thread
        result: List[Any] = []
        error: List[BaseException] = []

        def _worker() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result.append(new_loop.run_until_complete(coro))
            except BaseException as exc:
                error.append(exc)
            finally:
                new_loop.close()

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0]
    else:
        return asyncio.run(coro)


class AppleFMBackend(BaseBackend):
    """Native backend for Apple's built-in Foundation Models (Apple Intelligence)."""

    backend_name = "apple-fm"
    supports_vision = True
    supports_embeddings = False
    supports_native_chat = True

    def __init__(
        self,
        model_path: str = "default",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Apple Foundation Models backend.

        Args:
            model_path: Ignored or 'default'; Apple provides a single built-in system model.
            system_prompt: Optional default system instructions.
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            **kwargs: Extra parameters passed to the model or session.
        """
        self.model_path = model_path or "default"
        self.system_prompt = system_prompt
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.extra_kwargs = kwargs
        self._sdk: Any = None
        self._model: Any = None
        self._session: Any = None

        self._check_platform_prerequisites()

    @classmethod
    def _check_platform_prerequisites(cls) -> None:
        """Validate platform OS and architecture requirements."""
        if sys.platform != "darwin":
            raise RuntimeError(
                "The 'apple-fm' backend is only supported on macOS (detected: %s). "
                "For other platforms, use 'llamacpp', 'transformers', or 'ollama'." % sys.platform
            )
        machine = platform.machine().lower()
        if machine not in ("arm64", "aarch64"):
            raise RuntimeError(
                "The 'apple-fm' backend requires Apple Silicon (arm64). "
                "Intel Macs do not support Apple Intelligence on-device foundation models."
            )
        if sys.version_info < _APPLE_FM_MIN_PYTHON:
            raise RuntimeError(
                "The 'apple-fm' native backend requires Python 3.10+ (current: %s). "
                "On Python 3.9, you can run Apfel as a bridge and connect via "
                "LLM(backend='apfel') or LLM(backend='openai-compat', "
                "backend_kwargs={'base_url': 'http://127.0.0.1:11434/v1'})."
                % sys.version.split()[0]
            )

    def _get_sdk(self) -> Any:
        """Lazily import the apple-fm-sdk package."""
        if self._sdk is None:
            self._check_platform_prerequisites()
            try:
                import apple_fm_sdk as fm

                self._sdk = fm
            except ImportError as exc:
                raise ImportError(
                    "The 'apple-fm' backend requires the 'apple-fm-sdk' package. "
                    'Install it with: pip install "local-llm-kit[apple-fm]" '
                    "(or pip install apple-fm-sdk)."
                ) from exc
        return self._sdk

    def _get_model(self) -> Any:
        """Get or initialize the SystemLanguageModel instance and verify availability."""
        if self._model is None:
            fm = self._get_sdk()
            model = fm.SystemLanguageModel()
            is_available, reason = model.is_available()
            if not is_available:
                raise RuntimeError(
                    "Apple Foundation Model is currently unavailable: %s. "
                    "Please ensure Apple Intelligence is enabled in macOS System Settings "
                    "and that model assets are fully downloaded." % (reason,)
                )
            self._model = model
        return self._model

    def _get_or_create_session(self, instructions: Optional[str] = None) -> Any:
        """Get or create an on-device LanguageModelSession."""
        fm = self._get_sdk()
        model = self._get_model()
        sys_inst = instructions or self.system_prompt
        return fm.LanguageModelSession(instructions=sys_inst, model=model)

    # ------------------------------------------------------------------
    # BaseBackend interface: generate & generate_stream
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
        """Generate text completion using Apple Foundation Models."""
        if stream:
            chunks = list(
                self.generate_stream(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    **kwargs,
                )
            )
            text = "".join(c.get("text", "") for c in chunks)
            return {"text": text, "finish_reason": "stop", "model": "apple-foundationmodel"}

        session = self._get_or_create_session()

        options = self._build_generation_options(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            **kwargs,
        )

        async def _respond() -> str:
            resp = await session.respond(prompt, options=options)
            return str(resp)

        text = _run_async(_respond())
        return {
            "text": text,
            "finish_reason": "stop",
            "model": "apple-foundationmodel",
            "usage": {
                "prompt_tokens": self.count_tokens(prompt),
                "completion_tokens": self.count_tokens(text),
                "total_tokens": self.count_tokens(prompt) + self.count_tokens(text),
            },
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
        """Stream generated text chunks.

        Converts apple_fm_sdk cumulative snapshots into standard incremental deltas.
        """
        session = self._get_or_create_session()
        options = self._build_generation_options(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            **kwargs,
        )

        q: Any = None
        import queue

        q = queue.Queue()
        sentinel = object()
        error_container: List[BaseException] = []

        async def _stream_worker() -> None:
            previous_len = 0
            try:
                async for snapshot in session.stream_response(prompt, options=options):
                    snapshot_str = str(snapshot)
                    delta = snapshot_str[previous_len:]
                    if delta:
                        previous_len = len(snapshot_str)
                        q.put({"text": delta})
            except BaseException as exc:
                error_container.append(exc)
            finally:
                q.put(sentinel)

        def _thread_target() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(_stream_worker())
            finally:
                new_loop.close()

        thread = threading.Thread(target=_thread_target)
        thread.start()

        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield item
            if error_container:
                raise error_container[0]
            yield {"text": "", "finish_reason": "stop"}
        finally:
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Chat API
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Chat with OpenAI-style messages using the native Apple Foundation Model."""
        if stream:
            chunks = list(
                self.chat_stream(
                    messages,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop=stop,
                    **kwargs,
                )
            )
            text = "".join(c.get("text", "") for c in chunks)
            return {"text": text, "finish_reason": "stop", "model": "apple-foundationmodel"}

        system_instructions, user_prompt = self._extract_messages(messages)
        session = self._get_or_create_session(instructions=system_instructions)
        options = self._build_generation_options(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            **kwargs,
        )

        async def _respond() -> str:
            resp = await session.respond(user_prompt, options=options)
            return str(resp)

        text = _run_async(_respond())
        return {
            "text": text,
            "finish_reason": "stop",
            "model": "apple-foundationmodel",
            "usage": {
                "prompt_tokens": self.count_tokens(user_prompt),
                "completion_tokens": self.count_tokens(text),
                "total_tokens": self.count_tokens(user_prompt) + self.count_tokens(text),
            },
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
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Stream chat completions chunk by chunk."""
        system_instructions, user_prompt = self._extract_messages(messages)
        session = self._get_or_create_session(instructions=system_instructions)
        options = self._build_generation_options(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            **kwargs,
        )

        import queue

        q = queue.Queue()
        sentinel = object()
        error_container: List[BaseException] = []

        async def _stream_worker() -> None:
            previous_len = 0
            try:
                async for snapshot in session.stream_response(user_prompt, options=options):
                    snapshot_str = str(snapshot)
                    delta = snapshot_str[previous_len:]
                    if delta:
                        previous_len = len(snapshot_str)
                        q.put({"text": delta})
            except BaseException as exc:
                error_container.append(exc)
            finally:
                q.put(sentinel)

        def _thread_target() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(_stream_worker())
            finally:
                new_loop.close()

        thread = threading.Thread(target=_thread_target)
        thread.start()

        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield item
            if error_container:
                raise error_container[0]
            yield {"text": "", "finish_reason": "stop"}
        finally:
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Context, Tokens, and Info
    # ------------------------------------------------------------------

    def get_context_window(self) -> int:
        """Get the model context window.

        Applies a 4096-token floor to avoid cold-start 0 values reported by macOS 27 (#192).
        """
        try:
            model = self._get_model()
            ctx = getattr(model, "context_size", None)
            if ctx is not None and isinstance(ctx, int) and ctx > 0:
                return ctx
        except Exception:
            pass
        return _DEFAULT_CONTEXT_FLOOR

    def count_tokens(self, text: str) -> int:
        """Count the tokens in a text using the model tokenizer when available."""
        if not text:
            return 0
        try:
            model = self._get_model()
            if hasattr(model, "token_count"):

                async def _count() -> int:
                    return await model.token_count(text)

                return _run_async(_count())
        except Exception:
            pass
        # Fallback heuristic: approx 4 chars per token
        return max(1, len(text) // 4)

    def embed(self, texts: Union[str, List[str]], **kwargs: Any) -> List[List[float]]:
        """Embeddings are not supported by the on-device Apple Foundation Model."""
        raise NotImplementedError(
            "The on-device Apple Foundation Model does not expose embedding vectors. "
            "For local embeddings, use the 'sentence-transformers' extra or Ollama "
            "(e.g., 'local-llm-kit embed --model nomic-embed-text --backend ollama')."
        )

    def model_info(self) -> Dict[str, Any]:
        """Return static metadata about the Apple Foundation Model."""
        variant_name = "on-device-3b"
        try:
            model = self._get_model()
            variant = getattr(model, "variant", None)
            if variant and hasattr(variant, "display_name"):
                variant_name = variant.display_name
        except Exception:
            pass
        return {
            "name": "Apple Intelligence Foundation Model",
            "backend": "apple-fm",
            "variant": variant_name,
            "on_device": True,
            "zero_download": True,
            "requires_network": False,
            "context_window": self.get_context_window(),
        }

    def close(self) -> None:
        """Clean up active sessions and release references."""
        self._session = None
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
        """Extract system instructions and combined user/conversation prompt."""
        system_parts: List[str] = []
        user_parts: List[str] = []

        for msg in messages:
            role = (msg.get("role") or "").lower()
            content = msg.get("content") or ""
            # Handle list of content parts (e.g. vision or formatted parts)
            if isinstance(content, list):
                text_pieces = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content_text = " ".join(text_pieces)
            else:
                content_text = str(content)

            if role == "system":
                system_parts.append(content_text)
            elif role == "user":
                user_parts.append(content_text)
            elif role == "assistant":
                user_parts.append(f"Assistant: {content_text}")

        system_instructions = "\n".join(system_parts) if system_parts else None
        combined_prompt = "\n".join(user_parts) if user_parts else ""
        return system_instructions, combined_prompt

    def _build_generation_options(
        self,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Build GenerationOptions if apple_fm_sdk provides it, else None."""
        try:
            fm = self._get_sdk()
            if hasattr(fm, "GenerationOptions"):
                opts_kwargs: Dict[str, Any] = {}
                temp = temperature if temperature is not None else self.default_temperature
                if temp is not None:
                    opts_kwargs["temperature"] = float(temp)
                if max_new_tokens is not None:
                    opts_kwargs["maximum_response_tokens"] = int(max_new_tokens)
                p = top_p if top_p is not None else self.default_top_p
                if p is not None and hasattr(fm, "SamplingMode"):
                    try:
                        opts_kwargs["sampling"] = fm.SamplingMode.random(top_p=float(p))
                    except Exception:
                        pass
                return fm.GenerationOptions(**opts_kwargs)
        except Exception:
            pass
        return None
