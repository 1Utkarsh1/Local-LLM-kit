"""
Base class for model backends.

2026 extensions (backwards compatible — all new members have defaults,
so existing subclasses keep working unmodified):
  - ``supports_vision`` / ``supports_embeddings`` capability flags
  - ``chat()`` / ``chat_stream()`` (OpenAI-style messages; default impl
    falls back to prompt ``generate()`` so prompt-only backends inherit it)
  - ``embed()`` (raises ``NotImplementedError`` by default)
  - ``model_info()`` (static info dict, never requires a server)
  - ``list_models()`` (``[]`` = "listing not supported"; server backends override)
  - ``close()`` + context-manager support (no-op by default)
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union


class BaseBackend(ABC):
    """
    Abstract base class for model backends.

    All backend implementations must inherit from this class and
    implement :meth:`generate`, :meth:`generate_stream`,
    :meth:`get_context_window` and :meth:`count_tokens`.
    """

    #: Short backend identifier (e.g. "echo", "ollama", "transformers").
    backend_name = "base"

    #: Set True when the backend can accept vision-style messages
    #: (``content`` as a list of ``{"type": "text"|"image_url", ...}`` parts).
    #: Backends that only do plain text should leave this False; the
    #: default :meth:`chat` implementation degrades images to a placeholder.
    supports_vision = False

    #: Set True when the backend implements :meth:`embed`.
    supports_embeddings = False

    # ------------------------------------------------------------------
    # Required prompt-completion API (unchanged, backwards compatible)
    # ------------------------------------------------------------------

    @abstractmethod
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
        """
        Generate text completion for a prompt.

        Args:
            prompt: Input text to complete
            temperature: Sampling temperature (higher = more random)
            max_new_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            stream: Whether to stream the response (should be False for this method)
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top tokens to return logprobs for
            **kwargs: Additional backend-specific parameters

        Returns:
            Dictionary with at least a "text" key containing the generated text
        """
        pass

    @abstractmethod
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
        """
        Stream text completion for a prompt.

        Args:
            prompt: Input text to complete
            temperature: Sampling temperature (higher = more random)
            max_new_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top tokens to return logprobs for
            **kwargs: Additional backend-specific parameters

        Yields:
            Dictionaries with at least a "text" key containing a chunk of generated text
        """
        pass

    @abstractmethod
    def get_context_window(self) -> int:
        """
        Get the context window size of the model.

        Returns:
            Maximum context window size in tokens
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        pass

    def get_timestamp(self) -> int:
        """
        Get current timestamp.

        Returns:
            Current timestamp in seconds
        """
        return int(time.time())

    # ------------------------------------------------------------------
    # 2026 chat API (optional override; default falls back to generate)
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
        """Chat with OpenAI-style messages.

        Default implementation converts messages to a prompt with
        :meth:`messages_to_prompt` and calls :meth:`generate`, so
        prompt-only backends (transformers, llama.cpp) inherit this for
        free. Server backends (Ollama, OpenAI-compatible) override it to
        use the native chat endpoint (vision parts, tools, JSON mode).

        Returns:
            Dict with at least a ``"text"`` key (assistant content).
            Native overrides may add ``"tool_calls"``,
            ``"finish_reason"``, ``"usage"`` and ``"raw"`` keys.
        """
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
            result: Dict[str, Any] = {"text": text}
            for c in reversed(chunks):  # prefer final-chunk metadata
                if c.get("finish_reason"):
                    result["finish_reason"] = c["finish_reason"]
                    break
            return result
        prompt = self.messages_to_prompt(messages)
        return self.generate(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            **kwargs,
        )

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
        """Stream a chat response. Default: prompt fallback via generate_stream."""
        prompt = self.messages_to_prompt(messages)
        for chunk in self.generate_stream(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            **kwargs,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # 2026 embeddings API (optional)
    # ------------------------------------------------------------------

    def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> List[List[float]]:
        """Embed one string or a batch of strings.

        Returns:
            List of embedding vectors (one per input text), each a
            list of floats. Always returns a list-of-lists even for a
            single string input.

        Raises:
            NotImplementedError: if this backend has no embedding support.
        """
        raise NotImplementedError(
            "%s does not support embeddings." % type(self).__name__
        )

    # ------------------------------------------------------------------
    # Introspection (optional overrides)
    # ------------------------------------------------------------------

    def model_info(self) -> Dict[str, Any]:
        """Static info about this backend/model. Must never require a server."""
        try:
            context_window: Optional[int] = self.get_context_window()
        except Exception:
            context_window = None
        model = getattr(self, "model", getattr(self, "model_path", "unknown"))
        return {
            "backend": self.backend_name,
            "model": model,
            "context_window": context_window,
            "supports_vision": self.supports_vision,
            "supports_embeddings": self.supports_embeddings,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List models available from this backend.

        Returns ``[]`` when listing is not supported (single-model local
        backends). Server backends override this to query the server.
        Each entry is at least ``{"id": <model-id>}``.
        """
        return []

    def close(self) -> None:
        """Release backend resources (no-op by default)."""

    def __enter__(self) -> "BaseBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def message_text(message: Dict[str, Any]) -> str:
        """Extract plain text from a message (vision parts -> placeholders)."""
        content = message.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append(str(part))
                    continue
                ptype = part.get("type", "text")
                if ptype == "text":
                    parts.append(str(part.get("text", "")))
                elif ptype in ("image_url", "image"):
                    parts.append("[image]")
                else:
                    parts.append(str(part.get("text", "")))
            return "".join(parts)
        return str(content)

    @classmethod
    def messages_to_prompt(cls, messages: List[Dict[str, Any]]) -> str:
        """Flatten chat messages to a ``Role: text`` prompt (fallback only)."""
        lines = []
        for m in messages:
            role = str(m.get("role", "user")).capitalize()
            lines.append("%s: %s" % (role, cls.message_text(m)))
        lines.append("Assistant:")
        return "\n".join(lines)
