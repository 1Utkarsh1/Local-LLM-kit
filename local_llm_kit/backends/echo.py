"""
Deterministic offline Echo/Mock backend for tests, docs and CI.

No dependencies, no network, no model files. Echoes the prompt (or the
last user message for :meth:`chat`), streams word-by-word, and produces
deterministic fake embeddings. Token counting is whitespace-based and
documented as approximate — it exists so usage accounting and tests are
exercisable without a real model.
"""

import hashlib
import math
import random
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseBackend


class EchoBackend(BaseBackend):
    """Deterministic mock backend. Import-safe everywhere (stdlib only)."""

    backend_name = "echo"
    supports_vision = True  # vision parts degrade to "[image]" via message_text
    supports_embeddings = True

    def __init__(
        self,
        model: str = "echo-1",
        context_window: int = 8192,
        embedding_dim: int = 16,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Args:
        model: Fake model id reported by model_info()/list_models().
        context_window: Reported context window size.
        embedding_dim: Dimension of fake embedding vectors.
        seed: Extra seed mixed into the deterministic embedding hash.
        """
        self.model = model
        self.model_path = model  # alias so model_info works either way
        self.context_window = context_window
        self.embedding_dim = embedding_dim
        self.seed = seed

    # -- prompt API ----------------------------------------------------

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
        text = self._truncate("Echo: %s" % prompt, max_new_tokens)
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
            text = "".join(c["text"] for c in chunks)
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
        text = self._truncate("Echo: %s" % prompt, max_new_tokens)
        for chunk in self._stream_text(text):
            yield {"text": chunk}

    # -- chat API ------------------------------------------------------

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
        if stream:
            chunks = list(
                self.chat_stream(
                    messages,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
            )
            return {
                "text": "".join(c["text"] for c in chunks),
                "finish_reason": "stop",
                "model": self.model,
            }
        text = self._truncate(
            "Echo: %s" % self._last_user_text(messages), max_new_tokens
        )
        text = self._apply_stop(text, stop)
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
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        text = self._truncate(
            "Echo: %s" % self._last_user_text(messages), max_new_tokens
        )
        text = self._apply_stop(text, stop)
        for chunk in self._stream_text(text):
            yield {"text": chunk}

    # -- embeddings ----------------------------------------------------

    def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> List[List[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        return [self._fake_vector(t) for t in items]

    # -- introspection -------------------------------------------------

    def get_context_window(self) -> int:
        return self.context_window

    def count_tokens(self, text: str) -> int:
        """Whitespace word count (approximate by design; see module docstring)."""
        if not text or not text.strip():
            return 0
        return len(text.split())

    def model_info(self) -> Dict[str, Any]:
        info = super().model_info()
        info.update({"embedding_dim": self.embedding_dim, "deterministic": True})
        return info

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": self.model, "object": "model", "owned_by": "local-llm-kit"}
        ]

    # -- internals -----------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_new_tokens: int) -> str:
        words = text.split(" ")
        return " ".join(words[: max(0, max_new_tokens)])

    @staticmethod
    def _stream_text(text: str) -> Iterator[str]:
        if not text:
            return
        words = text.split(" ")
        for i, w in enumerate(words):
            yield w if i == len(words) - 1 else w + " "

    def _last_user_text(self, messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return self.message_text(m)
        return self.message_text(messages[-1]) if messages else ""

    @staticmethod
    def _apply_stop(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        cut = len(text)
        for s in stop:
            i = text.find(s)
            if i != -1:
                cut = min(cut, i)
        return text[:cut]

    def _fake_vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") ^ (self.seed & 0xFFFFFFFFFFFFFFFF)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.embedding_dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


#: Backwards-friendly alias (docs/tests may refer to either name).
MockBackend = EchoBackend
