"""
Generic OpenAI-compatible HTTP backend.

Works against any server exposing ``/v1/*``: vLLM, LM Studio, llama.cpp
``llama-server``, text-generation-webui (OpenAI extension), etc.

Endpoints used:
  - ``POST /v1/chat/completions`` (+ SSE streaming)
  - ``POST /v1/completions`` (+ SSE streaming, legacy/text models)
  - ``POST /v1/embeddings``
  - ``GET  /v1/models``

Provider profiles: ``provider`` selects protocol quirks. ``"apfel"`` targets
``apfel --serve`` (https://github.com/Arthur-Ficial/apfel), whose server
answers 400 to sampling fields it does not implement (``top_k``/penalties)
and 501 to ``POST /v1/completions`` and ``POST /v1/embeddings``. That
profile therefore omits those fields, rejects prompt-completion calls with
a pointer to ``chat()``, and reports no embedding support.

Stdlib-only (``urllib`` + ``json``). ``api_key`` is optional — local
servers usually don't need one.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Union

from ..retry import DEFAULT_RETRY, RetryPolicy, coerce_policy, retry_after_seconds
from .base import BaseBackend

_DONE = "[DONE]"


class OpenAICompatBackend(BaseBackend):
    """Backend for any OpenAI-compatible server. Import-safe (stdlib only)."""

    backend_name = "openai-compat"
    supports_vision = True  # content parts pass through untouched
    supports_embeddings = True
    supports_native_chat = True

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        context_window: int = 4096,
        headers: Optional[Dict[str, str]] = None,
        provider: str = "openai-compat",
        max_retries: Union[int, RetryPolicy, None] = None,
        **kwargs: Any,
    ) -> None:
        """Args:
        model: Model id as known to the server (e.g. ``"default"`` for
            single-model servers like llama-server).
        base_url: Server base URL including ``/v1``.
        api_key: Optional bearer token (``Authorization: Bearer ...``).
        timeout: HTTP timeout in seconds.
        context_window: Reported context window (override per server).
        headers: Extra HTTP headers.
        max_retries: Retries for transient failures — a refused connection or a
            ``429``/``5xx`` answer. Defaults to two retries with exponential
            backoff and ``Retry-After`` support, matching the official OpenAI
            Python client. Pass ``0`` for single-attempt behaviour, an int for a
            custom count, or a :class:`~local_llm_kit.retry.RetryPolicy`.
        """
        self.model = model
        self.provider = provider
        self.supports_embeddings = provider not in ("gpt4all", "mlx-lm", "apfel")
        self._chat_only = provider == "apfel"
        self.model_path = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.context_window = context_window
        self.extra_headers = dict(headers or {})
        self.retry = coerce_policy(DEFAULT_RETRY if max_retries is None else max_retries)
        self.default_params = dict(kwargs)

    # -- prompt API (/v1/completions) ----------------------------------

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
        if self._chat_only:
            raise RuntimeError(
                "The 'apfel' server exposes only POST /v1/chat/completions "
                "(POST /v1/completions answers 501). Use chat() instead of "
                "generate()/complete() with this provider."
            )
        if stream:
            text = "".join(
                c["text"]
                for c in self.generate_stream(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    **kwargs,
                )
            )
            return {"text": text, "finish_reason": "stop", "model": self.model}
        payload = self._completion_payload(
            prompt,
            temperature,
            max_new_tokens,
            top_p,
            top_k,
            repetition_penalty,
            stream=False,
            **kwargs,
        )
        resp = self._post_json("/completions", payload)
        try:
            choice = (resp.get("choices") or [])[0]
        except IndexError as e:
            raise RuntimeError("Server returned no completion choices: %r" % (resp,)) from e
        return {
            "text": choice.get("text", "") or "",
            "finish_reason": choice.get("finish_reason", "stop"),
            "usage": resp.get("usage"),
            "model": resp.get("model", self.model),
            "raw": resp,
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
        if self._chat_only:
            raise RuntimeError(
                "The 'apfel' server exposes only POST /v1/chat/completions "
                "(POST /v1/completions answers 501). Use chat_stream() instead "
                "of generate_stream() with this provider."
            )
        payload = self._completion_payload(
            prompt,
            temperature,
            max_new_tokens,
            top_p,
            top_k,
            repetition_penalty,
            stream=True,
            **kwargs,
        )
        for event in self._post_sse("/completions", payload):
            for choice in event.get("choices", []) or []:
                chunk: Dict[str, Any] = {"text": choice.get("text", "") or ""}
                if choice.get("finish_reason"):
                    chunk["finish_reason"] = choice["finish_reason"]
                if chunk["text"] or chunk.get("finish_reason"):
                    yield chunk

    # -- chat API (/v1/chat/completions) -------------------------------

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
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop=stop,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    seed=seed,
                    **kwargs,
                )
            )
            text = "".join(c.get("text", "") for c in chunks)
            tool_calls = None
            for c in chunks:
                if c.get("tool_calls"):
                    tool_calls = c["tool_calls"]
            finish = next(
                (c["finish_reason"] for c in reversed(chunks) if c.get("finish_reason")),
                "stop",
            )
            result: Dict[str, Any] = {
                "text": text,
                "finish_reason": finish,
                "model": self.model,
            }
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result
        payload = self._chat_payload(
            messages,
            temperature,
            max_new_tokens,
            top_p,
            top_k,
            repetition_penalty,
            stream=False,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            seed=seed,
            **kwargs,
        )
        resp = self._post_json("/chat/completions", payload)
        try:
            choice = (resp.get("choices") or [])[0]
        except IndexError as e:
            raise RuntimeError("Server returned no chat choices: %r" % (resp,)) from e
        message = choice.get("message", {}) or {}
        return {
            "text": self._content_text(message.get("content")),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason", "stop"),
            "usage": resp.get("usage"),
            "model": resp.get("model", self.model),
            "raw": resp,
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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        payload = self._chat_payload(
            messages,
            temperature,
            max_new_tokens,
            top_p,
            top_k,
            repetition_penalty,
            stream=True,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            seed=seed,
            **kwargs,
        )
        calls: Dict[int, Dict[str, Any]] = {}
        for event in self._post_sse("/chat/completions", payload):
            for choice in event.get("choices", []) or []:
                delta = choice.get("delta", {}) or {}
                chunk: Dict[str, Any] = {
                    "text": self._content_text(delta.get("content")),
                }
                if delta.get("tool_calls"):
                    chunk["tool_calls_delta"] = delta["tool_calls"]
                    for part in delta["tool_calls"]:
                        index = part.get("index", 0)
                        call = calls.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if part.get("id"):
                            call["id"] += part["id"]
                        function = part.get("function") or {}
                        for key in ("name", "arguments"):
                            if function.get(key) is not None:
                                call["function"][key] += function[key]
                if choice.get("finish_reason"):
                    chunk["finish_reason"] = choice["finish_reason"]
                    if calls:
                        chunk["tool_calls"] = [calls[index] for index in sorted(calls)]
                if chunk["text"] or chunk.get("finish_reason") or "tool_calls_delta" in chunk:
                    yield chunk

    # -- embeddings (/v1/embeddings) -----------------------------------

    def embed(self, texts: Union[str, List[str]], **kwargs: Any) -> List[List[float]]:
        if not self.supports_embeddings:
            raise NotImplementedError(
                "%s does not expose /v1/embeddings; use a separate embedding backend."
                % self.provider
            )
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        payload: Dict[str, Any] = {"model": self.model, "input": items}
        payload.update(kwargs)
        resp = self._post_json("/embeddings", payload)
        data = resp.get("data", []) or []
        try:
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            return [list(d["embedding"]) for d in ordered]
        except (KeyError, TypeError, AttributeError) as e:
            raise RuntimeError("Unexpected /v1/embeddings response: %r" % (resp,)) from e

    # -- introspection -------------------------------------------------

    def get_context_window(self) -> int:
        return self.context_window

    def count_tokens(self, text: str) -> int:
        """~4 chars/token heuristic (exact tokenization lives server-side)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def model_info(self) -> Dict[str, Any]:
        info = super().model_info()
        info.update({"base_url": self.base_url})
        return info

    def list_models(self) -> List[Dict[str, Any]]:
        """GET /v1/models (may be ``[]`` on servers that omit the route)."""
        try:
            resp = self._get_json("/models")
        except RuntimeError as e:
            if "404" in str(e):
                return []
            raise
        models = resp.get("data", []) if isinstance(resp, dict) else []
        out = []
        for m in models or []:
            if isinstance(m, dict):
                out.append(
                    {
                        "id": m.get("id", "unknown"),
                        "object": m.get("object", "model"),
                        "created": m.get("created"),
                        "owned_by": m.get("owned_by"),
                    }
                )
            else:
                out.append({"id": str(m), "object": "model"})
        return out

    # -- payload builders ----------------------------------------------

    def _base_sampling(
        self, temperature: float, max_new_tokens: int, top_p: float, **kwargs: Any
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "stream": kwargs.pop("stream", False),
        }
        params.update(self.default_params)
        params.update(kwargs)  # explicit call args win; unknown keys pass through
        return params

    def _completion_payload(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # top_k / repetition_penalty are not OpenAI fields; most local
        # servers accept them as extra body params, so pass through.
        # Exception: the apfel profile rejects them with HTTP 400, so omit.
        if self.provider != "apfel":
            kwargs.setdefault("top_k", top_k)
            penalty_key = (
                "repetition_penalty"
                if self.provider in ("vllm", "sglang", "mlx-lm")
                else "repeat_penalty"
            )
            kwargs.setdefault(penalty_key, repetition_penalty)
        payload = self._base_sampling(temperature, max_new_tokens, top_p, stream=stream, **kwargs)
        payload["prompt"] = prompt
        return payload

    def _chat_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stream: bool,
        stop: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Any,
        response_format: Optional[Dict[str, Any]],
        seed: Optional[int],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # See _completion_payload: the apfel profile rejects top_k/penalty
        # fields with HTTP 400, so omit them there.
        if self.provider != "apfel":
            kwargs.setdefault("top_k", top_k)
            penalty_key = (
                "repetition_penalty"
                if self.provider in ("vllm", "sglang", "mlx-lm")
                else "repeat_penalty"
            )
            kwargs.setdefault(penalty_key, repetition_penalty)
        payload = self._base_sampling(temperature, max_new_tokens, top_p, stream=stream, **kwargs)
        payload["messages"] = [
            {
                key: value
                for key, value in message.items()
                if not (key == "function_call" and message.get("tool_calls"))
                and not (key == "name" and message.get("role") == "tool")
            }
            for message in messages
        ]
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed
        return payload

    @staticmethod
    def _content_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # vision-style part list in a response
            return "".join(
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type", "text") == "text"
            )
        return str(content)

    # -- HTTP helpers (stdlib only) ------------------------------------

    def _headers(self, stream: bool = False) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        if stream:
            headers["Accept"] = "text/event-stream"
        headers.update(self.extra_headers)
        return headers

    def _open_with_retry(self, request_factory: Any, method: str, path: str) -> Any:
        """Run a request, retrying transient failures per :attr:`retry`.

        Retries a refused connection (the server has not started yet) and the
        transient HTTP statuses (``429``, ``5xx`` — llama.cpp answers ``503``
        while it loads a model). Client errors such as ``400`` fail immediately.
        The error messages match the single-attempt behaviour.
        """
        policy = self.retry
        attempt = 0
        while True:
            try:
                return urllib.request.urlopen(request_factory(), timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if policy.should_retry_status(e.code) and attempt < policy.max_retries:
                    policy.sleep(attempt + 1, retry_after_seconds(e.headers))
                    attempt += 1
                    continue
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    "OpenAI-compatible request %s %s%s failed with HTTP %s: %s"
                    % (method, self.base_url, path, e.code, body)
                ) from e
            except urllib.error.URLError as e:
                if attempt < policy.max_retries:
                    policy.sleep(attempt + 1)
                    attempt += 1
                    continue
                raise ConnectionError(
                    "Server not reachable at %s (%s). Is your OpenAI-compatible "
                    "server (vLLM / LM Studio / llama-server) running with "
                    "--host/--port matching base_url?" % (self.base_url, e.reason)
                ) from e

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")

        def _request() -> Any:
            return urllib.request.Request(
                self.base_url + path,
                data=data,
                headers=self._headers(),
                method="POST",
            )

        with self._open_with_retry(_request, "POST", path) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_sse(self, path: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """POST with ``stream=True``; parses SSE ``data:`` frames.

        Only the connection phase is retried: once the first frame has been
        yielded, retrying would replay tokens the caller already received.
        """
        data = json.dumps(payload).encode("utf-8")

        def _request() -> Any:
            return urllib.request.Request(
                self.base_url + path,
                data=data,
                headers=self._headers(stream=True),
                method="POST",
            )

        with self._open_with_retry(_request, "POST", path) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue  # blank / keep-alive comment
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == _DONE:
                    break
                try:
                    event = json.loads(data_str)
                except ValueError:
                    continue  # partial frame; skip
                if isinstance(event, dict):
                    yield event

    def _get_json(self, path: str) -> Dict[str, Any]:
        def _request() -> Any:
            return urllib.request.Request(
                self.base_url + path, headers=self._headers(), method="GET"
            )

        with self._open_with_retry(_request, "GET", path) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result if isinstance(result, dict) else {"data": result}
