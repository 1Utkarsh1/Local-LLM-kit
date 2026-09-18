"""
Generic OpenAI-compatible HTTP backend.

Works against any server exposing ``/v1/*``: vLLM, LM Studio, llama.cpp
``llama-server``, text-generation-webui (OpenAI extension), etc.

Endpoints used:
  - ``POST /v1/chat/completions`` (+ SSE streaming)
  - ``POST /v1/completions`` (+ SSE streaming, legacy/text models)
  - ``POST /v1/embeddings``
  - ``GET  /v1/models``

Stdlib-only (``urllib`` + ``json``). ``api_key`` is optional — local
servers usually don't need one.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseBackend

_DONE = "[DONE]"


class OpenAICompatBackend(BaseBackend):
    """Backend for any OpenAI-compatible server. Import-safe (stdlib only)."""

    backend_name = "openai-compat"
    supports_vision = True  # content parts pass through untouched
    supports_embeddings = True

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        context_window: int = 4096,
        headers: Optional[Dict[str, str]] = None,
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
        """
        self.model = model
        self.model_path = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.context_window = context_window
        self.extra_headers = dict(headers or {})
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
        if stream:
            text = "".join(
                c["text"]
                for c in self.generate_stream(
                    prompt, temperature=temperature,
                    max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty, **kwargs,
                )
            )
            return {"text": text, "finish_reason": "stop", "model": self.model}
        payload = self._completion_payload(
            prompt, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=False, **kwargs,
        )
        resp = self._post_json("/completions", payload)
        try:
            choice = (resp.get("choices") or [])[0]
        except IndexError:
            raise RuntimeError(
                "Server returned no completion choices: %r" % (resp,)
            )
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
        payload = self._completion_payload(
            prompt, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=True, **kwargs,
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
                    messages, temperature=temperature,
                    max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty, stop=stop,
                    tools=tools, tool_choice=tool_choice,
                    response_format=response_format, seed=seed, **kwargs,
                )
            )
            text = "".join(c.get("text", "") for c in chunks)
            tool_calls = None
            for c in chunks:
                if c.get("tool_calls"):
                    tool_calls = c["tool_calls"]
            finish = next(
                (c["finish_reason"] for c in reversed(chunks)
                 if c.get("finish_reason")),
                "stop",
            )
            result: Dict[str, Any] = {
                "text": text, "finish_reason": finish, "model": self.model,
            }
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result
        payload = self._chat_payload(
            messages, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=False, stop=stop, tools=tools,
            tool_choice=tool_choice, response_format=response_format,
            seed=seed, **kwargs,
        )
        resp = self._post_json("/chat/completions", payload)
        try:
            choice = (resp.get("choices") or [])[0]
        except IndexError:
            raise RuntimeError(
                "Server returned no chat choices: %r" % (resp,)
            )
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
            messages, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=True, stop=stop, tools=tools,
            tool_choice=tool_choice, response_format=response_format,
            seed=seed, **kwargs,
        )
        for event in self._post_sse("/chat/completions", payload):
            for choice in event.get("choices", []) or []:
                delta = choice.get("delta", {}) or {}
                chunk: Dict[str, Any] = {
                    "text": self._content_text(delta.get("content")),
                }
                if delta.get("tool_calls"):
                    chunk["tool_calls_delta"] = delta["tool_calls"]
                    chunk["tool_calls"] = delta["tool_calls"]
                if choice.get("finish_reason"):
                    chunk["finish_reason"] = choice["finish_reason"]
                if (chunk["text"] or chunk.get("finish_reason")
                        or "tool_calls_delta" in chunk):
                    yield chunk

    # -- embeddings (/v1/embeddings) -----------------------------------

    def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> List[List[float]]:
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
        except (KeyError, TypeError, AttributeError):
            raise RuntimeError(
                "Unexpected /v1/embeddings response: %r" % (resp,)
            )

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
                out.append({
                    "id": m.get("id", "unknown"),
                    "object": m.get("object", "model"),
                    "created": m.get("created"),
                    "owned_by": m.get("owned_by"),
                })
            else:
                out.append({"id": str(m), "object": "model"})
        return out

    # -- payload builders ----------------------------------------------

    def _base_sampling(self, temperature: float, max_new_tokens: int,
                       top_p: float, **kwargs: Any) -> Dict[str, Any]:
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

    def _completion_payload(self, prompt: str, temperature: float,
                            max_new_tokens: int, top_p: float, top_k: int,
                            repetition_penalty: float, stream: bool,
                            **kwargs: Any) -> Dict[str, Any]:
        # top_k / repetition_penalty are not OpenAI fields; most local
        # servers accept them as extra body params, so pass through.
        kwargs.setdefault("top_k", top_k)
        kwargs.setdefault("repeat_penalty", repetition_penalty)
        payload = self._base_sampling(temperature, max_new_tokens, top_p,
                                      stream=stream, **kwargs)
        payload["prompt"] = prompt
        return payload

    def _chat_payload(self, messages: List[Dict[str, Any]],
                      temperature: float, max_new_tokens: int, top_p: float,
                      top_k: int, repetition_penalty: float, stream: bool,
                      stop: Optional[List[str]],
                      tools: Optional[List[Dict[str, Any]]],
                      tool_choice: Any,
                      response_format: Optional[Dict[str, Any]],
                      seed: Optional[int], **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("top_k", top_k)
        kwargs.setdefault("repeat_penalty", repetition_penalty)
        payload = self._base_sampling(temperature, max_new_tokens, top_p,
                                      stream=stream, **kwargs)
        payload["messages"] = messages
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
                str(p.get("text", "")) for p in content
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

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers=self._headers(), method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "OpenAI-compatible request POST %s%s failed with HTTP %s: %s"
                % (self.base_url, path, e.code, body)
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                "Server not reachable at %s (%s). Is your OpenAI-compatible "
                "server (vLLM / LM Studio / llama-server) running with "
                "--host/--port matching base_url?" % (self.base_url, e.reason)
            )

    def _post_sse(
        self, path: str, payload: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """POST with ``stream=True``; parses SSE ``data:`` frames."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers=self._headers(stream=True), method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue  # blank / keep-alive comment
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == _DONE:
                        break
                    try:
                        event = json.loads(data_str)
                    except ValueError:
                        continue  # partial frame; skip
                    if isinstance(event, dict):
                        yield event
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "OpenAI-compatible request POST %s%s failed with HTTP %s: %s"
                % (self.base_url, path, e.code, body)
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                "Server not reachable at %s (%s). Is your OpenAI-compatible "
                "server running?" % (self.base_url, e.reason)
            )

    def _get_json(self, path: str) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path, headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result if isinstance(result, dict) else {"data": result}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "OpenAI-compatible request GET %s%s failed with HTTP %s: %s"
                % (self.base_url, path, e.code, body)
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                "Server not reachable at %s (%s)." % (self.base_url, e.reason)
            )
