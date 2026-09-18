"""
Ollama REST backend (chat, generate, embeddings, model listing).

Talks to a running Ollama server (default ``http://localhost:11434``)
with stdlib-only ``urllib`` — no third-party dependencies. All optional
behaviour degrades with clear errors, e.g. a ``ConnectionError`` telling
the user to run ``ollama serve`` / ``ollama pull <model>``.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseBackend

_GENERATE_PATH = "/api/generate"
_CHAT_PATH = "/api/chat"
_EMBED_PATH = "/api/embed"  # new batched endpoint; fallback: /api/embeddings
_EMBEDDINGS_PATH = "/api/embeddings"
_TAGS_PATH = "/api/tags"
_SHOW_PATH = "/api/show"

_OPTION_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "repeat_penalty",
    "repetition_penalty",  # accepted as alias, mapped to repeat_penalty
    "num_predict",
    "max_new_tokens",  # accepted as alias, mapped to num_predict
    "num_ctx",
    "seed",
    "stop",
    "tfs_z",
    "typical_p",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "num_thread",
}


class OllamaBackend(BaseBackend):
    """Backend for Ollama servers. Import-safe without any optional deps."""

    backend_name = "ollama"
    supports_vision = True  # images pass through in chat messages
    supports_embeddings = True

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        context_window: int = 4096,
        keep_alive: str = "5m",
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Args:
        model: Model name as known to Ollama (e.g. ``"llama3.1"``).
        host: Ollama server base URL.
        timeout: HTTP timeout in seconds.
        context_window: Reported context window (Ollama does not expose a
            single reliable value; override via ``num_ctx`` passthrough or
            this parameter).
        keep_alive: Ollama ``keep_alive`` value sent with requests.
        headers: Extra HTTP headers.
        """
        self.model = model
        self.model_path = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.context_window = context_window
        self.keep_alive = keep_alive
        self.extra_headers = dict(headers or {})
        self.default_options = dict(kwargs)  # e.g. num_ctx=8192

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
        payload = self._generate_payload(
            prompt, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=False, **kwargs,
        )
        resp = self._post_json(_GENERATE_PATH, payload)
        return {
            "text": resp.get("response", ""),
            "finish_reason": resp.get("done_reason", "stop"),
            "model": self.model,
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
        payload = self._generate_payload(
            prompt, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=True, **kwargs,
        )
        for obj in self._post_json_lines(_GENERATE_PATH, payload):
            text = obj.get("response", "")
            if text or obj.get("done"):
                chunk: Dict[str, Any] = {"text": text}
                if obj.get("done"):
                    chunk["finish_reason"] = obj.get("done_reason", "stop")
                yield chunk

    # -- chat API (native /api/chat) -----------------------------------

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
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if stream:
            chunks = list(
                self.chat_stream(
                    messages, temperature=temperature,
                    max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty, stop=stop,
                    tools=tools, response_format=response_format, **kwargs,
                )
            )
            text = "".join(c.get("text", "") for c in chunks)
            tool_calls = None
            for c in chunks:
                if c.get("tool_calls"):
                    tool_calls = c["tool_calls"]
            result: Dict[str, Any] = {
                "text": text, "finish_reason": "stop", "model": self.model,
            }
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result
        payload = self._chat_payload(
            messages, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=False, stop=stop,
            tools=tools, response_format=response_format, **kwargs,
        )
        resp = self._post_json(_CHAT_PATH, payload)
        message = resp.get("message", {}) or {}
        return {
            "text": message.get("content", "") or "",
            "tool_calls": message.get("tool_calls"),
            "finish_reason": resp.get("done_reason", "stop"),
            "model": self.model,
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
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        payload = self._chat_payload(
            messages, temperature, max_new_tokens, top_p, top_k,
            repetition_penalty, stream=True, stop=stop,
            tools=tools, response_format=response_format, **kwargs,
        )
        for obj in self._post_json_lines(_CHAT_PATH, payload):
            message = obj.get("message", {}) or {}
            chunk: Dict[str, Any] = {"text": message.get("content", "") or ""}
            if message.get("tool_calls"):
                chunk["tool_calls"] = message["tool_calls"]
            if obj.get("done"):
                chunk["finish_reason"] = obj.get("done_reason", "stop")
                yield chunk
                break
            if chunk["text"]:
                yield chunk

    # -- embeddings ----------------------------------------------------

    def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> List[List[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        payload = {"model": self.model, "input": items,
                   "keep_alive": self.keep_alive}
        try:
            resp = self._post_json(_EMBED_PATH, payload)
        except RuntimeError as e:
            if "404" not in str(e):
                raise
            return [self._embed_single(t) for t in items]  # old server fallback
        embeddings = resp.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(items):
            raise RuntimeError(
                "Unexpected Ollama /api/embed response: %r" % (resp,)
            )
        return embeddings

    def _embed_single(self, text: str) -> List[float]:
        resp = self._post_json(
            _EMBEDDINGS_PATH,
            {"model": self.model, "prompt": text,
             "keep_alive": self.keep_alive},
        )
        embedding = resp.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(
                "Unexpected Ollama /api/embeddings response: %r" % (resp,)
            )
        return embedding

    # -- introspection -------------------------------------------------

    def get_context_window(self) -> int:
        return self.context_window

    def count_tokens(self, text: str) -> int:
        """~4 chars/token heuristic (server-side tokenization unavailable)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def model_info(self) -> Dict[str, Any]:
        info = super().model_info()
        info.update({"host": self.host, "keep_alive": self.keep_alive})
        return info

    def list_models(self) -> List[Dict[str, Any]]:
        """GET /api/tags, normalized to ``[{"id", "object", ...}]``."""
        resp = self._get_json(_TAGS_PATH)
        out = []
        for m in resp.get("models", []) or []:
            out.append({
                "id": m.get("name", m.get("model", "unknown")),
                "object": "model",
                "owned_by": "ollama",
                "size": m.get("size"),
                "digest": m.get("digest"),
                "modified_at": m.get("modified_at"),
                "details": m.get("details", {}),
            })
        return out

    def show(self) -> Dict[str, Any]:
        """POST /api/show — raw server-side model details (requires server)."""
        return self._post_json(_SHOW_PATH, {"model": self.model})

    # -- payload builders ----------------------------------------------

    def _options(
        self,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop: Optional[List[str]],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "num_predict": max_new_tokens,
        }
        merged = dict(self.default_options)
        merged.update(extra)
        for key, value in merged.items():
            if value is None:
                continue
            if key == "repetition_penalty":
                options["repeat_penalty"] = value
            elif key == "max_new_tokens":
                options["num_predict"] = value
            elif key in _OPTION_KEYS:
                options[key] = value
        if stop:
            options["stop"] = stop
        return options

    def _split_kwargs(self, kwargs: Dict[str, Any]) -> Any:
        options_extra = {k: v for k, v in kwargs.items() if k in _OPTION_KEYS}
        top_level = {k: v for k, v in kwargs.items() if k not in _OPTION_KEYS}
        return options_extra, top_level

    def _generate_payload(self, prompt: str, temperature: float,
                          max_new_tokens: int, top_p: float, top_k: int,
                          repetition_penalty: float, stream: bool,
                          **kwargs: Any) -> Dict[str, Any]:
        options_extra, top_level = self._split_kwargs(kwargs)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self._options(temperature, max_new_tokens, top_p,
                                     top_k, repetition_penalty, None,
                                     options_extra),
        }
        payload.update(top_level)
        return payload

    def _chat_payload(self, messages: List[Dict[str, Any]], temperature: float,
                      max_new_tokens: int, top_p: float, top_k: int,
                      repetition_penalty: float, stream: bool,
                      stop: Optional[List[str]],
                      tools: Optional[List[Dict[str, Any]]],
                      response_format: Optional[Dict[str, Any]],
                      **kwargs: Any) -> Dict[str, Any]:
        options_extra, top_level = self._split_kwargs(kwargs)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self._options(temperature, max_new_tokens, top_p,
                                     top_k, repetition_penalty, stop,
                                     options_extra),
        }
        if tools:
            payload["tools"] = tools
        fmt = self._ollama_format(response_format)
        if fmt is not None:
            payload["format"] = fmt
        payload.update(top_level)
        return payload

    @staticmethod
    def _ollama_format(response_format: Optional[Dict[str, Any]]) -> Any:
        if not response_format:
            return None
        rtype = response_format.get("type")
        if rtype == "json_object":
            return "json"
        if rtype == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            return schema if schema is not None else "json"
        return None

    # -- HTTP helpers (stdlib only) ------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        return headers

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data, headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "Ollama request POST %s failed with HTTP %s: %s"
                % (path, e.code, body)
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                "Ollama server not reachable at %s (%s). Is Ollama running? "
                "Start it with `ollama serve` and pull a model with "
                "`ollama pull %s`." % (self.host, e.reason, self.model)
            )
        except (ValueError, json.JSONDecodeError) as e:
            raise RuntimeError("Ollama returned invalid JSON: %s" % e)

    def _post_json_lines(
        self, path: str, payload: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """POST with ``stream=True``; yields one dict per JSON line."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data, headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue  # skip keep-alive / partial lines
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "Ollama request POST %s failed with HTTP %s: %s"
                % (path, e.code, body)
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                "Ollama server not reachable at %s (%s). Is Ollama running? "
                "Start it with `ollama serve`." % (self.host, e.reason)
            )

    def _get_json(self, path: str) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.host + path, headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "Ollama request GET %s failed with HTTP %s: %s"
                % (path, e.code, body)
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                "Ollama server not reachable at %s (%s). Is Ollama running? "
                "Start it with `ollama serve`." % (self.host, e.reason)
            )
