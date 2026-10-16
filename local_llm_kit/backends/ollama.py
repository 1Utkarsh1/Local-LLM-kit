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

from ..retry import DEFAULT_RETRY, RetryPolicy, coerce_policy, retry_after_seconds
from .base import BaseBackend

_GENERATE_PATH = "/api/generate"
_CHAT_PATH = "/api/chat"
_EMBED_PATH = "/api/embed"  # new batched endpoint; fallback: /api/embeddings
_EMBEDDINGS_PATH = "/api/embeddings"
_TAGS_PATH = "/api/tags"
_SHOW_PATH = "/api/show"

#: Timeout used by the optional capability lookup, which must not stall a call.
_DETECT_TIMEOUT = 5.0

#: Capability detection is best-effort and must stay fast: one attempt only.
_NO_RETRY = RetryPolicy(max_retries=0)

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
    supports_embeddings = True
    supports_native_chat = True

    #: Vision is forwarded as image content parts, so it is *capable* by default.
    #: The loaded model may not be: :meth:`detect_capabilities` narrows this to
    #: what the server reports for the model.
    _default_supports_vision = True

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        context_window: int = 4096,
        keep_alive: str = "5m",
        headers: Optional[Dict[str, str]] = None,
        auto_detect_capabilities: bool = False,
        max_retries: Union[int, RetryPolicy, None] = None,
        **kwargs: Any,
    ) -> None:
        """Args:
        model: Model name as known to Ollama (e.g. ``"llama3.1"``).
        host: Ollama server base URL.
        timeout: HTTP timeout in seconds.
        context_window: Fallback context window, used until the server reports
            the model's real one. Ollama's ``/api/show`` publishes it (for
            example 32768 for ``qwen2.5:0.5b``), so 4096 is only a floor.
        keep_alive: Ollama ``keep_alive`` value sent with requests.
        headers: Extra HTTP headers.
        auto_detect_capabilities: Read the model's real context length and
            capability list from the server on first use. Off by default so
            constructing a backend never requires a running server; turn it on
            to get truthful ``context_window``/``supports_vision``.
        max_retries: Retries for transient failures — a refused connection
            (``ollama serve`` has not started yet) or a ``429``/``5xx`` answer.
            Defaults to two retries with exponential backoff and ``Retry-After``
            support, matching the official OpenAI Python client. Pass ``0`` for
            single-attempt behaviour, an int for a custom count, or a
            :class:`~local_llm_kit.retry.RetryPolicy`.
        """
        self.model = model
        self.model_path = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.context_window = context_window
        self.keep_alive = keep_alive
        self.extra_headers = dict(headers or {})
        self.default_options = dict(kwargs)  # e.g. num_ctx=8192
        self.auto_detect_capabilities = bool(auto_detect_capabilities)
        self.retry = coerce_policy(DEFAULT_RETRY if max_retries is None else max_retries)
        #: Capability names reported by the server (``None`` until detected).
        self.capabilities: Optional[List[str]] = None
        #: True when the model is routed to a remote host such as ollama.com.
        self.remote = False
        self.remote_host: Optional[str] = None
        self._detected = False

    # -- capability detection ------------------------------------------

    @property
    def supports_vision(self) -> bool:  # type: ignore[override]
        """True when images can usefully reach the model.

        Reports the server's own ``vision`` capability once detected, and the
        adapter's forwarding ability otherwise.
        """
        if self.capabilities is not None:
            return any(c.lower() in ("vision", "image") for c in self.capabilities)
        return self._default_supports_vision

    def detect_capabilities(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Ask the server what this model can do. Never raises.

        Reads ``POST /api/show`` for the true ``context_length`` and
        ``capabilities``, and ``GET /api/tags`` for ``remote_host`` (Ollama
        marks provider-routed models there). Updates
        :attr:`context_window`, :attr:`capabilities`, :attr:`remote` and
        :attr:`remote_host` in place.

        Returns:
            The detection outcome, including what changed. A failure is
            reported in ``"error"`` and leaves the current values untouched.
        """
        result: Dict[str, Any] = {
            "model": self.model,
            "detected": False,
            "capabilities": None,
            "context_window": self.context_window,
            "remote": self.remote,
            "remote_host": self.remote_host,
            "error": None,
        }
        effective_timeout = self.timeout if timeout is None else timeout
        try:
            show = self._post_json(
                _SHOW_PATH, {"model": self.model}, timeout=effective_timeout, policy=_NO_RETRY
            )
        except Exception as exc:  # noqa: BLE001 - detection is best-effort
            result["error"] = str(exc)
            return result

        capabilities = show.get("capabilities")
        if isinstance(capabilities, list) and capabilities:
            self.capabilities = [str(c) for c in capabilities]
            result["capabilities"] = list(self.capabilities)

        context_length = self._context_length_from(show)
        if context_length:
            self.context_window = context_length
            result["context_window"] = context_length

        self._detect_remote(effective_timeout)
        result.update(
            {
                "detected": True,
                "remote": self.remote,
                "remote_host": self.remote_host,
            }
        )
        self._detected = True
        return result

    @staticmethod
    def _context_length_from(show: Dict[str, Any]) -> Optional[int]:
        """Pull ``<architecture>.context_length`` out of an ``/api/show`` body."""
        info = show.get("model_info")
        if not isinstance(info, dict):
            return None
        for key, value in info.items():
            if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                return value
        return None

    def _detect_remote(self, timeout: float) -> None:
        """Record whether ``/api/tags`` marks this model as provider-routed."""
        try:
            tags = self._get_json(_TAGS_PATH, timeout=timeout, policy=_NO_RETRY)
        except Exception:  # noqa: BLE001 - best-effort
            return
        for entry in tags.get("models") or []:
            if not isinstance(entry, dict):
                continue
            if (entry.get("name") or entry.get("model")) != self.model:
                continue
            host = entry.get("remote_host")
            if isinstance(host, str) and host:
                self.remote = True
                self.remote_host = host
            elif entry.get("remote_model"):
                self.remote = True
                self.remote_host = host or "remote"
            elif "cloud" in self.model.lower():
                self.remote = True
                self.remote_host = host or "remote"
            return

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
            prompt,
            temperature,
            max_new_tokens,
            top_p,
            top_k,
            repetition_penalty,
            stream=False,
            **kwargs,
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
            prompt,
            temperature,
            max_new_tokens,
            top_p,
            top_k,
            repetition_penalty,
            stream=True,
            **kwargs,
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
                    messages,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop=stop,
                    tools=tools,
                    response_format=response_format,
                    **kwargs,
                )
            )
            text = "".join(c.get("text", "") for c in chunks)
            tool_calls = []
            for c in chunks:
                if c.get("tool_calls"):
                    tool_calls.extend(c["tool_calls"])
            result: Dict[str, Any] = {
                "text": text,
                "finish_reason": "stop",
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
            response_format=response_format,
            **kwargs,
        )
        resp = self._post_json(_CHAT_PATH, payload)
        message = resp.get("message", {}) or {}
        return {
            "text": message.get("content", "") or "",
            "thinking": message.get("thinking"),
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
            messages,
            temperature,
            max_new_tokens,
            top_p,
            top_k,
            repetition_penalty,
            stream=True,
            stop=stop,
            tools=tools,
            response_format=response_format,
            **kwargs,
        )
        for obj in self._post_json_lines(_CHAT_PATH, payload):
            message = obj.get("message", {}) or {}
            chunk: Dict[str, Any] = {"text": message.get("content", "") or ""}
            if message.get("thinking"):
                chunk["thinking"] = message["thinking"]
            if message.get("tool_calls"):
                chunk["tool_calls"] = message["tool_calls"]
            if obj.get("done"):
                chunk["finish_reason"] = obj.get("done_reason", "stop")
                yield chunk
                break
            if chunk["text"] or chunk.get("tool_calls") or chunk.get("thinking"):
                yield chunk

    # -- embeddings ----------------------------------------------------

    def embed(self, texts: Union[str, List[str]], **kwargs: Any) -> List[List[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        payload = {"model": self.model, "input": items, "keep_alive": self.keep_alive}
        try:
            resp = self._post_json(_EMBED_PATH, payload)
        except RuntimeError as e:
            if "404" not in str(e):
                raise
            return [self._embed_single(t) for t in items]  # old server fallback
        embeddings = resp.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(items):
            raise RuntimeError("Unexpected Ollama /api/embed response: %r" % (resp,))
        return embeddings

    def _embed_single(self, text: str) -> List[float]:
        resp = self._post_json(
            _EMBEDDINGS_PATH,
            {"model": self.model, "prompt": text, "keep_alive": self.keep_alive},
        )
        embedding = resp.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Unexpected Ollama /api/embeddings response: %r" % (resp,))
        return embedding

    # -- introspection -------------------------------------------------

    def get_context_window(self) -> int:
        """Return the model's context window, detecting the real value if asked.

        With ``auto_detect_capabilities=True`` the first call asks the server;
        a server that is down or silent leaves the configured fallback in place.
        """
        if self.auto_detect_capabilities and not self._detected:
            self.detect_capabilities(timeout=_DETECT_TIMEOUT)
        return self.context_window

    def count_tokens(self, text: str) -> int:
        """~4 chars/token heuristic (server-side tokenization unavailable)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def model_info(self) -> Dict[str, Any]:
        info = super().model_info()
        info.update({"host": self.host, "keep_alive": self.keep_alive})
        if self.capabilities is not None:
            info["capabilities"] = list(self.capabilities)
        if self.remote:
            info["remote"] = True
            info["remote_host"] = self.remote_host
        return info

    def list_models(self) -> List[Dict[str, Any]]:
        """GET /api/tags, normalized to ``[{"id", "object", ...}]``.

        Each entry keeps the server's own ``capabilities`` (for example
        ``["completion", "tools", "thinking"]``), the model's real
        ``context_length``, and ``remote_host`` when Ollama routes the model to
        a provider such as ollama.com rather than running it on this machine.
        """
        resp = self._get_json(_TAGS_PATH)
        out = []
        for m in resp.get("models", []) or []:
            details = m.get("details", {}) or {}
            entry: Dict[str, Any] = {
                "id": m.get("name", m.get("model", "unknown")),
                "object": "model",
                "owned_by": "ollama",
                "size": m.get("size"),
                "digest": m.get("digest"),
                "modified_at": m.get("modified_at"),
                "details": details,
            }
            capabilities = m.get("capabilities")
            if isinstance(capabilities, list):
                entry["capabilities"] = [str(c) for c in capabilities]
            context_length = details.get("context_length")
            if isinstance(context_length, int):
                entry["context_length"] = context_length
            remote_host = m.get("remote_host")
            if isinstance(remote_host, str) and remote_host:
                entry["remote"] = True
                entry["remote_host"] = remote_host
            elif m.get("remote_model") or "cloud" in str(entry["id"]).lower():
                entry["remote"] = True
            out.append(entry)
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

    def _generate_payload(
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
        options_extra, top_level = self._split_kwargs(kwargs)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self._options(
                temperature, max_new_tokens, top_p, top_k, repetition_penalty, None, options_extra
            ),
        }
        payload.update(top_level)
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
        response_format: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..tools import ToolCall

        options_extra, top_level = self._split_kwargs(kwargs)
        # Ollama's native endpoint has no tool_choice parameter. Named choices
        # narrow the offered tools; enforcement remains model-dependent.
        choice = top_level.pop("tool_choice", None)
        if choice == "none":
            tools = None
        elif isinstance(choice, dict):
            name = choice.get("function", choice).get("name")
            tools = [item for item in tools or [] if item["function"]["name"] == name]
        native_messages = []
        for message in messages:
            native = dict(message)
            if isinstance(native.get("content"), list):
                texts = []
                images = list(native.get("images") or [])
                for part in native["content"]:
                    if isinstance(part, str):
                        texts.append(part)
                        continue
                    if not isinstance(part, dict):
                        raise ValueError("Ollama content parts must be text or image objects.")
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        image = part.get("image_url", {})
                        url = image.get("url", "") if isinstance(image, dict) else image
                        if (
                            not isinstance(url, str)
                            or not url.startswith("data:image/")
                            or ";base64," not in url
                        ):
                            raise ValueError(
                                "Ollama image_url parts require inline base64 data URLs."
                            )
                        images.append(url.split(";base64,", 1)[1])
                native["content"] = "\n".join(texts)
                if images:
                    native["images"] = images
            if native.get("tool_calls"):
                native.pop("function_call", None)
                calls = [ToolCall.from_dict(item) for item in native["tool_calls"]]
                native["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ]
            if native.get("role") == "tool":
                native["tool_name"] = native.pop("name", native.get("tool_name", ""))
                native.pop("tool_call_id", None)
            native_messages.append(native)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": native_messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self._options(
                temperature, max_new_tokens, top_p, top_k, repetition_penalty, stop, options_extra
            ),
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

    def _open_with_retry(
        self,
        request_factory: Any,
        method: str,
        path: str,
        timeout: Optional[float],
        policy: Optional[Any] = None,
    ) -> Any:
        """Run a request, retrying transient failures per :attr:`retry`.

        Retries a refused connection — the usual "``ollama serve`` has not
        started yet" case — and the transient statuses (``429``/``5xx``), which
        cover a model still being loaded. ``4xx`` client errors fail at once.
        """
        effective = self.retry if policy is None else policy
        effective_timeout = self.timeout if timeout is None else timeout
        attempt = 0
        while True:
            try:
                return urllib.request.urlopen(request_factory(), timeout=effective_timeout)
            except urllib.error.HTTPError as e:
                if effective.should_retry_status(e.code) and attempt < effective.max_retries:
                    effective.sleep(attempt + 1, retry_after_seconds(e.headers))
                    attempt += 1
                    continue
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    "Ollama request %s %s failed with HTTP %s: %s" % (method, path, e.code, body)
                ) from e
            except urllib.error.URLError as e:
                if attempt < effective.max_retries:
                    effective.sleep(attempt + 1)
                    attempt += 1
                    continue
                raise ConnectionError(
                    "Ollama server not reachable at %s (%s). Is Ollama running? "
                    "Start it with `ollama serve` and pull a model with "
                    "`ollama pull %s`." % (self.host, e.reason, self.model)
                ) from e

    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")

        def _request() -> Any:
            return urllib.request.Request(
                self.host + path,
                data=data,
                headers=self._headers(),
                method="POST",
            )

        try:
            with self._open_with_retry(_request, "POST", path, timeout, policy) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as e:
            raise RuntimeError("Ollama returned invalid JSON: %s" % e) from e

    def _post_json_lines(
        self,
        path: str,
        payload: Dict[str, Any],
        policy: Optional[Any] = None,
    ) -> Iterator[Dict[str, Any]]:
        """POST with ``stream=True``; yields one dict per JSON line.

        Only the connection phase is retried: once lines have been yielded,
        retrying would replay tokens the caller already received.
        """
        data = json.dumps(payload).encode("utf-8")

        def _request() -> Any:
            return urllib.request.Request(
                self.host + path,
                data=data,
                headers=self._headers(),
                method="POST",
            )

        with self._open_with_retry(_request, "POST", path, None, policy) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # skip keep-alive / partial lines

    def _get_json(
        self, path: str, timeout: Optional[float] = None, policy: Optional[Any] = None
    ) -> Dict[str, Any]:
        def _request() -> Any:
            return urllib.request.Request(self.host + path, headers=self._headers(), method="GET")

        with self._open_with_retry(_request, "GET", path, timeout, policy) as resp:
            return json.loads(resp.read().decode("utf-8"))
