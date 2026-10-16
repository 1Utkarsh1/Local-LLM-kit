"""
OpenAI-compatible HTTP server for local-llm-kit.

Exposes a ``create_app()`` factory returning a FastAPI app so the toolkit
works with the OpenAI SDK, LangChain, Open WebUI, Continue, etc.::

    from local_llm_kit.server import create_app

    app = create_app(llm=my_llm)          # injected (great for tests)
    app = create_app(model_path="...")    # lazy-loads LLM on first request

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (non-stream + SSE stream, tools + response_format)
    POST /v1/completions        (non-stream + SSE stream)
    POST /v1/embeddings         (native llm.embed if present, else offline fallback)

Security defaults (see SECURITY review):
    * Tool/function calls are returned WITHOUT execution unless the request
      sets ``"execute_tools": true`` — remote callers can never trigger local
      function execution by default. Bind to loopback unless you understand
      the trust boundary of the tools you register.

Design notes:
    * No third-party imports at module top (stdlib + typing only) so the
      core install stays light. ``fastapi`` / ``uvicorn`` are imported
      lazily inside :func:`create_app` / :func:`run` / :func:`main` with
      helpful errors.
    * Plain ``dict`` + ``try/except`` request handling -- no pydantic
      models required, works with any FastAPI version.
    * Python 3.9 compatible (no ``X | Y`` unions, no ``match``).
"""

import argparse
import hashlib
import inspect
import json
import math
import time
from typing import Any, Dict, List, Optional

__all__ = ["create_app", "run", "main"]


# ---------------------------------------------------------------------------
# Pure-stdlib helpers (safe to import without fastapi installed)
# ---------------------------------------------------------------------------


def _now() -> int:
    return int(time.time())


def _coerce_message_content(content: Any) -> Optional[str]:
    """Coerce an OpenAI message ``content`` to plain text or None.

    Vision-style content (list of ``{"type": "text"|"image_url", ...}``
    parts) is reduced to its text parts joined by newlines so older
    backends keep working. Image parts are noted with a placeholder
    (vision inference itself is backend-dependent).
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []  # type: List[str]
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type", "text")
                if ptype == "text":
                    if part.get("text") is not None:
                        texts.append(str(part.get("text", "")))
                elif ptype in ("image_url", "image"):
                    url = part.get("image_url", part.get("image", ""))
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    texts.append("[image: %s]" % url)
                else:
                    texts.append(str(part))
            elif part is not None:
                texts.append(str(part))
        return "\n".join(texts)
    return str(content)


def _coerce_messages(messages: Any) -> List[Dict[str, Any]]:
    coerced = []  # type: List[Dict[str, Any]]
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        out = dict(msg)
        if "content" in out:
            content = out.get("content")
            out["content"] = (
                content if isinstance(content, list) else _coerce_message_content(content)
            )
        coerced.append(out)
    return coerced


def _extract_functions(body: Dict[str, Any]):
    """Return ``(functions, function_call)`` for the LLM.chat API.

    Accepts both legacy ``functions``/``function_call`` and modern
    ``tools``/``tool_choice``. Modern tool specs are converted to the
    legacy function-spec shape the backend understands.
    """
    functions = body.get("functions")
    function_call = body.get("function_call", "auto")

    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    if tools and not functions:
        converted = []  # type: List[Dict[str, Any]]
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type", "function") != "function":
                continue
            fn = tool.get("function", tool)
            if isinstance(fn, dict) and fn.get("name"):
                converted.append(fn)
        functions = converted or None

    if tool_choice is not None and "function_call" not in body:
        # tool_choice: "auto" | "none" | "required" | {"type":"function","function":{"name":...}}
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                function_call = "auto"
            elif tool_choice in ("auto", "none"):
                function_call = tool_choice
            else:
                function_call = "auto"
        elif isinstance(tool_choice, dict):
            fn = tool_choice.get("function", {})
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                function_call = {"name": name}
            else:
                function_call = "auto"

    return functions, function_call, tools, tool_choice


def _extract_format(body: Dict[str, Any]) -> Optional[str]:
    """Normalise ``response_format`` / ``format`` to ``"json"`` or None."""
    fmt = body.get("format")
    if fmt == "json":
        return "json"
    rf = body.get("response_format")
    if isinstance(rf, dict):
        rtype = rf.get("type", "")
        # {"type": "json_object"} or {"type": "json_schema", ...}
        if "json" in str(rtype):
            return "json"
    elif isinstance(rf, str) and "json" in rf:
        return "json"
    return None


def _model_id_for(llm: Any, fallback: Optional[str]) -> str:
    for attr in ("model_path", "model_name", "model"):
        try:
            val = getattr(llm, attr, None)
        except Exception:
            val = None
        if isinstance(val, str) and val:
            return val
    return fallback or "local-model"


def _ensure_tool_calls(message: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror legacy ``function_call`` as modern ``tool_calls`` (in place)."""
    fn_call = message.get("function_call")
    if isinstance(fn_call, dict) and "tool_calls" not in message:
        try:
            args = fn_call.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args)
            message["tool_calls"] = [
                {
                    "id": "call_%s" % (fn_call.get("name", "fn"),),
                    "type": "function",
                    "function": {
                        "name": fn_call.get("name", ""),
                        "arguments": args,
                    },
                }
            ]
        except Exception:
            pass
    return message


def _pseudo_embedding(text: str, dim: int = 32) -> List[float]:
    """Deterministic offline embedding fallback (hash-based, L2-normalised)."""
    vec = []  # type: List[float]
    counter = 0
    while len(vec) < dim:
        digest = hashlib.sha256(("%s#%d" % (text, counter)).encode("utf-8")).digest()
        for byte in digest:
            vec.append((byte / 127.5) - 1.0)
            if len(vec) >= dim:
                break
        counter += 1
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _coerce_vectors(result: Any, n: int) -> Optional[List[List[float]]]:
    """Coerce an embedding result to ``n`` float vectors, or None.

    Accepts any iterable of vectors (list/tuple/numpy array/generator), the
    OpenAI-shaped ``{"data": [{"embedding": [...]}, ...]}`` dicts returned by
    :meth:`local_llm_kit.LLM.embed`, and per-item ``{"embedding": [...]}`` or
    single-result ``{"data": [{"embedding": [...]}]}`` dicts. Returns None
    when the shape does not line up with ``n`` inputs, so callers never pair
    a text with the wrong vector.
    """
    if isinstance(result, dict):
        if "data" not in result:
            return None
        result = result.get("data")
    try:
        items = list(result)
    except TypeError:
        return None
    if len(items) != n:
        return None
    vectors = []  # type: List[List[float]]
    for item in items:
        if isinstance(item, dict):
            if "embedding" not in item and isinstance(item.get("data"), list) and item["data"]:
                item = item["data"][0]  # single-result envelope for this text
            if isinstance(item, dict):
                item = item.get("embedding")
        try:
            vectors.append(list(map(float, item)))
        except (TypeError, ValueError):
            return None
    return vectors


def _generation_params(body: Dict[str, Any]) -> Dict[str, Any]:
    """Validate shared chat/completion options before calling the backend."""
    from .llm import _normalize_response_format

    temperature = body.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
    ):
        raise ValueError("'temperature' must be a finite number.")
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0
    ):
        raise ValueError("'max_tokens' must be a non-negative integer.")
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": _normalize_response_format(
            body.get("response_format"), _extract_format(body)
        ),
    }
    return {key: value for key, value in params.items() if value is not None}


def _try_get_embeddings(llm: Any, texts: List[str]) -> Optional[List[List[float]]]:
    """Use native embedding support if the llm provides it, else None."""
    for method_name in ("embed", "embeddings", "get_embeddings"):
        method = getattr(llm, method_name, None)
        if callable(method):
            try:
                result = method(texts)
            except TypeError:
                try:
                    result = [method(t) for t in texts]
                except Exception:
                    continue
            except Exception:
                continue
            vectors = _coerce_vectors(result, len(texts))
            if vectors is not None:
                return vectors
    return None


def _call_llm_chat(llm_obj: Any, chat_kwargs: Dict[str, Any]) -> Any:
    """Adapt older signatures before calling; never retry a runtime TypeError."""
    modern_keys = ("tools", "tool_choice", "response_format", "execute_tools")
    kwargs = dict(chat_kwargs)
    try:
        params = inspect.signature(llm_obj.chat).parameters
    except (TypeError, ValueError):
        return llm_obj.chat(**kwargs)
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        if (
            "execute_tools" not in params
            and not kwargs.get("execute_tools", False)
            and (kwargs.get("tools") or kwargs.get("functions"))
        ):
            raise TypeError("This LLM cannot disable tool execution; use a current client.")
        kwargs = {
            key: value for key, value in kwargs.items() if key not in modern_keys or key in params
        }
    return llm_obj.chat(**kwargs)


def _call_llm_complete(llm_obj: Any, complete_kwargs: Dict[str, Any]) -> Any:
    kwargs = dict(complete_kwargs)
    try:
        params = inspect.signature(llm_obj.complete).parameters
    except (TypeError, ValueError):
        return llm_obj.complete(**kwargs)
    if "response_format" not in params and not any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    ):
        kwargs.pop("response_format", None)
    return llm_obj.complete(**kwargs)


# ---------------------------------------------------------------------------
# App factory (fastapi imported lazily so core stays light)
# ---------------------------------------------------------------------------


def create_app(
    llm: Any = None,
    model_path: Optional[str] = None,
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    **llm_kwargs: Any,
):
    """Create and return a FastAPI app serving an OpenAI-compatible API.

    Args:
        llm: Pre-built LLM-like object with ``.chat()`` / ``.complete()``
            methods (duck-typed). Ideal for tests / injected mocks.
            If ``None``, the LLM is constructed lazily from ``model_path``
            on first request.
        model_path: Model path used for lazy ``LLM`` construction and as
            the default model id when no llm is injected yet.
        backend: Backend name forwarded to ``LLM`` for lazy construction.
        model_name: Override for the model id reported by ``/v1/models``.
        **llm_kwargs: Extra kwargs forwarded to ``LLM(...)``.

    Raises:
        ImportError: If ``fastapi`` is not installed, with install hint.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The local-llm-kit server requires 'fastapi'. "
            'Install it with: pip install "local-llm-kit[server]" '
            "or: pip install fastapi uvicorn"
        ) from exc

    state = {"llm": llm}  # mutable holder so tests can swap llm after creation

    def _get_llm() -> Any:
        if state["llm"] is not None:
            return state["llm"]
        nonlocal_model_path = model_path
        if nonlocal_model_path is None:
            if (backend or "").lower() in ("echo", "mock"):
                # Model-free backends work without a model id.
                nonlocal_model_path = (backend or "echo").lower()
            else:
                raise RuntimeError(
                    "No LLM available: pass llm=... or model_path=... to create_app()."
                )
        # Lazy import: keeps `import local_llm_kit.server` light and avoids
        # pulling heavy backends (torch/llama.cpp) until first request.
        from .llm import LLM

        state["llm"] = LLM(model_path=nonlocal_model_path, backend=backend, **llm_kwargs)
        return state["llm"]

    def _model_id() -> str:
        if state["llm"] is not None:
            return _model_id_for(state["llm"], model_name or model_path)
        return model_name or model_path or "local-model"

    def _error(message: str, status: int = 400, err_type: str = "invalid_request_error"):
        return JSONResponse(
            status_code=status,
            content={"error": {"message": message, "type": err_type}},
        )

    app = FastAPI(title="local-llm-kit server", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- health ---------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def root():
        return {"status": "ok", "service": "local-llm-kit", "model": _model_id()}

    # -- models ---------------------------------------------------------
    @app.get("/v1/models")
    @app.get("/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": _model_id(),
                    "object": "model",
                    "created": _now(),
                    "owned_by": "local",
                }
            ],
        }

    # -- chat completions ----------------------------------------------
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error("Invalid JSON body.")
        if not isinstance(body, dict):
            return _error("JSON body must be an object.")
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return _error("'messages' must be a non-empty list.")

        try:
            llm_obj = _get_llm()
        except RuntimeError as exc:
            return _error(str(exc), status=503, err_type="server_error")

        coerced = _coerce_messages(messages)
        functions, function_call, tools, tool_choice = _extract_functions(body)
        fmt = _extract_format(body)
        stream = bool(body.get("stream", False))
        # Security default: never execute local tools for remote callers
        # unless they explicitly opt in per request.
        execute_tools = body.get("execute_tools", False)
        if not isinstance(execute_tools, bool):
            return _error("'execute_tools' must be a boolean.")
        try:
            generation_params = _generation_params(body)
        except (TypeError, ValueError) as exc:
            return _error(str(exc))
        logprobs = bool(body.get("logprobs", False))
        top_logprobs = body.get("top_logprobs")

        chat_kwargs = {
            "messages": coerced,
            "functions": functions,
            "function_call": function_call,
            "tools": tools,
            "tool_choice": tool_choice,
            "execute_tools": execute_tools,
            **generation_params,
            "format": fmt,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        # Drop None values so LLM defaults apply.
        chat_kwargs = {k: v for k, v in chat_kwargs.items() if v is not None}
        if "function_call" not in chat_kwargs:
            chat_kwargs["function_call"] = function_call

        requested_model = body.get("model") or _model_id()

        if not stream:
            try:
                result = _call_llm_chat(llm_obj, dict(chat_kwargs, stream=False))
            except Exception as exc:
                return _error("Chat failed: %s" % exc, status=500, err_type="server_error")
            try:
                if isinstance(result, dict):
                    result.setdefault("model", requested_model)
                    choices = result.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message")
                        if isinstance(msg, dict):
                            _ensure_tool_calls(msg)
                return JSONResponse(content=result)
            except Exception as exc:
                return _error("Failed to serialise response: %s" % exc, status=500)

        # -- streaming (SSE) --
        def event_iter():
            try:
                chunks = _call_llm_chat(llm_obj, dict(chat_kwargs, stream=True))
            except Exception as exc:
                yield "data: %s\n\n" % json.dumps({"error": str(exc)})
                yield "data: [DONE]\n\n"
                return
            try:
                # Allow mock llms that return a single dict even in stream mode.
                if isinstance(chunks, dict):
                    chunks = [chunks]
                for chunk in chunks:
                    try:
                        if isinstance(chunk, dict):
                            chunk.setdefault("model", requested_model)
                            choices = chunk.get("choices", [])
                            if choices and isinstance(choices[0], dict):
                                delta = choices[0].get("delta")
                                if isinstance(delta, dict):
                                    fn = delta.get("function_call")
                                    if isinstance(fn, dict) and "tool_calls" not in delta:
                                        delta["tool_calls"] = [
                                            {
                                                "id": "call_%s" % fn.get("name", "fn"),
                                                "type": "function",
                                                "function": fn,
                                            }
                                        ]
                        yield "data: %s\n\n" % json.dumps(chunk)
                    except Exception as exc:
                        yield "data: %s\n\n" % json.dumps({"error": str(exc)})
                        break
            except Exception as exc:
                yield "data: %s\n\n" % json.dumps({"error": str(exc)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_iter(), media_type="text/event-stream")

    # -- legacy completions ---------------------------------------------
    @app.post("/v1/completions")
    @app.post("/completions")
    async def completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error("Invalid JSON body.")
        if not isinstance(body, dict):
            return _error("JSON body must be an object.")
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            # Token-id lists or multi-prompts: join into text for local backends.
            prompt = "".join(str(p) for p in prompt)
        if not isinstance(prompt, str) or not prompt:
            return _error("'prompt' must be a non-empty string.")

        try:
            llm_obj = _get_llm()
        except RuntimeError as exc:
            return _error(str(exc), status=503, err_type="server_error")

        fmt = _extract_format(body)
        try:
            generation_params = _generation_params(body)
        except (TypeError, ValueError) as exc:
            return _error(str(exc))
        stream = bool(body.get("stream", False))
        requested_model = body.get("model") or _model_id()
        complete_kwargs = {
            "prompt": prompt,
            **generation_params,
            "format": fmt,
            "logprobs": body.get("logprobs", False),
            "top_logprobs": body.get("top_logprobs"),
        }
        complete_kwargs = {k: v for k, v in complete_kwargs.items() if v is not None}

        if not stream:
            try:
                result = _call_llm_complete(llm_obj, dict(complete_kwargs, stream=False))
            except Exception as exc:
                return _error("Completion failed: %s" % exc, status=500, err_type="server_error")
            if isinstance(result, dict):
                result.setdefault("model", requested_model)
                return JSONResponse(content=result)
            return _error("Backend returned non-dict completion.", status=500)

        def event_iter():
            try:
                chunks = _call_llm_complete(llm_obj, dict(complete_kwargs, stream=True))
            except Exception as exc:
                yield "data: %s\n\n" % json.dumps({"error": str(exc)})
                yield "data: [DONE]\n\n"
                return
            try:
                if isinstance(chunks, dict):
                    chunks = [chunks]
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        chunk.setdefault("model", requested_model)
                    yield "data: %s\n\n" % json.dumps(chunk)
            except Exception as exc:
                yield "data: %s\n\n" % json.dumps({"error": str(exc)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_iter(), media_type="text/event-stream")

    # -- embeddings ------------------------------------------------------
    @app.post("/v1/embeddings")
    @app.post("/embeddings")
    async def embeddings(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error("Invalid JSON body.")
        if not isinstance(body, dict):
            return _error("JSON body must be an object.")
        raw_input = body.get("input", "")
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, list):
            if any(not isinstance(text, str) for text in raw_input):
                return _error("'input' must be a string or list of strings.")
            texts = raw_input
        else:
            return _error("'input' must be a string or list of strings.")
        if not texts:
            return _error("'input' must be non-empty.")

        try:
            llm_obj = _get_llm()
        except RuntimeError:
            llm_obj = None
        requested_model = body.get("model") or _model_id()

        vectors = None
        if llm_obj is not None:
            try:
                vectors = _try_get_embeddings(llm_obj, texts)
            except Exception:
                vectors = None
        if vectors is None:
            vectors = [_pseudo_embedding(t) for t in texts]

        prompt_tokens = sum(len(t.split()) for t in texts)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vec}
                for i, vec in enumerate(vectors)
            ],
            "model": requested_model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    return app


# ---------------------------------------------------------------------------
# uvicorn runner
# ---------------------------------------------------------------------------


def run(app: Any, host: str = "127.0.0.1", port: int = 8000, **kwargs: Any) -> None:
    """Serve a FastAPI ``app`` with uvicorn (requires the ``server`` extra)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Serving requires 'uvicorn'. " 'Install it with: pip install "local-llm-kit[server]"'
        ) from exc
    uvicorn.run(app, host=host, port=port, **kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python -m local_llm_kit.server --model <path>``."""
    parser = argparse.ArgumentParser(description="local-llm-kit OpenAI-compatible server")
    parser.add_argument("--model", dest="model_path", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args(argv)

    app = create_app(
        model_path=args.model_path,
        backend=args.backend,
        model_name=args.model_name,
    )
    run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
