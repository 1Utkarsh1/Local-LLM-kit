"""
High-level functions for chat and completion.

These convenience helpers reuse cached :class:`LLM` clients so repeated
calls don't reload the model — the biggest performance flaw of the v0.1.x
helpers. The cache lives in this module and constructs clients through the
module-global :class:`LLM` name (importable/patchable as
``local_llm_kit.chat.LLM``).
"""
import asyncio
import json
import threading
from typing import Any, Dict, Iterator, List, Optional, Union

from .llm import LLM, get_client  # noqa: F401 (get_client re-exported for compat)

__all__ = ["chat", "complete", "embed", "achat", "acomplete", "get_client"]

_CHAT_CLIENT_CACHE: Dict[str, LLM] = {}
_CHAT_CACHE_LOCK = threading.Lock()


def _freeze(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except Exception:
        return repr(value)


def _cached_llm(
    model_path: str,
    backend: Optional[str] = None,
    backend_instance: Any = None,
    use_cache: bool = True,
    llm_kwargs: Optional[Dict[str, Any]] = None,
) -> LLM:
    """Return a cached shared LLM (module-global ``LLM`` name)."""
    llm_kwargs = dict(llm_kwargs or {})
    if backend_instance is not None or not use_cache:
        return LLM(  # noqa: F821 (module-global lookup keeps it patchable)
            model_path=model_path, backend=backend,
            backend_instance=backend_instance, use_cache=use_cache,
            **llm_kwargs,
        )
    key = "%s|%s|%s|%s" % (
        "%s.%s" % (LLM.__module__, getattr(LLM, "__qualname__", "?")),
        model_path,
        (backend or "auto").lower(),
        _freeze(llm_kwargs),
    )
    with _CHAT_CACHE_LOCK:
        hit = _CHAT_CLIENT_CACHE.get(key)
    if hit is not None:
        return hit
    client = LLM(model_path=model_path, backend=backend, **llm_kwargs)  # noqa: F821
    with _CHAT_CACHE_LOCK:
        _CHAT_CLIENT_CACHE.setdefault(key, client)
    return client


def clear_chat_cache() -> None:
    """Drop all clients cached by the module-level helpers."""
    with _CHAT_CACHE_LOCK:
        _CHAT_CLIENT_CACHE.clear()


def chat(
    messages: List[Dict[str, str]],
    model_path: str,
    backend: Optional[str] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Union[str, Dict[str, str]] = "auto",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    format: Optional[str] = None,  # noqa: A002 (legacy name, kept for compat)
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Union[str, Dict[str, Any], None] = None,
    response_format: Optional[Union[str, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    backend_instance: Any = None,
    use_cache: bool = True,
    **backend_kwargs: Any,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """Generate a chat completion (client is cached across calls).

    Accepts both the legacy ``functions``/``function_call``/``format``
    arguments and the modern ``tools``/``tool_choice``/``response_format``.
    Extra ``backend_kwargs`` are forwarded to the backend constructor.
    """
    llm = _cached_llm(
        model_path,
        backend=backend,
        backend_instance=backend_instance,
        use_cache=use_cache,
        llm_kwargs={"temperature": temperature,
                    "max_new_tokens": max_tokens or 512,
                    "backend_kwargs": backend_kwargs},
    )
    return llm.chat(
        messages=messages,
        functions=functions,
        function_call=function_call,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        format=format,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        seed=seed,
        stop=stop,
    )


def complete(
    prompt: str,
    model_path: str,
    backend: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    format: Optional[str] = None,  # noqa: A002 (legacy name, kept for compat)
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
    response_format: Optional[Union[str, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    backend_instance: Any = None,
    use_cache: bool = True,
    **backend_kwargs: Any,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """Generate a text completion (client is cached across calls)."""
    llm = _cached_llm(
        model_path,
        backend=backend,
        backend_instance=backend_instance,
        use_cache=use_cache,
        llm_kwargs={"temperature": temperature,
                    "max_new_tokens": max_tokens or 512,
                    "backend_kwargs": backend_kwargs},
    )
    return llm.complete(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        format=format,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        response_format=response_format,
        seed=seed,
        stop=stop,
    )


def embed(
    input: Union[str, List[str]],  # noqa: A002 (OpenAI field name)
    model_path: str,
    backend: Optional[str] = None,
    backend_instance: Any = None,
    use_cache: bool = True,
    **backend_kwargs: Any,
) -> Dict[str, Any]:
    """Generate OpenAI-shaped embeddings (client is cached across calls)."""
    llm = _cached_llm(
        model_path,
        backend=backend,
        backend_instance=backend_instance,
        use_cache=use_cache,
        llm_kwargs={"backend_kwargs": backend_kwargs},
    )
    return llm.embed(input)


async def achat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Async chat (``stream=False``)."""
    kwargs.pop("stream", None)
    return await asyncio.to_thread(chat, *args, **kwargs)


async def achat_stream(*args: Any, **kwargs: Any) -> Any:
    """Async chat streaming generator (``stream=True``)."""
    kwargs["stream"] = True
    gen = chat(*args, **kwargs)
    for chunk in gen:
        await asyncio.sleep(0)
        yield chunk


async def acomplete(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Async completion (``stream=False``)."""
    kwargs.pop("stream", None)
    return await asyncio.to_thread(complete, *args, **kwargs)


async def acomplete_stream(*args: Any, **kwargs: Any) -> Any:
    """Async completion streaming generator (``stream=True``)."""
    kwargs["stream"] = True
    gen = complete(*args, **kwargs)
    for chunk in gen:
        await asyncio.sleep(0)
        yield chunk
