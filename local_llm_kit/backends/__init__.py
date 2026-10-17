"""
Model backends for inference.

Lazy design: importing this package never imports heavy optional
dependencies (``torch``/``transformers``/``llama_cpp``). Backend classes
are imported on first attribute access (PEP 562) or via
:func:`get_backend_class` / :func:`get_backend`.

Registry:
    ``BACKENDS`` maps a short backend name to ``"<module>:<class>"``.
    ``get_backend(name, **kwargs)`` instantiates from the registry.
"""

import importlib
from typing import Any, Dict, List

from .base import BaseBackend

__all__ = [
    "BaseBackend",
    "TransformersBackend",
    "LlamaCppBackend",
    "EchoBackend",
    "MockBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "AirLLMBackend",
    "AppleFMBackend",
    "BACKENDS",
    "BACKEND_ALIASES",
    "OPENAI_SERVER_URLS",
    "get_backend_class",
    "get_backend",
]

#: name -> "relative_module:ClassName". Relative modules resolve against
#: this package so the registry survives renames of the top-level package.
_BACKEND_IMPORTS = {
    "TransformersBackend": (".transformers", "TransformersBackend"),
    "LlamaCppBackend": (".llamacpp", "LlamaCppBackend"),
    "EchoBackend": (".echo", "EchoBackend"),
    "MockBackend": (".echo", "MockBackend"),
    "OllamaBackend": (".ollama", "OllamaBackend"),
    "OpenAICompatBackend": (".openai_compat", "OpenAICompatBackend"),
    "AirLLMBackend": (".airllm", "AirLLMBackend"),
    "AppleFMBackend": (".apple_fm", "AppleFMBackend"),
}

#: Short backend name -> "module:Class" (module relative to this package).
BACKENDS: Dict[str, str] = {
    "transformers": ".transformers:TransformersBackend",
    "airllm": ".airllm:AirLLMBackend",
    "apple-fm": ".apple_fm:AppleFMBackend",
    "llamacpp": ".llamacpp:LlamaCppBackend",
    "llama_cpp": ".llamacpp:LlamaCppBackend",
    "llama.cpp": ".llamacpp:LlamaCppBackend",
    "echo": ".echo:EchoBackend",
    "mock": ".echo:EchoBackend",
    "ollama": ".ollama:OllamaBackend",
    "openai": ".openai_compat:OpenAICompatBackend",
    "openai_compat": ".openai_compat:OpenAICompatBackend",
    "openai-compat": ".openai_compat:OpenAICompatBackend",
    "apfel": ".openai_compat:OpenAICompatBackend",
    "vllm": ".openai_compat:OpenAICompatBackend",
    "lmstudio": ".openai_compat:OpenAICompatBackend",
    "llama-server": ".openai_compat:OpenAICompatBackend",
}

#: Canonical-name aliases kept as a separate dict for docs/CLI completion.
BACKEND_ALIASES: Dict[str, str] = {
    "mock": "echo",
    "apple": "apple-fm",
    "apple-foundation": "apple-fm",
    "foundation-models": "apple-fm",
    "foundationmodels": "apple-fm",
    "llama_cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "openai": "openai-compat",
    "openai_compat": "openai-compat",
    "apfel": "openai-compat",
    "vllm": "openai-compat",
    "lmstudio": "openai-compat",
    "llama-server": "openai-compat",
    "sglang": "openai-compat",
    "localai": "openai-compat",
    "jan": "openai-compat",
    "gpt4all": "openai-compat",
    "mlx-lm": "openai-compat",
    "koboldcpp": "openai-compat",
    "dummy": "echo",
    "test": "echo",
}

for _alias, _canonical in BACKEND_ALIASES.items():
    BACKENDS[_alias] = BACKENDS[_canonical]

# Connection presets; callers may override base_url for any server or port.
#
# NOTE on llama-server: llama.cpp moved its default listen port from 8080 to
# 9931 in October 2026 (ggml-org/llama.cpp#30159, merged 2026-10-09); the
# published Docker image still binds 8080 inside the container. This preset
# follows the upstream binary. On an older llama-server build, or against the
# container, pass the port explicitly:
#     backend_kwargs={"base_url": "http://localhost:8080/v1"}
# `local-llm-kit doctor` probes both ports and reports what is actually up.
OPENAI_SERVER_URLS: Dict[str, str] = {
    "apfel": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "localai": "http://localhost:8080/v1",
    "llama-server": "http://localhost:9931/v1",
    "jan": "http://127.0.0.1:1337/v1",
    "gpt4all": "http://127.0.0.1:4891/v1",
    "mlx-lm": "http://127.0.0.1:8080/v1",
    "koboldcpp": "http://localhost:5001/v1",
}


def _load(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attr)


def get_backend_class(name: str) -> Any:
    """Return the backend class for a registry name (lazy import).

    Raises:
        ValueError: for unknown names (message lists valid names).
        ImportError: with install hint when the backend's optional
            dependency is missing.
    """
    key = (name or "").strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            "Unknown backend %r. Available: %s"
            % (name, sorted(set(BACKENDS) - set(BACKEND_ALIASES)))
        )
    try:
        return _load(BACKENDS[key])
    except ImportError as e:
        hints = {
            "transformers": "pip install local-llm-kit[transformers]",
            "llamacpp": "pip install local-llm-kit[llamacpp]",
            "airllm": "pip install local-llm-kit[airllm]",
        }
        canonical = BACKEND_ALIASES.get(key, key)
        hint = hints.get(canonical)
        msg = "Backend %r could not be imported: %s" % (name, e)
        if hint:
            msg += ". Install it with `%s`." % hint
        raise ImportError(msg) from e


def get_backend(name: str, **kwargs: Any) -> BaseBackend:
    """Instantiate a backend from the registry: ``get_backend("ollama", model=...)``."""
    key = (name or "").strip().lower()
    if key in OPENAI_SERVER_URLS:
        kwargs.setdefault("base_url", OPENAI_SERVER_URLS[key])
        kwargs.setdefault("provider", key)
    return get_backend_class(name)(**kwargs)


def available_backends() -> List[str]:
    """Canonical backend names (excludes aliases)."""
    return sorted(set(BACKENDS) - set(BACKEND_ALIASES))


def __getattr__(name: str) -> Any:
    # PEP 562 lazy exports — keeps `import backends` light.
    if name in _BACKEND_IMPORTS:
        module_name, attr = _BACKEND_IMPORTS[name]
        try:
            module = importlib.import_module(module_name, __name__)
        except ImportError as e:
            raise ImportError(
                "Backend %r needs an optional dependency that is not " "installed: %s" % (name, e)
            ) from e
        return getattr(module, attr)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_BACKEND_IMPORTS.keys()))
