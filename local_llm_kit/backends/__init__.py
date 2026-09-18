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
    "BACKENDS",
    "BACKEND_ALIASES",
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
}

#: Short backend name -> "module:Class" (module relative to this package).
BACKENDS: Dict[str, str] = {
    "transformers": ".transformers:TransformersBackend",
    "llamacpp": ".llamacpp:LlamaCppBackend",
    "llama_cpp": ".llamacpp:LlamaCppBackend",
    "llama.cpp": ".llamacpp:LlamaCppBackend",
    "echo": ".echo:EchoBackend",
    "mock": ".echo:EchoBackend",
    "ollama": ".ollama:OllamaBackend",
    "openai": ".openai_compat:OpenAICompatBackend",
    "openai_compat": ".openai_compat:OpenAICompatBackend",
    "openai-compat": ".openai_compat:OpenAICompatBackend",
    "vllm": ".openai_compat:OpenAICompatBackend",
    "lmstudio": ".openai_compat:OpenAICompatBackend",
    "llama-server": ".openai_compat:OpenAICompatBackend",
}

#: Canonical-name aliases kept as a separate dict for docs/CLI completion.
BACKEND_ALIASES: Dict[str, str] = {
    "mock": "echo",
    "llama_cpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "openai": "openai-compat",
    "openai_compat": "openai-compat",
    "vllm": "openai-compat",
    "lmstudio": "openai-compat",
    "llama-server": "openai-compat",
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
        }
        canonical = BACKEND_ALIASES.get(key, key)
        hint = hints.get(canonical)
        msg = "Backend %r could not be imported: %s" % (name, e)
        if hint:
            msg += ". Install it with `%s`." % hint
        raise ImportError(msg) from e


def get_backend(name: str, **kwargs: Any) -> BaseBackend:
    """Instantiate a backend from the registry: ``get_backend("ollama", model=...)``."""
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
                "Backend %r needs an optional dependency that is not "
                "installed: %s" % (name, e)
            ) from e
        return getattr(module, attr)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_BACKEND_IMPORTS.keys()))
