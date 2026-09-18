"""
Command-line interface for local_llm_kit.

Subcommands:
    chat        Interactive (or one-shot) chat with a model.
    complete    Complete a prompt (one-shot text completion).
    serve       Serve a model over HTTP via an OpenAI-compatible API.
    pull        Download a model using the models.py helper.
    list        List cached/downloaded models (aliases: list-models, models).
    embed       Generate embeddings for one or more texts (JSON output).

Backwards compatibility:
    The legacy ``chat`` / ``complete`` flags (``--model``, ``--backend``,
    ``--system``, ``--temperature``, ``--max-tokens``, ``--stream``,
    ``--json``, ``--functions``, ``--function-call``, ``--gpu-layers``,
    ``--device``) are all preserved. New modern flags (``--tools``,
    ``--execute-tools``, ``--response-format``, ``--message``, ``--base-url``)
    are added alongside them.

Only the standard library is used here (argparse, json, os, sys, inspect).
Heavy/optional dependencies (transformers, llama-cpp, fastapi, uvicorn,
model-download helpers) are imported lazily inside handlers so that
``--help`` and ``--version`` always work and missing extras produce a
clear, actionable error instead of a traceback.
"""

import argparse
import inspect
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import readline  # noqa: F401  (history + arrow keys on Unix; optional)
except ImportError:  # pragma: no cover - Windows has no readline
    readline = None  # type: ignore

__all__ = [
    "BACKEND_CHOICES",
    "build_parser",
    "main",
    "handle_chat_command",
    "handle_completion_command",
    "handle_serve_command",
    "handle_pull_command",
    "handle_list_command",
    "handle_embed_command",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND_CHOICES = [
    "transformers",
    "llamacpp",
    "llama-cpp",
    "ollama",
    "openai-compat",
    "openai_compat",
    "echo",
    "mock",
    "auto",
]

_BACKEND_ALIASES = {
    "llama-cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "openai_compat": "openai-compat",
    "openai": "openai-compat",
    "mock": "echo",
    "dummy": "echo",
    "test": "echo",
}

_PULL_FUNC_NAMES = (
    "download_model",
    "pull_model",
    "download",
    "fetch_model",
    "pull",
    "download_hf_model",
    "download_from_hub",
)

_LIST_FUNC_NAMES = (
    "list_cached_models",
    "list_models",
    "list_cached",
    "cached_models",
    "list_downloaded_models",
    "list_local_models",
)

_EMBED_METHOD_NAMES = (
    "embed",
    "embeddings",
    "get_embeddings",
    "create_embeddings",
    "embedding",
    "embed_texts",
)

_MODEL_ARG_KEYS = ("model", "model_id", "model_name", "repo_id", "repo", "path", "name")
_OUTPUT_DIR_KEYS = ("output_dir", "dest_dir", "destination", "out_dir", "target_dir", "cache_dir")
_REVISION_KEYS = ("revision", "rev", "ref")
_FORCE_KEYS = ("force", "overwrite", "force_download")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _eprint(msg: str) -> None:
    """Print a message to stderr."""
    print(msg, file=sys.stderr)


def _fail(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    """Print an error to stderr and exit. (Only used inside handlers.)"""
    _eprint("Error: {0}".format(msg))
    sys.exit(code)


def _get_version() -> str:
    """Return the package version without requiring optional dependencies."""
    try:
        from . import __version__ as _v
        return str(_v)
    except Exception:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("local-llm-kit")
        except PackageNotFoundError:
            pass
        except Exception:
            pass
    except Exception:
        pass
    return "unknown"


def _normalize_backend(name: Optional[str]) -> Optional[str]:
    """Normalize backend aliases; 'auto'/None means auto-detect."""
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in ("auto", "none", ""):
        return None
    return _BACKEND_ALIASES.get(key, key)


def _load_json_file(path: str, max_bytes: int = 1024 * 1024) -> Any:
    """Load a JSON file or exit with a clear error (size-capped at 1 MB)."""
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            _fail("file too large (>{0} bytes): {1}".format(max_bytes, path))
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        _fail("file not found: {0}".format(path))
    except json.JSONDecodeError as exc:
        _fail("invalid JSON in {0}: {1}".format(path, exc))
    except OSError as exc:
        _fail("could not read {0}: {1}".format(path, exc))
    return None  # unreachable (keeps type checkers happy)


def _load_json_value_or_file(value: str, name: str) -> Any:
    """Accept either a path to a JSON file or an inline JSON string."""
    if os.path.exists(value):
        return _load_json_file(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        _fail(
            "{0} must be a path to a JSON file or inline JSON; "
            "got: {1!r}".format(name, value)
        )
    return None  # unreachable


def _load_tools(
    functions_path: Optional[str],
    tools_paths: Optional[List[str]],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Load legacy ``functions`` + modern ``tools`` JSON files.

    Returns ``(legacy_functions, modern_tools)`` where either entry may be
    ``None`` when nothing was provided. A tools file may contain OpenAI-style
    tool objects (``{"type": "function", "function": {...}}``) or bare
    function specs (``{"name": ...}``); both are normalized.
    """
    raw_items: List[Any] = []

    def _collect(path: str) -> None:
        data = _load_json_file(path)
        if isinstance(data, dict):
            # Also accept {"tools": [...]} / {"functions": [...]} wrappers.
            for wrapper in ("tools", "functions"):
                if isinstance(data.get(wrapper), list):
                    raw_items.extend(data[wrapper])
                    return
            raw_items.append(data)
        elif isinstance(data, list):
            raw_items.extend(data)
        else:
            _fail(
                "tools file {0} must contain a JSON object or array, "
                "got {1}".format(path, type(data).__name__)
            )

    if functions_path:
        _collect(functions_path)
    for path in tools_paths or []:
        _collect(path)

    if not raw_items:
        return None, None

    modern_tools: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            _fail("each tool must be a JSON object, got: {0!r}".format(item))
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            modern_tools.append(item)
        elif "name" in item:
            modern_tools.append({"type": "function", "function": item})
        else:
            _fail(
                "tool object must have a 'name' key or be an OpenAI-style "
                "tool; got keys: {0}".format(sorted(item.keys()))
            )
    legacy_functions = [t["function"] for t in modern_tools]
    return legacy_functions, modern_tools


def _parse_choice(value: Optional[str], name: str) -> Optional[Any]:
    """Parse ``--function-call`` / ``--tool-choice`` values.

    Accepts ``auto`` / ``none`` or inline JSON such as
    ``'{"name": "get_weather"}'``.
    """
    if value is None:
        return None
    text = value.strip()
    if text in ("auto", "none"):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        _fail(
            "invalid {0} format: {1!r} "
            "(expected 'auto', 'none', or JSON)".format(name, value)
        )
    return None  # unreachable


def _resolve_formats(args: argparse.Namespace) -> Tuple[Optional[str], Optional[Any]]:
    """Resolve structured-output flags to (legacy_format, modern_format).

    ``--json`` (legacy) and ``--response-format`` (modern) are both honored.
    ``--response-format`` accepts ``text``/``json``/``json_object``, inline
    JSON (e.g. a JSON-schema response_format object), or a path to a JSON file
    holding a schema / response_format object.
    """
    requested = getattr(args, "response_format", None)
    legacy_json = bool(getattr(args, "json", False))

    if requested is None:
        if legacy_json:
            return "json", {"type": "json_object"}
        return None, None

    text = str(requested).strip()
    if text.lower() in ("text", "none", ""):
        return None, None
    if text.lower() in ("json", "json_object", "json-object"):
        return "json", {"type": "json_object"}
    loaded = _load_json_value_or_file(text, "--response-format")
    if isinstance(loaded, dict) and "type" in loaded and "function" not in loaded:
        # Looks like a full {"type": ..., ...} response_format object.
        return "json", loaded
    # Otherwise treat it as a JSON schema for structured output.
    return "json", {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": loaded},
    }


def _tool_choice_to_function_call(choice: Optional[Any]) -> Any:
    """Map a modern tool_choice value back to legacy function_call."""
    if choice is None:
        return "auto"
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        name = choice.get("name")
        if name:
            return {"name": name}
        fn = choice.get("function", {})
        if isinstance(fn, dict) and fn.get("name"):
            return {"name": fn["name"]}
    return "auto"


def _build_backend_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect backend-specific kwargs from parsed args (stdlib only)."""
    kwargs: Dict[str, Any] = {}
    backend = _normalize_backend(getattr(args, "backend", None))

    gpu_layers = getattr(args, "gpu_layers", None)
    if backend == "llamacpp" and gpu_layers is not None:
        kwargs["n_gpu_layers"] = gpu_layers

    device = getattr(args, "device", None)
    if backend == "transformers" and device:
        kwargs["device"] = device

    base_url = getattr(args, "base_url", None) or os.environ.get(
        "OPENAI_BASE_URL", ""
    ) or os.environ.get("OLLAMA_HOST", "")
    if base_url and backend in ("ollama", "openai-compat", None):
        kwargs["base_url"] = base_url
    api_key = getattr(args, "api_key", None) or os.environ.get("OPENAI_API_KEY", "")
    if api_key and backend in ("ollama", "openai-compat", None):
        # Only pass through when the kwarg was explicitly given or env set.
        if getattr(args, "api_key", None) or "OPENAI_API_KEY" in os.environ:
            kwargs["api_key"] = api_key
    return kwargs


def _create_llm(args: argparse.Namespace, backend_kwargs: Dict[str, Any]) -> Any:
    """Lazily import and instantiate LLM with graceful error messages."""
    try:
        from .llm import LLM
    except ImportError as exc:
        _fail(
            "could not import the LLM core ({0}). "
            "Is local-llm-kit installed correctly?".format(exc)
        )
    model = getattr(args, "model", None)
    if not model:
        _fail("a model is required (pass --model/-m).")
    backend = _normalize_backend(getattr(args, "backend", None))
    try:
        return LLM(
            model_path=model,
            backend=backend,
            temperature=getattr(args, "temperature", 0.7),
            max_new_tokens=getattr(args, "max_tokens", 512) or 512,
            backend_kwargs=backend_kwargs,
        )
    except ImportError as exc:
        missing = str(exc)
        _fail(
            "missing optional dependency for backend {0!r}: {1}\n"
            "Install extras, e.g.: pip install 'local-llm-kit[{2}]'".format(
                backend or "auto",
                missing,
                "transformers" if backend == "transformers"
                else "llamacpp" if backend == "llamacpp"
                else "ollama" if backend in ("ollama", "openai-compat")
                else "all",
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        _fail("could not load model {0!r}: {1}".format(model, exc))
    return None  # unreachable


def _chat_once(
    llm: Any,
    messages: List[Dict[str, Any]],
    *,
    functions: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    legacy_format: Optional[str] = None,
    modern_format: Optional[Any] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    execute_tools: bool = False,
    logprobs: bool = False,
) -> Any:
    """Call ``llm.chat`` bridging modern and legacy signatures.

    Prefers modern kwargs (``tools``/``tool_choice``/``response_format``/
    ``execute_tools``) when the installed ``LLM.chat`` accepts them, and
    falls back to legacy (``functions``/``function_call``/``format``)
    otherwise — keeping this CLI compatible with both v0.1.x and v0.2.x.
    """
    try:
        params = set(inspect.signature(llm.chat).parameters)
    except (TypeError, ValueError):
        params = set()

    kwargs: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if logprobs:
        kwargs["logprobs"] = True

    if tools is not None or functions is not None:
        modern = (
            list(tools)
            if tools is not None
            else [{"type": "function", "function": f} for f in (functions or [])]
        )
        if "tools" in params or not params:
            if tools is not None or "tools" in params:
                kwargs["tools"] = modern
            elif functions is not None:
                kwargs["functions"] = functions
        elif "functions" in params:
            kwargs["functions"] = (
                functions
                if functions is not None
                else [t["function"] for t in modern]
            )
        else:  # unknown signature: try modern, fall back on TypeError below
            kwargs["tools"] = modern

        if tool_choice is not None:
            if "tool_choice" in params or not params:
                kwargs["tool_choice"] = tool_choice
            elif "function_call" in params:
                kwargs["function_call"] = _tool_choice_to_function_call(tool_choice)
            else:
                kwargs["tool_choice"] = tool_choice
    if modern_format is not None and ("response_format" in params or not params):
        kwargs["response_format"] = modern_format
    if legacy_format is not None and ("format" in params or not params):
        kwargs["format"] = legacy_format
    if execute_tools and ("execute_tools" in params or not params):
        kwargs["execute_tools"] = True

    # Drop Nones the callee may not expect (legacy chat.py tolerates them,
    # but third-party-compatible signatures may not).
    kwargs = {k: v for k, v in kwargs.items() if v is not None or k == "messages"}

    try:
        return llm.chat(**kwargs)
    except TypeError:
        # Last-resort legacy fallback for unfamiliar signatures.
        fallback: Dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }
        if functions is not None:
            fallback["functions"] = functions
        elif tools is not None:
            fallback["functions"] = [t["function"] for t in tools]
        if tool_choice is not None:
            fallback["function_call"] = _tool_choice_to_function_call(tool_choice)
        if legacy_format is not None:
            fallback["format"] = legacy_format
        if temperature is not None:
            fallback["temperature"] = temperature
        if max_tokens is not None:
            fallback["max_tokens"] = max_tokens
        return llm.chat(**fallback)


def _complete_once(
    llm: Any,
    prompt: str,
    *,
    legacy_format: Optional[str] = None,
    modern_format: Optional[Any] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
) -> Any:
    """Call ``llm.complete`` bridging modern and legacy signatures."""
    try:
        params = set(inspect.signature(llm.complete).parameters)
    except (TypeError, ValueError):
        params = set()
    kwargs: Dict[str, Any] = {"prompt": prompt, "stream": stream}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if modern_format is not None and ("response_format" in params or not params):
        kwargs["response_format"] = modern_format
    if legacy_format is not None and ("format" in params or not params):
        kwargs["format"] = legacy_format
    try:
        return llm.complete(**kwargs)
    except TypeError:
        fallback = {"prompt": prompt, "stream": stream}
        if legacy_format is not None:
            fallback["format"] = legacy_format
        return llm.complete(**fallback)


def _extract_delta_text(chunk: Any) -> str:
    """Best-effort extraction of streamed chat text from a chunk dict."""
    try:
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return delta.get("content") or ""
    except AttributeError:
        return ""


def _extract_chunk_text(chunk: Any) -> str:
    """Best-effort extraction of streamed completion text from a chunk."""
    try:
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("text") or ""
    except AttributeError:
        return ""


def _print_tool_calls_from_message(message: Dict[str, Any]) -> bool:
    """Pretty-print tool/function calls found in a chat message.

    Returns True when something was printed.
    """
    printed = False
    tool_calls = message.get("tool_calls")
    if tool_calls:
        print("Assistant (tool calls): {0}".format(json.dumps(tool_calls, indent=2)))
        printed = True
    func_call = message.get("function_call")
    if func_call:
        print("Assistant (function call): {0}".format(json.dumps(func_call, indent=2)))
        printed = True
    return printed


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common_llm_args(parser: argparse.ArgumentParser, *, require_model: bool) -> None:
    """Shared model/backend/generation flags for chat/complete/embed."""
    parser.add_argument(
        "--model", "-m",
        required=require_model,
        default=None,
        help="Path to the model, model ID, or model name.",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=BACKEND_CHOICES,
        default=None,
        help="Backend to use: transformers, llamacpp, ollama, "
             "openai-compat, echo (offline mock). Default: auto-detect.",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens", "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for 'ollama' / 'openai-compat' backends "
             "(also honors OPENAI_BASE_URL / OLLAMA_HOST).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for 'openai-compat' servers (also honors OPENAI_API_KEY).",
    )


def _add_common_backend_tuning(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of GPU layers for llama.cpp (-1 for all).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for transformers ('cpu', 'cuda', 'mps').",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser (split out for testability)."""
    parser = argparse.ArgumentParser(
        prog="local-llm-kit",
        description="Local LLM Kit - use local LLMs with an OpenAI-like API.",
        epilog=(
            "Examples:\n"
            "  local-llm-kit chat -m ./models/mistral.gguf\n"
            "  local-llm-kit chat -m llama3 --backend ollama --system 'Be concise'\n"
            "  local-llm-kit complete -m ./model.gguf -p 'Once upon a time'\n"
            "  local-llm-kit serve --model ./model.gguf --port 8000\n"
            "  local-llm-kit pull owner/repo --output-dir ./models\n"
            "  local-llm-kit list\n"
            "  local-llm-kit embed -m ./model.gguf --text 'hello world'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Show the local-llm-kit version and exit.",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -- chat ------------------------------------------------------------
    chat_p = sub.add_parser("chat", help="Interactive chat with a model.")
    _add_common_llm_args(chat_p, require_model=True)
    _add_common_backend_tuning(chat_p)
    chat_p.add_argument("--system", "-s", default=None, help="System message.")
    chat_p.add_argument(
        "--message", "-q", "--prompt",
        dest="message",
        default=None,
        help="Send a single message non-interactively instead of opening the REPL.",
    )
    chat_p.add_argument("--stream", dest="stream", action="store_true", default=False,
                        help="Stream the response token by token.")
    chat_p.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable streaming.")
    chat_p.add_argument("--json", dest="json", action="store_true", default=False,
                        help="Request JSON responses (legacy; prefer --response-format).")
    chat_p.add_argument(
        "--response-format",
        default=None,
        help="Structured output: 'json', inline JSON, or a path to a JSON "
             "schema / response_format file.",
    )
    chat_p.add_argument("--functions", "-f", default=None,
                        help="Path to JSON file with legacy function definitions.")
    chat_p.add_argument(
        "--tools",
        dest="tools",
        action="append",
        default=None,
        help="Path to a JSON file with OpenAI-style tool definitions. "
             "May be given multiple times.",
    )
    chat_p.add_argument(
        "--function-call", default=None,
        help="Legacy call control: 'auto', 'none', or JSON like '{\"name\": \"fn\"}'.",
    )
    chat_p.add_argument(
        "--tool-choice", default=None,
        help="Modern call control: 'auto', 'none', 'required', or JSON like "
             "'{\"type\": \"function\", \"function\": {\"name\": \"fn\"}}'.",
    )
    chat_p.add_argument(
        "--execute-tools",
        dest="execute_tools",
        action="store_true",
        default=False,
        help="Allow the model to trigger tool calls (single-turn executor). "
             "Tools declared in JSON files carry schemas only, so calls are "
             "displayed; locally registered implementations are executed.",
    )
    chat_p.add_argument(
        "--no-execute-tools", dest="execute_tools", action="store_false",
        help="Never execute tool calls (display them only).",
    )
    chat_p.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    chat_p.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    chat_p.add_argument("--repetition-penalty", type=float, default=None,
                        help="Repetition penalty.")
    chat_p.add_argument("--logprobs", action="store_true", default=False,
                        help="Request log probabilities (if backend supports them).")

    # -- complete --------------------------------------------------------
    comp_p = sub.add_parser("complete", help="Complete a prompt (one-shot).")
    _add_common_llm_args(comp_p, require_model=True)
    _add_common_backend_tuning(comp_p)
    comp_p.add_argument("--prompt", "-p", default=None,
                        help="Prompt to complete (reads from stdin if omitted).")
    comp_p.add_argument("--stream", dest="stream", action="store_true", default=False,
                        help="Stream the response token by token.")
    comp_p.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable streaming.")
    comp_p.add_argument("--json", dest="json", action="store_true", default=False,
                        help="Request JSON output (legacy; prefer --response-format).")
    comp_p.add_argument("--response-format", default=None,
                        help="Structured output: 'json', inline JSON, or a path "
                             "to a JSON schema file.")

    # -- serve -----------------------------------------------------------
    serve_p = sub.add_parser(
        "serve", help="Serve a model over HTTP (OpenAI-compatible API)."
    )
    serve_p.add_argument("--model", "-m", default=None,
                         help="Model path or ID to serve (backend default if omitted).")
    serve_p.add_argument("--backend", "-b", choices=BACKEND_CHOICES, default=None,
                         help="Backend to serve with (default: auto-detect).")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    serve_p.add_argument("--device", default=None, help="Device for transformers.")
    serve_p.add_argument("--gpu-layers", type=int, default=None,
                         help="GPU layers for llama.cpp.")
    serve_p.add_argument("--base-url", default=None, help="Upstream base URL (for proxy backends).")
    serve_p.add_argument("--api-key", default=None, help="Upstream API key (for proxy backends).")
    serve_p.add_argument("--reload", action="store_true", default=False,
                         help="Enable uvicorn auto-reload (development only).")

    # -- pull ------------------------------------------------------------
    pull_p = sub.add_parser("pull", help="Download a model (e.g. from Hugging Face).")
    pull_p.add_argument("model_pos", nargs="?", default=None, metavar="MODEL",
                        help="Model ID or URL to download.")
    pull_p.add_argument("--model", "-m", dest="model_flag", default=None,
                        help="Model ID or URL (alternative to the positional arg).")
    pull_p.add_argument("--output-dir", "-o", default=None,
                        help="Directory to download into (default: helper default).")
    pull_p.add_argument("--revision", default=None, help="Git revision / tag to fetch.")
    pull_p.add_argument("--force", action="store_true", default=False,
                        help="Re-download even if the model is already cached.")

    # -- list ------------------------------------------------------------
    list_p = sub.add_parser(
        "list",
        aliases=["list-models", "models"],
        help="List cached/downloaded models.",
    )
    list_p.add_argument("--json", dest="json", action="store_true", default=False,
                        help="Emit machine-readable JSON instead of a human-readable list.")

    # -- embed -----------------------------------------------------------
    embed_p = sub.add_parser("embed", help="Generate embeddings for text (JSON output).")
    _add_common_llm_args(embed_p, require_model=True)
    _add_common_backend_tuning(embed_p)
    embed_p.add_argument("inputs", nargs="*", metavar="TEXT",
                         help="Text(s) to embed (alternative to --text).")
    embed_p.add_argument("--text", dest="text", action="append", default=None,
                         help="Text to embed. May be given multiple times.")
    embed_p.add_argument("--input-file", default=None,
                         help="Read input texts from a file (one per non-empty line), "
                              "or '-' for stdin.")

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def handle_chat_command(args: argparse.Namespace,
                        backend_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Handle the chat command (interactive REPL or one-shot --message)."""
    if backend_kwargs is None:
        backend_kwargs = _build_backend_kwargs(args)

    functions, tools = _load_tools(
        getattr(args, "functions", None), getattr(args, "tools", None)
    )
    legacy_choice = _parse_choice(getattr(args, "function_call", None), "--function-call")
    modern_choice = _parse_choice(getattr(args, "tool_choice", None), "--tool-choice")
    tool_choice = modern_choice if modern_choice is not None else legacy_choice
    if tool_choice is None and (functions is not None or tools is not None):
        tool_choice = "auto"

    legacy_format, modern_format = _resolve_formats(args)
    execute_tools = bool(getattr(args, "execute_tools", False))

    llm = _create_llm(args, backend_kwargs)

    messages: List[Dict[str, Any]] = []
    if getattr(args, "system", None):
        messages.append({"role": "system", "content": args.system})

    stream = bool(getattr(args, "stream", False))
    temperature = getattr(args, "temperature", 0.7)
    max_tokens = getattr(args, "max_tokens", 512)
    logprobs = bool(getattr(args, "logprobs", False))

    def _run_once() -> Dict[str, Any]:
        result = _chat_once(
            llm,
            messages,
            functions=functions,
            tools=tools,
            tool_choice=tool_choice,
            legacy_format=legacy_format,
            modern_format=modern_format,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            execute_tools=execute_tools,
            logprobs=logprobs,
        )
        if not isinstance(result, dict) or not result.get("choices"):
            _fail("backend returned an unexpected response: {0!r}".format(result))
        return result

    # -- one-shot mode ----------------------------------------------------
    if getattr(args, "message", None):
        messages.append({"role": "user", "content": args.message})
        if stream:
            print("Assistant: ", end="", flush=True)
            acc = ""
            chunks = _chat_once(
                llm, messages, functions=functions, tools=tools,
                tool_choice=tool_choice, legacy_format=legacy_format,
                modern_format=modern_format, temperature=temperature,
                max_tokens=max_tokens, stream=True,
                execute_tools=execute_tools, logprobs=logprobs,
            )
            try:
                for chunk in chunks:
                    piece = _extract_delta_text(chunk)
                    if piece:
                        print(piece, end="", flush=True)
                        acc += piece
            except ImportError as exc:
                _fail("streaming needs no extra deps, but the backend failed: {0}".format(exc))
            print()
            messages.append({"role": "assistant", "content": acc})
            return
        response = _run_once()
        assistant = response["choices"][0]["message"]
        messages.append(assistant)
        if not _print_tool_calls_from_message(assistant):
            print(assistant.get("content"))
        if execute_tools and ("tool_calls" in assistant or "function_call" in assistant):
            _eprint(
                "Note: --execute-tools was given, but tools loaded from JSON "
                "files provide schemas only, so there is nothing local to run. "
                "Register Python implementations via llm.add_tool() to enable "
                "in-process execution."
            )
        return

    # -- interactive REPL ---------------------------------------------------
    print(
        "Chat with {0} (type 'exit' or 'quit' to end, 'clear' to reset history)".format(
            args.model
        )
    )
    try:
        while True:
            try:
                user_input = input("\nYou: ")
            except EOFError:
                print()
                break
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if user_input.strip().lower() == "clear":
                messages = []
                if getattr(args, "system", None):
                    messages.append({"role": "system", "content": args.system})
                print("Chat history cleared.")
                continue
            if not user_input.strip():
                continue
            messages.append({"role": "user", "content": user_input})

            if stream:
                print("\nAssistant: ", end="", flush=True)
                acc_text = ""
                chunks = _chat_once(
                    llm, messages, functions=functions, tools=tools,
                    tool_choice=tool_choice, legacy_format=legacy_format,
                    modern_format=modern_format, temperature=temperature,
                    max_tokens=max_tokens, stream=True,
                    execute_tools=execute_tools, logprobs=logprobs,
                )
                try:
                    for chunk in chunks:
                        piece = _extract_delta_text(chunk)
                        if piece:
                            print(piece, end="", flush=True)
                            acc_text += piece
                except Exception as exc:  # noqa: BLE001
                    print("\nError during streaming: {0}".format(exc))
                    messages.pop()
                    continue
                print()
                messages.append({"role": "assistant", "content": acc_text})
            else:
                try:
                    response = _chat_once(
                        llm, messages, functions=functions, tools=tools,
                        tool_choice=tool_choice, legacy_format=legacy_format,
                        modern_format=modern_format, temperature=temperature,
                        max_tokens=max_tokens, stream=False,
                        execute_tools=execute_tools, logprobs=logprobs,
                    )
                except Exception as exc:  # noqa: BLE001
                    print("\nError: {0}".format(exc))
                    messages.pop()
                    continue
                if "choices" in response and response["choices"]:
                    assistant_msg = response["choices"][0]["message"]
                    messages.append(assistant_msg)
                    if not _print_tool_calls_from_message(assistant_msg):
                        print("\nAssistant: {0}".format(assistant_msg.get("content")))
    except KeyboardInterrupt:
        print("\nExiting chat...")


def handle_completion_command(args: argparse.Namespace,
                              backend_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Handle the completion command."""
    if backend_kwargs is None:
        backend_kwargs = _build_backend_kwargs(args)

    if getattr(args, "prompt", None):
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        print("Enter your prompt (Ctrl+D to finish):")
        try:
            prompt = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nPrompt input cancelled.")
            sys.exit(1)

    if not prompt.strip():
        _fail("empty prompt.")

    legacy_format, modern_format = _resolve_formats(args)
    llm = _create_llm(args, backend_kwargs)
    stream = bool(getattr(args, "stream", False))

    if stream:
        print("Completion: ", end="", flush=True)
        chunks = _complete_once(
            llm, prompt, legacy_format=legacy_format, modern_format=modern_format,
            temperature=getattr(args, "temperature", 0.7),
            max_tokens=getattr(args, "max_tokens", 512), stream=True,
        )
        try:
            for chunk in chunks:
                piece = _extract_chunk_text(chunk)
                if piece:
                    print(piece, end="", flush=True)
        except Exception as exc:  # noqa: BLE001
            _fail("error during streaming: {0}".format(exc))
        print()
    else:
        try:
            response = _complete_once(
                llm, prompt, legacy_format=legacy_format, modern_format=modern_format,
                temperature=getattr(args, "temperature", 0.7),
                max_tokens=getattr(args, "max_tokens", 512), stream=False,
            )
        except Exception as exc:  # noqa: BLE001
            _fail("completion failed: {0}".format(exc))
        if isinstance(response, dict) and response.get("choices"):
            print("Completion: {0}".format(response["choices"][0].get("text", "")))
        else:
            _fail("backend returned an unexpected response: {0!r}".format(response))


def _instantiate_app(create_app: Any, args: argparse.Namespace,
                     backend_kwargs: Dict[str, Any]) -> Any:
    """Call server.create_app compatibly across signature variants."""
    try:
        params = set(inspect.signature(create_app).parameters)
        accepts_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(create_app).parameters.values()
        )
    except (TypeError, ValueError):
        params, accepts_kw = set(), True

    backend = _normalize_backend(getattr(args, "backend", None))
    candidate: Dict[str, Any] = {}
    model = getattr(args, "model", None)
    if "model" in params:
        candidate["model"] = model
    elif "model_path" in params:
        candidate["model_path"] = model
    elif "model_name" in params:
        candidate["model_name"] = model
    if "backend" in params:
        candidate["backend"] = backend
    if "backend_kwargs" in params and backend_kwargs:
        candidate["backend_kwargs"] = backend_kwargs
    if accepts_kw:
        if model and not any(k in candidate for k in ("model", "model_path", "model_name")):
            candidate["model"] = model
        if backend and "backend" not in candidate:
            candidate["backend"] = backend
    try:
        return create_app(**candidate)
    except TypeError:
        if candidate:
            try:
                return create_app()
            except TypeError:
                pass
        raise


def handle_serve_command(args: argparse.Namespace) -> None:
    """Handle `serve`: run the OpenAI-compatible HTTP server."""
    try:
        from .server import create_app
    except ImportError as exc:
        _fail(
            "server support is unavailable ({0}). "
            "Install it with: pip install 'local-llm-kit[server]'".format(exc)
        )
    try:
        import uvicorn  # type: ignore
    except ImportError:
        _fail(
            "'uvicorn' is required to serve models. "
            "Install it with: pip install 'local-llm-kit[server]'"
        )

    backend_kwargs = _build_backend_kwargs(args)
    try:
        app = _instantiate_app(create_app, args, backend_kwargs)
    except Exception as exc:  # noqa: BLE001
        _fail("could not initialise the server app: {0}".format(exc))

    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = int(getattr(args, "port", 8000) or 8000)
    print("Serving model {0!r} on http://{1}:{2}".format(
        getattr(args, "model", None) or "(backend default)", host, port
    ))
    try:
        uvicorn.run(app, host=host, port=port,
                    reload=bool(getattr(args, "reload", False)))
    except KeyboardInterrupt:
        print("\nServer stopped.")


def _adapt_download_call(func: Any, model: str,
                         output_dir: Optional[str],
                         revision: Optional[str],
                         force: bool) -> Any:
    """Invoke a models.py download helper across signature variants."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(model)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs: Dict[str, Any] = {"model": model}
        if output_dir is not None:
            kwargs["output_dir"] = output_dir
        if revision is not None:
            kwargs["revision"] = revision
        if force:
            kwargs["force"] = True
        return func(**kwargs)

    def _pick(keys: Tuple[str, ...]) -> Optional[str]:
        for key in keys:
            if key in params:
                return key
        return None

    model_key = _pick(_MODEL_ARG_KEYS)
    call: Dict[str, Any] = {}
    positional: List[Any] = []
    if model_key is not None:
        call[model_key] = model
    else:
        # Single positional parameter with an unfamiliar name.
        positional_params = [
            p for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_params) == 1 and len(params) == 1:
            return func(model)
        _fail(
            "download helper {0}{1} has an unrecognised signature; "
            "cannot pass the model name.".format(
                getattr(func, "__module__", ""), getattr(func, "__name__", func)
            )
        )
    out_key = _pick(_OUTPUT_DIR_KEYS)
    if out_key is not None and output_dir is not None:
        call[out_key] = output_dir
    rev_key = _pick(_REVISION_KEYS)
    if rev_key is not None and revision is not None:
        call[rev_key] = revision
    force_key = _pick(_FORCE_KEYS)
    if force_key is not None and force:
        call[force_key] = True
    if positional:
        return func(*positional, **call)
    return func(**call)


def handle_pull_command(args: argparse.Namespace) -> None:
    """Handle `pull`: download a model via the models.py helper."""
    model = getattr(args, "model_flag", None) or getattr(args, "model_pos", None)
    if not model:
        _fail("a model is required: pull MODEL or pull --model MODEL.")
    try:
        from . import models as models_mod
    except ImportError as exc:
        _fail(
            "model download helpers are unavailable ({0}). "
            "Upgrade local-llm-kit to a version shipping local_llm_kit.models.".format(exc)
        )
    func = None
    for name in _PULL_FUNC_NAMES:
        candidate = getattr(models_mod, name, None)
        if callable(candidate):
            func = candidate
            break
    if func is None:
        _fail(
            "no download helper found in local_llm_kit.models "
            "(looked for: {0}).".format(", ".join(_PULL_FUNC_NAMES))
        )
    print("Downloading model {0!r}...".format(model))
    try:
        result = _adapt_download_call(
            func, model,
            getattr(args, "output_dir", None),
            getattr(args, "revision", None),
            bool(getattr(args, "force", False)),
        )
    except KeyboardInterrupt:
        _fail("download interrupted.", code=130)
    except Exception as exc:  # noqa: BLE001
        _fail("download failed: {0}".format(exc))
    if result is not None:
        print("{0}".format(result))
    else:
        print("Done.")


def _fallback_cached_models() -> List[str]:
    """Best-effort scan of common cache dirs when models.py has no lister."""
    found: List[str] = []
    candidates: List[str] = []
    home = os.path.expanduser("~")
    hub = os.path.join(home, ".cache", "huggingface", "hub")
    if os.path.isdir(hub):
        try:
            for entry in sorted(os.listdir(hub)):
                if entry.startswith("models--"):
                    found.append(entry.replace("--", "/", 1).replace("--", "_"))
        except OSError:
            pass
    for directory in (
        os.path.join(home, ".cache", "local-llm-kit"),
        os.path.join(os.getcwd(), "models"),
    ):
        candidates.append(directory)
    for directory in candidates:
        if os.path.isdir(directory):
            try:
                for entry in sorted(os.listdir(directory)):
                    if entry.startswith("."):
                        continue
                    found.append(os.path.join(directory, entry))
            except OSError:
                pass
    # De-duplicate, preserving order.
    seen = set()
    unique = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def handle_list_command(args: argparse.Namespace) -> None:
    """Handle `list` / `list-models` / `models`: show cached models."""
    entries: Optional[List[Any]] = None
    try:
        from . import models as models_mod
    except ImportError:
        models_mod = None  # type: ignore
    if models_mod is not None:
        for name in _LIST_FUNC_NAMES:
            candidate = getattr(models_mod, name, None)
            if callable(candidate):
                try:
                    entries = list(candidate())
                except TypeError:
                    try:
                        entries = list(candidate(None))  # type: ignore
                    except Exception as exc:  # noqa: BLE001
                        _fail("could not list models: {0}".format(exc))
                except Exception as exc:  # noqa: BLE001
                    _fail("could not list models: {0}".format(exc))
                break
    if entries is None:
        entries = _fallback_cached_models()

    names: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            for key in ("name", "model", "model_id", "id", "path"):
                if entry.get(key):
                    names.append(str(entry[key]))
                    break
            else:
                names.append(json.dumps(entry))
        else:
            names.append(str(entry))

    if bool(getattr(args, "json", False)):
        print(json.dumps(names, indent=2))
        return
    if not names:
        print("No cached models found.")
        return
    print("Cached models:")
    for name in names:
        print("  - {0}".format(name))


def _normalize_vectors(result: Any, expected: int) -> Optional[List[List[float]]]:
    """Normalize assorted embedding return shapes to a list of vectors."""
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, list) and data:
            vecs = []
            for item in data:
                if isinstance(item, dict) and "embedding" in item:
                    vecs.append(list(item["embedding"]))
                elif isinstance(item, (list, tuple)):
                    vecs.append(list(item))
            if vecs:
                return vecs
        for key in ("embeddings", "vectors", "embedding"):
            if isinstance(result.get(key), list):
                payload = result[key]
                if payload and all(isinstance(v, (int, float)) for v in payload):
                    return [list(payload)] if expected == 1 else None
                if payload and isinstance(payload[0], (list, tuple)):
                    return [list(v) for v in payload]
    if isinstance(result, (list, tuple)) and result:
        if all(isinstance(v, (int, float)) for v in result):
            return [list(result)]  # type: ignore
        if isinstance(result[0], (list, tuple)):
            return [list(v) for v in result]  # type: ignore
        if isinstance(result[0], dict) and "embedding" in result[0]:
            return [list(v["embedding"]) for v in result]  # type: ignore
    return None


def _embed_texts(llm: Any, texts: List[str]) -> List[List[float]]:
    """Obtain embeddings via the LLM, backend, or embeddings module."""
    attempts: List[str] = []

    def _try_call(obj: Any, label: str) -> Optional[List[List[float]]]:
        for name in _EMBED_METHOD_NAMES:
            fn = getattr(obj, name, None)
            if not callable(fn):
                continue
            for kwargs in (
                {"texts": texts},
                {"inputs": texts},
                {"input": texts},
                {"texts": texts[0] if len(texts) == 1 else texts},
                {"text": texts[0] if len(texts) == 1 else texts},
            ):
                try:
                    vecs = _normalize_vectors(fn(**kwargs), len(texts))
                except TypeError:
                    continue
                except Exception as exc:  # noqa: BLE001
                    attempts.append("{0}.{1}: {2}".format(label, name, exc))
                    break
                if vecs is not None:
                    return vecs
                attempts.append("{0}.{1}: unrecognised return shape".format(label, name))
                break
        return None

    hit = _try_call(llm, "LLM")
    if hit is not None:
        return hit
    backend = getattr(llm, "backend", None)
    if backend is not None:
        hit = _try_call(backend, "backend")
        if hit is not None:
            return hit
    try:
        import importlib

        try:
            emb_mod = importlib.import_module(".embeddings", __package__ or "local_llm_kit")
        except ImportError:
            emb_mod = None
        if emb_mod is not None:
            for name in ("embed_texts", "get_embeddings", "embed", "embeddings"):
                fn = getattr(emb_mod, name, None)
                if not callable(fn):
                    continue
                try:
                    try:
                        vecs = _normalize_vectors(fn(texts), len(texts))
                    except TypeError:
                        vecs = _normalize_vectors(fn(texts, llm), len(texts))
                except Exception as exc:  # noqa: BLE001
                    attempts.append("embeddings.{0}: {1}".format(name, exc))
                    continue
                if vecs is not None:
                    return vecs
    except Exception:  # noqa: BLE001
        pass
    detail = "; ".join(attempts) if attempts else "no embedding method found"
    _fail(
        "this backend/model does not expose embeddings ({0}).".format(detail)
    )
    return []  # unreachable


def handle_embed_command(args: argparse.Namespace) -> None:
    """Handle `embed`: print embeddings for the given texts as JSON."""
    texts: List[str] = []
    for piece in (getattr(args, "text", None) or []) + (getattr(args, "inputs", None) or []):
        if piece is not None and str(piece) != "":
            texts.append(str(piece))

    input_file = getattr(args, "input_file", None)
    if input_file:
        if input_file == "-":
            raw = sys.stdin.read()
        else:
            try:
                with open(input_file, "r", encoding="utf-8") as fh:
                    raw = fh.read()
            except OSError as exc:
                _fail("could not read --input-file: {0}".format(exc))
        texts.extend([ln for ln in (ln.strip() for ln in raw.splitlines()) if ln])

    if not texts and not sys.stdin.isatty() and not input_file:
        raw = sys.stdin.read()
        texts.extend([ln for ln in (ln.strip() for ln in raw.splitlines()) if ln])

    if not texts:
        _fail("no input text: pass TEXT, --text, --input-file, or pipe stdin.")

    llm = _create_llm(args, _build_backend_kwargs(args))
    try:
        vectors = _embed_texts(llm, texts)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail("embedding failed: {0}".format(exc))

    payload = {
        "object": "list",
        "model": getattr(args, "model", None),
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
    }
    print(json.dumps(payload))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print("local-llm-kit {0}".format(_get_version()))
        return 0

    command = getattr(args, "command", None)
    if not command:
        parser.print_help()
        return 1

    try:
        if command == "chat":
            handle_chat_command(args, _build_backend_kwargs(args))
        elif command == "complete":
            handle_completion_command(args, _build_backend_kwargs(args))
        elif command == "serve":
            handle_serve_command(args)
        elif command == "pull":
            handle_pull_command(args)
        elif command in ("list", "list-models", "models"):
            handle_list_command(args)
        elif command == "embed":
            handle_embed_command(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
