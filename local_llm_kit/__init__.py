"""
local_llm_kit - OpenAI-compatible interface for local LLMs.

Backends: transformers, llama.cpp (GGUF), Ollama, generic OpenAI-compatible
servers (vLLM, LM Studio, llama-server), and an offline ``echo`` mock.

Core install stays light (stdlib + typing-extensions); heavy dependencies
are optional extras, lazily imported.
"""

__version__ = "0.2.0"

from .llm import LLM, clear_backend_cache, clear_client_cache, get_client
from .chat import achat, acomplete, chat, complete, embed
from .function_calling import FunctionCall, FunctionRegistry, add_function, parse_function_calls
from .tools import (
    ToolCall,
    ToolRegistry,
    function_to_json_schema,
    normalize_tools,
    parse_tool_calls,
    tool,
)

__all__ = [
    "__version__",
    "LLM",
    "get_client",
    "clear_backend_cache",
    "clear_client_cache",
    "chat",
    "complete",
    "embed",
    "achat",
    "acomplete",
    "add_function",
    "tool",
    "ToolCall",
    "ToolRegistry",
    "FunctionCall",
    "FunctionRegistry",
    "function_to_json_schema",
    "normalize_tools",
    "parse_function_calls",
    "parse_tool_calls",
]
