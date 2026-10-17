"""
local_llm_kit - OpenAI-compatible interface for local LLMs.

Backends: transformers, llama.cpp (GGUF), Ollama, generic OpenAI-compatible
servers (vLLM, LM Studio, llama-server), and an offline ``echo`` mock.

Core install stays light (stdlib + typing-extensions); heavy dependencies
are optional extras, lazily imported.
"""

__version__ = "0.2.0"

from .chat import achat, acomplete, chat, complete, embed
from .function_calling import FunctionCall, FunctionRegistry, add_function, parse_function_calls
from .llm import LLM, clear_backend_cache, clear_client_cache, get_client
from .tools import (
    ToolCall,
    ToolRegistry,
    function_to_json_schema,
    normalize_tools,
    parse_tool_calls,
    parse_tool_calls_safe,
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
    "parse_tool_calls_safe",
    # Phase 2: Conversation logging & prompt templates (lazy / stdlib-only)
    "ConversationLog",
    "PromptTemplate",
    "TemplateRegistry",
]


def __getattr__(name: str):
    """Lazy exports for optional features (sqlite3-based conversation log)."""
    if name == "ConversationLog":
        from .conversation_log import ConversationLog

        return ConversationLog
    if name == "PromptTemplate":
        from .prompt_templates import PromptTemplate

        return PromptTemplate
    if name == "TemplateRegistry":
        from .prompt_templates import TemplateRegistry

        return TemplateRegistry
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__() -> list:
    return list(globals().keys()) + ["ConversationLog", "PromptTemplate", "TemplateRegistry"]
