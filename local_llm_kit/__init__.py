"""
local_llm_kit - OpenAI-compatible interface for local LLMs.

Backends: transformers, llama.cpp (GGUF), Ollama, Apple Foundation Models
(on-device system model), generic OpenAI-compatible
servers (vLLM, LM Studio, llama-server, apfel), and an offline ``echo`` mock.

Core install stays light (stdlib + typing-extensions); heavy dependencies
are optional extras, lazily imported.
"""

__version__ = "0.2.0"

from .chat import achat, acomplete, chat, complete, embed
from .discovery import (
    DiscoveryReport,
    ModelRecord,
    ServerReport,
    discover,
    find_local_server,
    probe_endpoint,
)
from .function_calling import FunctionCall, FunctionRegistry, add_function, parse_function_calls
from .llm import LLM, clear_backend_cache, clear_client_cache, get_client
from .mcp import (
    MCPError,
    MCPServer,
    MCPTool,
    MCPToolError,
    mcp_tools_to_openai,
)
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
    "MCPServer",
    "MCPTool",
    "MCPError",
    "MCPToolError",
    "mcp_tools_to_openai",
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
    "discover",
    "probe_endpoint",
    "find_local_server",
    "DiscoveryReport",
    "ServerReport",
    "ModelRecord",
]


_LAZY_EXPORTS = {
    "Agent": "agent",
    "AgentResult": "agent",
    "ConversationLog": "conversation_log",
    "get_conversation_log": "conversation_log",
    "PromptTemplate": "prompt_templates",
    "TemplateRegistry": "prompt_templates",
    "register_template": "prompt_templates",
    "get_template": "prompt_templates",
    "render_template": "prompt_templates",
    "chat_structured": "structured",
    "extract_json": "structured",
    "repair_json": "structured",
    "validate_schema": "structured",
    "get_system_memory": "hardware",
    "estimate_model_vram": "hardware",
    "recommend_quantization": "hardware",
    "RAGPipeline": "rag",
    "SimpleVectorStore": "rag",
}
__all__ += list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib import import_module

    value = getattr(import_module("." + _LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
