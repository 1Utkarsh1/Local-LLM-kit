"""
local_llm_kit - OpenAI-compatible interface for local LLMs.

Backends: transformers, llama.cpp (GGUF), Ollama, generic OpenAI-compatible
servers (vLLM, LM Studio, llama-server), and an offline ``echo`` mock.

Core install stays light (stdlib + typing-extensions); heavy dependencies
are optional extras, lazily imported.
"""

__version__ = "0.2.0"

from .agent import Agent, AgentResult
from .chat import achat, acomplete, chat, complete, embed
from .function_calling import FunctionCall, FunctionRegistry, add_function, parse_function_calls
from .hardware import estimate_model_vram, get_system_memory, recommend_quantization
from .llm import LLM, clear_backend_cache, clear_client_cache, get_client
from .mcp import MCPClient, MCPError
from .rag import (
    RAGPipeline,
    SimpleVectorStore,
    bm25_scores,
    build_rag_prompt,
    chunk_text,
    reciprocal_rank_fusion,
    retrieve,
)
from .structured import chat_structured, extract_json, repair_json
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
    # Agent & MCP
    "Agent",
    "AgentResult",
    "MCPClient",
    "MCPError",
    # Structured outputs & repair
    "repair_json",
    "extract_json",
    "chat_structured",
    # Hardware & VRAM
    "estimate_model_vram",
    "recommend_quantization",
    "get_system_memory",
    # RAG & Hybrid search
    "RAGPipeline",
    "SimpleVectorStore",
    "chunk_text",
    "retrieve",
    "build_rag_prompt",
    "bm25_scores",
    "reciprocal_rank_fusion",
]
