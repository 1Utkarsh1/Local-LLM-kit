"""Shared types for local_llm_kit.

All message/tool/response shapes follow the OpenAI API so that code
written against OpenAI (or LangChain / LlamaIndex / Continue / Open WebUI)
works against local models with minimal changes.
"""
from typing import Any, Dict, List, Optional, Union

# A chat message. ``content`` may be:
#   - a plain string, or
#   - a list of content parts for vision models, e.g.
#     [{"type": "text", "text": "What is this?"},
#      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
ChatMessage = Dict[str, Any]

# OpenAI-style tool definition:
# {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
ToolSpec = Dict[str, Any]

# Legacy function definition (still supported):
# {"name": ..., "description": ..., "parameters": {...}}
FunctionSpec = Dict[str, Any]

# Structured-output request:
# {"type": "json_object"} or
# {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": ...}}
ResponseFormat = Dict[str, Any]

__all__ = [
    "ChatMessage",
    "ToolSpec",
    "FunctionSpec",
    "ResponseFormat",
]
