"""Modern tool calling for local LLMs.

Supports the OpenAI ``tools`` / ``tool_choice`` format (including parallel
calls and the ``tool`` role) while staying backwards compatible with the
legacy ``functions`` / ``function_call`` format.

Example:
    from local_llm_kit import LLM, tool

    @tool(description="Get the current weather for a city")
    def get_weather(location: str, unit: str = "celsius") -> dict:
        '''Get the weather for a location.'''
        return {"temperature": 22, "unit": unit, "location": location}

    llm = LLM(model_path="...")   # or backend="ollama", backend="echo", ...
    llm.add_tool(get_weather)     # or pass tools=[...] per-call

    resp = llm.chat(
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[get_weather.spec],
        tool_choice="auto",
    )
"""

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from .function_calling import FunctionCall, FunctionRegistry, parse_function_calls

logger = logging.getLogger(__name__)

_PYTHON_TO_JSON = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def _json_type(annotation: Any) -> str:
    """Map a Python annotation to a JSON Schema type string."""
    if annotation is None or annotation is inspect.Parameter.empty:
        return "string"
    name = getattr(annotation, "__name__", str(annotation))
    # Handle Optional[X] / Union[X, None] -> X
    origin = getattr(annotation, "__origin__", None)
    if origin is Union:
        args = [a for a in getattr(annotation, "__args__", []) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return _json_type(args[0])
        return "string"
    if name in _PYTHON_TO_JSON:
        return _PYTHON_TO_JSON[name]
    # Pydantic model -> object (full schema handled by caller if available)
    if hasattr(annotation, "model_json_schema"):
        return "object"
    if hasattr(annotation, "schema"):
        return "object"
    return "string"


def function_to_json_schema(func: Callable) -> Dict[str, Any]:
    """Build a JSON Schema ``parameters`` object from a Python signature.

    Uses type hints, defaults (to infer ``required``), and docstrings.
    No third-party dependencies required. If a pydantic model is used as
    an annotation and pydantic is installed, its schema is embedded.
    """
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = param.annotation
        prop: Dict[str, Any] = {"type": _json_type(annotation)}
        # Embed pydantic model schema when possible
        try:
            if hasattr(annotation, "model_json_schema"):
                prop = annotation.model_json_schema()  # type: ignore
            elif hasattr(annotation, "schema") and inspect.isclass(annotation):
                maybe = annotation.schema
                if callable(maybe):
                    prop = maybe()
        except Exception:
            pass
        if param.default is not inspect.Parameter.empty:
            try:
                json.dumps(param.default)  # ensure serializable
                prop["default"] = param.default
            except TypeError:
                pass
        else:
            required.append(name)
        properties[name] = prop
    return {"type": "object", "properties": properties, "required": required}


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """Decorator turning a Python function into an LLM tool.

    The decorated function gains a ``.spec`` attribute containing the
    OpenAI-style tool definition, so it can be passed directly to
    ``LLM.chat(tools=[...])`` or ``llm.add_tool(fn)``.

    Can be used bare (``@tool``) or with arguments
    (``@tool(description="...")``).
    """

    def _wrap(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        doc = (description or (fn.__doc__ or "")).strip()
        schema = function_to_json_schema(fn)
        fn.spec = {  # type: ignore[attr-defined]
            "type": "function",
            "function": {
                "name": tool_name,
                "description": doc,
                "parameters": schema,
            },
        }
        fn.tool_name = tool_name  # type: ignore[attr-defined]
        return fn

    if func is not None:
        return _wrap(func)
    return _wrap


class ToolCall:
    """A single parsed tool call (OpenAI ``tool_calls`` item)."""

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
    ):
        self.name = name
        self.arguments = arguments
        self.id = (
            call_id
            or f"call_{abs(hash((name, json.dumps(arguments, sort_keys=True)))) % 10**8:08d}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        fn = data.get("function", data)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        return cls(name=fn["name"], arguments=args, call_id=data.get("id"))

    def to_function_call(self) -> FunctionCall:
        return FunctionCall(name=self.name, arguments=self.arguments)


class ToolRegistry(FunctionRegistry):
    """Registry supporting both ``tools`` and legacy ``functions``."""

    def add_tool(self, func: Callable, name: Optional[str] = None) -> Dict[str, Any]:
        """Register a Python callable as a tool. Returns its OpenAI tool spec."""
        spec = getattr(func, "spec", None)
        if spec is None:
            # Build spec on the fly
            tool_name = name or getattr(func, "tool_name", None) or func.__name__
            doc = (func.__doc__ or "").strip()
            spec = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": doc,
                    "parameters": function_to_json_schema(func),
                },
            }
        else:
            tool_name = name or spec["function"]["name"]
        schema = dict(spec["function"])
        schema["name"] = tool_name
        self.add_function(name=tool_name, schema=schema, implementation=func)
        return {"type": "function", "function": schema}

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Return registered functions as OpenAI-style ``tools`` list."""
        return [{"type": "function", "function": dict(schema)} for schema in self.get_schema_list()]


def normalize_tools(
    tools: Optional[List[Any]] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Merge ``tools`` (OpenAI format, callables, or @tool fns) and legacy
    ``functions`` into a single OpenAI-style ``tools`` list."""
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        if callable(t) and not isinstance(t, dict):
            spec = getattr(t, "spec", None)
            if spec is not None:
                out.append(spec)
            else:
                tname = getattr(t, "__name__", "tool")
                out.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tname,
                            "description": (t.__doc__ or "").strip(),
                            "parameters": function_to_json_schema(t),
                        },
                    }
                )
        elif isinstance(t, dict):
            if t.get("type") == "function" and "function" in t:
                out.append(t)
            elif "name" in t:
                # Bare function spec -> wrap
                out.append({"type": "function", "function": t})
            else:
                out.append(t)
    for f in functions or []:
        if isinstance(f, dict) and f.get("type") == "function":
            out.append(f)
        elif isinstance(f, dict):
            out.append({"type": "function", "function": f})
    # Deduplicate by function name (last wins)
    seen: Dict[str, Dict[str, Any]] = {}
    for t in out:
        try:
            seen[t["function"]["name"]] = t
        except (KeyError, TypeError):
            continue
    return list(seen.values())


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Parse one or more tool calls from model output text.

    Handles, in order:
      1. OpenAI-style JSON: ``{"tool_calls": [{"function": {...}}]}``
      2. ``<tool_call>{...}</tool_call>`` tags (Llama 3 / Qwen style)
      3. ``[TOOL_CALLS] [...]`` blocks
      4. Legacy single function-call formats (via ``parse_function_calls``)
    Raises ``ValueError`` when nothing is found.

    Inputs longer than ``max_chars`` are truncated before the regex
    strategies run (ReDoS mitigation for unbounded model output).
    """
    import re

    max_chars = 20000
    if len(text) > max_chars:
        text = text[:max_chars]

    # 1. Direct JSON with tool_calls
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool_calls" in data:
            return [ToolCall.from_dict(tc) for tc in data["tool_calls"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # 2/3. Tagged blocks — may contain a JSON array or single object
    for pattern in (
        r"<tool_call>(.*?)</tool_call>",
        r"\[TOOL_CALLS\](.*?)\[/TOOL_CALLS\]",
        r"<tools>(.*?)</tools>",
    ):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            payload = m.group(1).strip()
            try:
                data = json.loads(payload)
                items = data if isinstance(data, list) else [data]
                calls = []
                for item in items:
                    if "function" in item:
                        calls.append(ToolCall.from_dict(item))
                    elif "name" in item:
                        args = item.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args) if args.strip() else {}
                        calls.append(
                            ToolCall(name=item["name"], arguments=args, call_id=item.get("id"))
                        )
                if calls:
                    return calls
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    # 4. Legacy formats (also covers bare {"name","arguments"} JSON)
    try:
        fns = parse_function_calls(text)
        return [ToolCall(name=f.name, arguments=f.arguments) for f in fns]
    except ValueError:
        pass

    raise ValueError("No tool call could be parsed from the output")


def should_call_tools(
    tool_choice: Union[str, Dict[str, Any], None],
    function_call: Union[str, Dict[str, str]],
) -> bool:
    """Return True when the model is allowed to emit tool/function calls."""
    if tool_choice is not None:
        if tool_choice == "none":
            return False
        return True
    return function_call != "none"


__all__ = [
    "ToolCall",
    "ToolRegistry",
    "tool",
    "function_to_json_schema",
    "normalize_tools",
    "parse_tool_calls",
    "should_call_tools",
    # Re-exported legacy names so `from .tools import ...` covers both APIs
    "FunctionCall",
    "FunctionRegistry",
    "parse_function_calls",
]
