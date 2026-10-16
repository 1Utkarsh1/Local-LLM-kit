"""
Resilient structured JSON generation and zero-dependency JSON auto-repair.

Local LLMs frequently emit 'almost valid' JSON:
- Markdown fences (`````json ... `````)
- Conversational prefaces and postscripts ('Here is the JSON you requested:')
- Unquoted property keys (``{name: "Alice"}``)
- Single-quoted strings (``{'key': 'value'}``)
- Python literals (``True``, ``False``, ``None``) instead of JSON standards
- Trailing commas (``{"a": 1, "b": 2,}``)
- Truncated braces/brackets at context window boundaries

This module provides:
1. :func:`repair_json` -- pure standard-library parser/regex pipeline repairing
   common local model syntax corruptions without external dependencies.
2. :func:`extract_json` -- robust extraction returning parsed Python dicts/lists.
3. :func:`chat_structured` -- multi-turn self-correcting structured chat loop
   that combines native schema enforcement with error-feedback retries.

Python 3.9+ compatible.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

logger = logging.getLogger(__name__)

# Regular expressions for repair heuristics
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
_PYTHON_LITERALS = [
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
]
_UNQUOTED_KEY_RE = re.compile(r"([,{]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:")
_SINGLE_QUOTED_STRING_RE = re.compile(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'")


def repair_json(text: str) -> str:
    """Repair common syntax defects in local LLM generated JSON strings.

    Args:
        text: Raw text string from an LLM response.

    Returns:
        Cleaned and normalized JSON string ready for ``json.loads()``.
    """
    if not isinstance(text, str):
        raise TypeError(f"repair_json expects string input, got {type(text).__name__}")

    s = text.strip()
    if not s:
        return "{}"

    # 1. Strip markdown code fences if present
    fence_match = _CODE_FENCE_RE.search(s)
    if fence_match:
        s = fence_match.group(1).strip()

    # 2. Extract JSON substring if surrounded by conversational text
    if not (s.startswith("{") or s.startswith("[")):
        start_brace = s.find("{")
        start_bracket = s.find("[")
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            s = s[start_brace:]
        elif start_bracket != -1:
            s = s[start_bracket:]

    # 3. Trim trailing conversational text only after root object/array is fully closed
    if s.startswith("{") or s.startswith("["):
        depth = 0
        in_str = False
        esc = False
        root_end = -1
        for idx, ch in enumerate(s):
            if ch == '"' and not esc:
                in_str = not in_str
            elif not in_str:
                if ch in ("{", "["):
                    depth += 1
                elif ch in ("}", "]"):
                    depth -= 1
                    if depth == 0:
                        root_end = idx
                        break
            if ch == "\\" and not esc:
                esc = True
            else:
                esc = False
        if root_end != -1 and root_end < len(s) - 1:
            s = s[: root_end + 1]

    # 4. Replace Python literals: True -> true, False -> false, None -> null
    for pattern, replacement in _PYTHON_LITERALS:
        s = pattern.sub(replacement, s)

    # 5. Remove trailing commas before closing braces/brackets
    s = _TRAILING_COMMA_RE.sub(r"\1", s)

    # 6. Convert single-quoted strings to double-quoted JSON strings
    # Check if string contains single quotes but not double quotes
    if "'" in s and '"' not in s:
        s = _SINGLE_QUOTED_STRING_RE.sub(r'"\1"', s)
    elif "'" in s:
        # Replace single-quoted dictionary keys: 'key': -> "key":
        s = re.sub(r"([,{]\s*)'([^'\\]+)'\s*:", r'\1"\2":', s)

    # 7. Quote unquoted alphanumeric object keys: {key: -> {"key":
    s = _UNQUOTED_KEY_RE.sub(r'\1"\2":', s)

    # 8. Balance unclosed brackets or braces in exact LIFO order
    s = s.rstrip(", \t\n")
    if s.count('"') % 2 != 0:
        s += '"'

    stack: List[str] = []
    in_str = False
    esc = False
    for ch in s:
        if ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch == "}" and stack and stack[-1] == "}":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "]":
                stack.pop()
        if ch == "\\" and not esc:
            esc = True
        else:
            esc = False

    if stack:
        s += "".join(reversed(stack))

    return s


def extract_json(text: str) -> Any:
    """Extract and parse a JSON object or array from raw LLM output.

    Attempts standard ``json.loads()`` first. If decoding fails, applies
    :func:`repair_json` heuristics before retrying.

    Args:
        text: Raw response string from the model.

    Returns:
        Parsed Python dictionary, list, or primitive.

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after repair attempts.
    """
    if not isinstance(text, str):
        raise TypeError(f"extract_json expects string, got {type(text).__name__}")

    # Direct fast-path
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Heuristic repair path
    repaired = repair_json(text)
    return json.loads(repaired)


def validate_schema(data: Any, schema: Union[Dict[str, Any], Type, Callable]) -> None:
    """Validate parsed JSON data against a JSON schema or Pydantic/dataclass model.

    Raises:
        ValueError: If validation fails with description of mismatch.
    """
    if schema is None:
        return

    # 1. Pydantic model (v1 or v2)
    if hasattr(schema, "model_validate"):  # Pydantic v2
        try:
            schema.model_validate(data)
            return
        except Exception as exc:
            raise ValueError(f"Schema validation failed: {exc}") from exc
    elif hasattr(schema, "parse_obj"):  # Pydantic v1
        try:
            schema.parse_obj(data)
            return
        except Exception as exc:
            raise ValueError(f"Schema validation failed: {exc}") from exc

    # 2. JSON Schema dictionary (type: object, required, properties)
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object (dict), got {type(data).__name__}")
            required = schema.get("required", [])
            for req in required:
                if req not in data:
                    raise ValueError(f"Missing required field: {req!r}")
        elif schema.get("type") == "array":
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array (list), got {type(data).__name__}")


def chat_structured(
    llm: Any,
    messages: Sequence[Dict[str, Any]],
    schema: Optional[Union[Dict[str, Any], Type, Any]] = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute a chat completion requiring structured JSON output with self-correction.

    Combines native backend grammar/schema enforcement with automatic JSON repair
    and multi-turn validation retries.

    Args:
        llm: An :class:`local_llm_kit.LLM` instance.
        messages: Conversation messages.
        schema: Optional JSON schema dict or Pydantic model for validation.
        max_retries: Maximum number of correction turns if the model produces invalid JSON.
        **kwargs: Additional options forwarded to ``llm.chat()``.

    Returns:
        Parsed and validated JSON data (dict or list).

    Raises:
        ValueError: If the model fails to produce valid structured JSON after retries.
    """
    active_messages = list(messages)

    # Configure response_format
    req_kwargs = dict(kwargs)
    if "response_format" not in req_kwargs:
        if isinstance(schema, dict) and "properties" in schema:
            req_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output_schema", "schema": schema},
            }
        else:
            req_kwargs["response_format"] = {"type": "json_object"}

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        resp = llm.chat(messages=active_messages, **req_kwargs)
        choices = resp.get("choices", [])
        if not choices:
            raise ValueError("LLM returned empty choices in chat_structured.")
        raw_text = choices[0].get("message", {}).get("content", "")

        try:
            parsed = extract_json(raw_text)
            if schema:
                validate_schema(parsed, schema)
            return parsed
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Structured output attempt %d failed: %s", attempt + 1, last_error)
            if attempt < max_retries:
                # Append assistant output and user correction request
                active_messages.append({"role": "assistant", "content": raw_text})
                active_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response produced an error: {last_error}.\n"
                            "Please output ONLY valid, well-formed JSON matching the required structure."
                        ),
                    }
                )

    raise ValueError(
        f"Failed to obtain valid structured JSON after {max_retries + 1} attempts. Last error: {last_error}"
    )
