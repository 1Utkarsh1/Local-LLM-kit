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

import ast
import json
import logging
import re
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

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


def repair_json(text: str) -> str:
    """Repair JSON presentation errors without rewriting quoted string values.

    This opt-in output helper is never used to complete executable tool arguments.
    It can close missing containers, but never invent an unfinished string/value.
    """
    if not isinstance(text, str):
        raise TypeError("repair_json expects a string.")
    if len(text) > 1024 * 1024:
        raise ValueError("JSON output exceeds the 1 MiB repair limit.")
    s = text.strip()
    if not s:
        raise ValueError("JSON output is empty.")
    try:
        json.loads(s)
        return s
    except ValueError:
        pass
    fence = _CODE_FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        raise ValueError("No JSON object or array was found.")
    s = s[min(starts) :]
    # Find the end of the first container, respecting both JSON and Python strings.
    stack = []
    quote = None
    escaped = False
    for index, char in enumerate(s):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack.pop():
                raise ValueError("Mismatched JSON containers.")
            if not stack:
                s = s[: index + 1]
                break
    if quote:
        raise ValueError("Unterminated JSON string; regenerate the output.")
    s = s.rstrip() + "".join(reversed(stack))
    try:
        return json.dumps(ast.literal_eval(s), allow_nan=False)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        pass
    strings = []

    def hold(match):
        token = match.group()
        value = json.loads(token) if token.startswith('"') else ast.literal_eval(token)
        strings.append(json.dumps(value))
        return '"@llmk[%d]@"' % (len(strings) - 1)

    masked = re.sub(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'", hold, s)
    for pattern, replacement in _PYTHON_LITERALS:
        masked = pattern.sub(replacement, masked)
    masked = _TRAILING_COMMA_RE.sub(r"\1", masked)
    masked = _UNQUOTED_KEY_RE.sub(r'\1"\2":', masked)
    repaired = re.sub(r'"@llmk\[(\d+)\]@"', lambda match: strings[int(match.group(1))], masked)
    json.loads(repaired)
    return repaired


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
    """Validate all JSON Schema constraints, or a Pydantic v1/v2 model.

    Dictionary schemas require the optional ``structured`` extra. Unsupported
    validators are rejected rather than reported as successful validation.
    """
    if schema is None:
        return
    if isinstance(schema, dict):
        try:
            from jsonschema import validators
        except ImportError as exc:
            raise ImportError(
                'JSON Schema validation needs pip install "local-llm-kit[structured]".'
            ) from exc
        validator_class = validators.validator_for(schema)
        validator_class.check_schema(schema)
        from referencing import Registry

        # No automatic remote schema retrieval in an offline helper.
        error = next(validator_class(schema, registry=Registry()).iter_errors(data), None)
        if error:
            raise ValueError("Schema validation failed: %s" % error.message)
        return
    validate = getattr(schema, "model_validate", None) or getattr(schema, "parse_obj", None)
    if not callable(validate):
        raise TypeError("schema must be a JSON Schema dictionary or a Pydantic model class.")
    try:
        validate(data)
    except Exception as exc:
        raise ValueError("Schema validation failed: %s" % exc) from exc


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
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    if kwargs.get("stream"):
        raise ValueError("chat_structured requires stream=False.")
    if not isinstance(messages, (list, tuple)) or any(not isinstance(m, dict) for m in messages):
        raise TypeError("messages must be a sequence of message dictionaries.")
    if schema is not None:
        # Check validator availability/schema validity before running inference.
        if isinstance(schema, dict):
            try:
                from jsonschema import validators
            except ImportError as exc:
                raise ImportError(
                    'Install "local-llm-kit[structured]" for JSON Schema validation.'
                ) from exc
            validators.validator_for(schema).check_schema(schema)
        elif not callable(
            getattr(schema, "model_validate", None) or getattr(schema, "parse_obj", None)
        ):
            raise TypeError("Unsupported schema validator.")
    active_messages = [dict(m) for m in messages]

    # Configure response_format
    req_kwargs = dict(kwargs)
    req_kwargs.setdefault("execute_tools", False)
    if "response_format" not in req_kwargs:
        if isinstance(schema, dict):
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
        except (ValueError, TypeError) as exc:
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
