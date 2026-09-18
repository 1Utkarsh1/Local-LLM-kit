"""
Function calling implementation for local LLMs.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class FunctionCall:
    """Represents a function call parsed from model output."""

    def __init__(self, name: str, arguments: Dict[str, Any]):
        """
        Initialize a function call.

        Args:
            name: Name of the function
            arguments: Arguments to pass to the function
        """
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary compatible with OpenAI's function call format.

        Returns:
            Dict with name and arguments
        """
        return {"name": self.name, "arguments": json.dumps(self.arguments)}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "FunctionCall":
        """
        Create a FunctionCall from an OpenAI-style function call dict.

        Args:
            data: Dictionary with "name" and "arguments" keys

        Returns:
            FunctionCall instance
        """
        return cls(name=data["name"], arguments=json.loads(data["arguments"]))


class FunctionRegistry:
    """Registry for functions that can be called by the LLM."""

    def __init__(self):
        """Initialize an empty function registry."""
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.implementations: Dict[str, Callable] = {}

    def add_function(self, name: str, schema: Dict[str, Any], implementation: Callable) -> None:
        """
        Register a function that can be called by the model.

        Args:
            name: Name of the function
            schema: JSON Schema describing the function parameters
            implementation: Python function to execute when called
        """
        if "name" not in schema:
            schema["name"] = name

        self.functions[name] = schema
        self.implementations[name] = implementation

    def get_schema_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of function schemas suitable for inclusion in prompts.

        Returns:
            List of function schemas
        """
        return list(self.functions.values())

    def has_functions(self) -> bool:
        """
        Check if any functions are registered.

        Returns:
            True if functions are registered, False otherwise
        """
        return len(self.functions) > 0

    def has_function(self, name: str) -> bool:
        """
        Check if a specific function is registered.

        Args:
            name: Name of the function

        Returns:
            True if the function is registered, False otherwise
        """
        return name in self.functions

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a registered function with the given arguments.

        Args:
            name: Name of the function to execute
            arguments: Arguments to pass to the function

        Returns:
            Result of the function execution

        Raises:
            ValueError: If the function is not registered
        """
        if not self.has_function(name):
            raise ValueError(f"Function '{name}' is not registered")

        implementation = self.implementations[name]
        return implementation(**arguments)


def parse_function_calls(text: str) -> List[FunctionCall]:
    """
    Parse function calls from model output text.

    The function tries several strategies:
    1. Parse JSON function call with {"name": "...", "arguments": {...}}
    2. Look for function_call({...}) or similar patterns
    3. Fallback to regex pattern matching

    Args:
        text: Model output text

    Returns:
        List of FunctionCall objects

    Raises:
        ValueError: If no function call could be parsed
    """
    # Strategy 1: Try to parse as JSON directly
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            if isinstance(data["arguments"], str):
                args = json.loads(data["arguments"])
            else:
                args = data["arguments"]
            return [FunctionCall(data["name"], args)]
        elif isinstance(data, dict) and "function_call" in data:
            func_call = data["function_call"]
            if isinstance(func_call["arguments"], str):
                args = json.loads(func_call["arguments"])
            else:
                args = func_call["arguments"]
            return [FunctionCall(func_call["name"], args)]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Strategy 2: Look for function call patterns
    function_call_patterns = [
        # Pattern: function_call({ "name": "...", "arguments": {...} })
        r"function_call\(\s*(\{.*\})\s*\)",
        # Pattern: { "name": "...", "arguments": {...} }
        r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\})',
        # Pattern: name: "...", arguments: {...}
        r'name\s*:\s*"([^"]+)"\s*,\s*arguments\s*:\s*(\{.*\})',
    ]

    for pattern in function_call_patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            try:
                if len(matches.groups()) == 1:
                    # Full JSON object in one group
                    data = json.loads(matches.group(1))
                    if isinstance(data, dict) and "name" in data and "arguments" in data:
                        if isinstance(data["arguments"], str):
                            args = json.loads(data["arguments"])
                        else:
                            args = data["arguments"]
                        return [FunctionCall(data["name"], args)]
                elif len(matches.groups()) == 2:
                    # Name and arguments in separate groups
                    name = matches.group(1)
                    args = json.loads(matches.group(2))
                    return [FunctionCall(name, args)]
            except (json.JSONDecodeError, IndexError, TypeError):
                continue

    # Strategy 3: Regex-based extraction for less structured outputs
    try:
        # Try to find function name followed by JSON object
        name_match = re.search(r"(?:call|use|execute|invoke)\s+(\w+)\s*\(", text, re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            # Find JSON arguments that follow the function name
            args_match = re.search(r"\{.*\}", text[name_match.end() :], re.DOTALL)
            if args_match:
                args = json.loads(args_match.group(0))
                return [FunctionCall(name, args)]
    except (json.JSONDecodeError, IndexError, TypeError):
        pass

    # If all strategies fail, raise an error
    raise ValueError("No function call could be parsed from the output")


def add_function(name: str, schema: Dict[str, Any], implementation: Callable) -> None:
    """
    Global function to register a function that can be called by models.
    This is a convenience function for simple scripts.

    Args:
        name: Name of the function
        schema: JSON Schema describing the function parameters
        implementation: Python function to execute when called
    """
    from .llm import LLM

    # Create a global registry if needed
    if not hasattr(add_function, "_registry"):
        add_function._registry = FunctionRegistry()

    # Add the function to the registry
    add_function._registry.add_function(name, schema, implementation)

    # Monkey-patch LLM to use this registry by default
    original_init = LLM.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for name, schema in add_function._registry.functions.items():
            implementation = add_function._registry.implementations[name]
            self.add_function(name, schema, implementation)

    if not hasattr(add_function, "_patched"):
        LLM.__init__ = patched_init
        add_function._patched = True
