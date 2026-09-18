"""
Tests for function calling functionality.
"""

import json
import unittest

from local_llm_kit.function_calling import FunctionCall, FunctionRegistry, parse_function_calls


class TestFunctionCalling(unittest.TestCase):
    """Tests for function calling functionality."""

    def test_function_call_creation(self):
        """Test creation of FunctionCall objects."""
        # Create a function call directly
        func_call = FunctionCall(name="test_function", arguments={"arg1": "value1", "arg2": 42})

        self.assertEqual(func_call.name, "test_function")
        self.assertEqual(func_call.arguments["arg1"], "value1")
        self.assertEqual(func_call.arguments["arg2"], 42)

        # Create a function call from dict
        func_call_dict = {"name": "another_function", "arguments": json.dumps({"key": "value"})}

        func_call = FunctionCall.from_dict(func_call_dict)
        self.assertEqual(func_call.name, "another_function")
        self.assertEqual(func_call.arguments["key"], "value")

        # Test to_dict method
        result_dict = func_call.to_dict()
        self.assertEqual(result_dict["name"], "another_function")
        self.assertIn("key", json.loads(result_dict["arguments"]))

    def test_function_registry(self):
        """Test function registry functionality."""
        # Create registry
        registry = FunctionRegistry()

        # Define test function and schema
        def test_func(x, y=0):
            return x + y

        schema = {
            "name": "test_func",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number", "default": 0}},
                "required": ["x"],
            },
        }

        # Register the function
        registry.add_function(name="test_func", schema=schema, implementation=test_func)

        # Test registry state
        self.assertTrue(registry.has_functions())
        self.assertTrue(registry.has_function("test_func"))
        self.assertFalse(registry.has_function("nonexistent_func"))

        schemas = registry.get_schema_list()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "test_func")

        # Test function execution
        result = registry.execute("test_func", {"x": 5, "y": 3})
        self.assertEqual(result, 8)

        # Test execution with default parameter
        result = registry.execute("test_func", {"x": 5})
        self.assertEqual(result, 5)

        # Test execution with invalid function
        with self.assertRaises(ValueError):
            registry.execute("nonexistent_func", {})

    def test_function_call_parsing(self):
        """Test parsing function calls from model output."""
        # Direct JSON
        json_output = '{"name": "get_weather", "arguments": {"location": "Paris"}}'
        func_calls = parse_function_calls(json_output)
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        self.assertEqual(func_calls[0].arguments["location"], "Paris")

        # Function call with nested JSON
        nested_output = (
            "I need to call a function.\n"
            'function_call({"name": "calculate", "arguments": {"expression": "5+7"}})\n'
            "to get the result."
        )
        func_calls = parse_function_calls(nested_output)
        self.assertEqual(func_calls[0].name, "calculate")
        self.assertEqual(func_calls[0].arguments["expression"], "5+7")

        # Function call with explicit name and args
        pattern_output = 'name: "search_database", arguments: {"query": "users", "limit": 10}'
        func_calls = parse_function_calls(pattern_output)
        self.assertEqual(func_calls[0].name, "search_database")
        self.assertEqual(func_calls[0].arguments["query"], "users")

        # Function call with more complex structure
        complex_output = (
            "To answer your question, I need to execute the following function:\n\n"
            "call fetch_data({\n"
            '  "source": "api",\n'
            '  "endpoint": "/users",\n'
            '  "params": {\n'
            '    "active": true,\n'
            '    "limit": 5\n'
            "  }\n"
            "})"
        )
        func_calls = parse_function_calls(complex_output)
        self.assertEqual(func_calls[0].name, "fetch_data")
        self.assertEqual(func_calls[0].arguments["source"], "api")
        self.assertEqual(func_calls[0].arguments["params"]["limit"], 5)


if __name__ == "__main__":
    unittest.main()
