"""Offline tests for modern tool calling (local_llm_kit.tools).

Covers: @tool decorator, function_to_json_schema, normalize_tools,
parse_tool_calls (incl. <tool_call> tags), ToolCall, ToolRegistry.execute,
should_call_tools. Stdlib unittest only; no model downloads, no
torch/transformers. Python 3.9 compatible.

Run: python -m pytest tests/test_tools.py -q
"""
import inspect
import json
import subprocess
import sys
import unittest
from typing import Dict, Optional, Union

from local_llm_kit.tools import (
    FunctionCall,
    FunctionRegistry,
    ToolCall,
    ToolRegistry,
    function_to_json_schema,
    normalize_tools,
    parse_function_calls,
    parse_tool_calls,
    should_call_tools,
    tool,
)


def _repo_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCoreStaysLight(unittest.TestCase):
    """Core import must not pull torch/transformers (light-core requirement)."""

    def test_import_tools_pulls_no_heavy_deps(self):
        code = (
            "import sys; "
            "import local_llm_kit.tools as t; "
            "bad = [m for m in ('torch', 'transformers', 'llama_cpp', 'llama-cpp-python') if m in sys.modules]; "
            "print(','.join(bad))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=_repo_root(),
        )
        self.assertEqual(out.returncode, 0, msg="import local_llm_kit.tools failed offline: %s" % out.stderr[-2000:])
        self.assertEqual(out.stdout.strip(), "", msg="heavy dep imported at core import: %s" % out.stdout.strip())


class TestFunctionToJsonSchema(unittest.TestCase):
    def test_basic_types_and_required(self):
        def fn(name: str, count: int, ratio: float, flag: bool, meta: dict, tags: list):
            pass

        schema = function_to_json_schema(fn)
        self.assertEqual(schema["type"], "object")
        props = schema["properties"]
        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["count"]["type"], "integer")
        self.assertEqual(props["ratio"]["type"], "number")
        self.assertEqual(props["flag"]["type"], "boolean")
        self.assertEqual(props["meta"]["type"], "object")
        self.assertEqual(props["tags"]["type"], "array")
        self.assertEqual(sorted(schema["required"]),
                         ["count", "flag", "meta", "name", "ratio", "tags"])

    def test_defaults_become_optional_with_default(self):
        def fn(location: str, unit: str = "celsius", limit: int = 5):
            pass

        schema = function_to_json_schema(fn)
        self.assertEqual(schema["required"], ["location"])
        self.assertEqual(schema["properties"]["unit"]["default"], "celsius")
        self.assertEqual(schema["properties"]["limit"]["default"], 5)

    def test_untyped_and_unknown_annotations_default_to_string(self):
        def fn(a, b=None):
            pass

        class Custom(object):
            pass

        def fn2(x: Custom):
            pass

        schema = function_to_json_schema(fn)
        self.assertEqual(schema["properties"]["a"]["type"], "string")
        self.assertEqual(function_to_json_schema(fn2)["properties"]["x"]["type"], "string")

    def test_optional_unwraps_to_inner_type(self):
        def fn(name: Optional[str], count: Optional[int] = None):
            pass

        props = function_to_json_schema(fn)["properties"]
        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["count"]["type"], "integer")

    def test_union_of_two_types_falls_back_to_string(self):
        def fn(x: Union[int, str]):
            pass

        self.assertEqual(function_to_json_schema(fn)["properties"]["x"]["type"], "string")

    def test_self_is_skipped(self):
        class A(object):
            def method(self, x: int):
                pass

        schema = function_to_json_schema(A().method)
        self.assertEqual(list(schema["properties"]), ["x"])
        self.assertEqual(schema["required"], ["x"])

    def test_non_serializable_default_is_ignored_but_stays_optional(self):
        def fn(cb=lambda x: x):
            pass

        schema = function_to_json_schema(fn)
        self.assertEqual(schema["required"], [])
        self.assertNotIn("default", schema["properties"]["cb"])


class TestToolDecorator(unittest.TestCase):
    def test_bare_decorator(self):
        @tool
        def add(a: int, b: int = 0):
            """Add two numbers."""
            return a + b

        self.assertEqual(add(2, 3), 5)  # still callable
        self.assertEqual(add.tool_name, "add")
        spec = add.spec
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "add")
        self.assertEqual(spec["function"]["description"], "Add two numbers.")
        self.assertEqual(spec["function"]["parameters"]["required"], ["a"])
        self.assertEqual(spec["function"]["parameters"]["properties"]["b"]["default"], 0)

    def test_decorator_with_description_override(self):
        @tool(description="overridden")
        def greet(name: str):
            """Original docstring."""
            return "hi " + name

        self.assertEqual(greet.spec["function"]["description"], "overridden")
        self.assertEqual(greet.tool_name, "greet")

    def test_decorator_with_explicit_name(self):
        @tool(name="custom_name", description="d")
        def fn(x: str):
            return x

        self.assertEqual(fn.spec["function"]["name"], "custom_name")
        self.assertEqual(fn.tool_name, "custom_name")

    def test_empty_docstring_gives_empty_description(self):
        @tool
        def fn(x):
            pass

        self.assertEqual(fn.spec["function"]["description"], "")


class TestNormalizeTools(unittest.TestCase):
    def test_none_and_empty_give_empty_list(self):
        self.assertEqual(normalize_tools(), [])
        self.assertEqual(normalize_tools(tools=[]), [])
        self.assertEqual(normalize_tools(tools=None, functions=None), [])

    def test_openai_dict_passes_through(self):
        spec = {"type": "function", "function": {"name": "w", "description": "d",
                "parameters": {"type": "object", "properties": {}}}}
        self.assertEqual(normalize_tools(tools=[spec]), [spec])

    def test_bare_function_spec_gets_wrapped(self):
        bare = {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}
        out = normalize_tools(tools=[bare])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(out[0]["function"]["name"], "f")

    def test_tool_decorated_callable_uses_its_spec(self):
        @tool(description="weather")
        def get_weather(location: str):
            """Get weather."""
            return location

        out = normalize_tools(tools=[get_weather])
        self.assertEqual(out, [get_weather.spec])

    def test_plain_callable_gets_schema_built(self):
        def search(query: str, limit: int = 10):
            """Search things."""
            return query

        out = normalize_tools(tools=[search])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "search")
        self.assertEqual(out[0]["function"]["description"], "Search things.")
        self.assertEqual(out[0]["function"]["parameters"]["required"], ["query"])

    def test_legacy_functions_are_merged_and_wrapped(self):
        legacy = [{"name": "old_fn", "description": "legacy",
                   "parameters": {"type": "object", "properties": {}}}]
        out = normalize_tools(functions=legacy)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(out[0]["function"]["name"], "old_fn")

    def test_dedupe_last_wins(self):
        a = {"type": "function", "function": {"name": "dup", "description": "first",
             "parameters": {"type": "object", "properties": {}}}}
        b = {"type": "function", "function": {"name": "dup", "description": "second",
             "parameters": {"type": "object", "properties": {}}}}
        out = normalize_tools(tools=[a, b])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["description"], "second")

    def test_tools_and_functions_combine(self):
        t = {"type": "function", "function": {"name": "new", "description": "",
             "parameters": {"type": "object", "properties": {}}}}
        f = {"name": "old", "description": "", "parameters": {"type": "object", "properties": {}}}
        out = normalize_tools(tools=[t], functions=[f])
        self.assertEqual(sorted(x["function"]["name"] for x in out), ["new", "old"])


class TestParseToolCalls(unittest.TestCase):
    def test_openai_tool_calls_json_with_string_args(self):
        payload = {"tool_calls": [{"id": "call_1", "type": "function",
                                   "function": {"name": "w", "arguments": json.dumps({"q": 1})}}]}
        calls = parse_tool_calls(json.dumps(payload))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "w")
        self.assertEqual(calls[0].arguments, {"q": 1})
        self.assertEqual(calls[0].id, "call_1")

    def test_openai_tool_calls_json_with_dict_args(self):
        payload = {"tool_calls": [{"id": "call_2", "type": "function",
                                   "function": {"name": "w", "arguments": {"q": 2}}}]}
        calls = parse_tool_calls(json.dumps(payload))
        self.assertEqual(calls[0].arguments, {"q": 2})

    def test_single_tool_call_tag(self):
        text = ('Thinking... <tool_call>{"name": "get_weather", '
                '"arguments": {"location": "Paris"}}</tool_call> done.')
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments, {"location": "Paris"})

    def test_parallel_tool_call_tag_array(self):
        text = ('<tool_call>[{"name": "a", "arguments": {}}, '
                '{"name": "b", "arguments": {"x": 1}}]</tool_call>')
        calls = parse_tool_calls(text)
        self.assertEqual([(c.name, c.arguments) for c in calls],
                         [("a", {}), ("b", {"x": 1})])

    def test_tool_calls_bracket_block(self):
        text = '[TOOL_CALLS][{"name": "a", "arguments": {}}][/TOOL_CALLS]'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "a")

    def test_bare_name_arguments_json_fallback(self):
        calls = parse_tool_calls(json.dumps({"name": "get_weather", "arguments": {"location": "Paris"}}))
        self.assertEqual(calls[0].name, "get_weather")

    def test_legacy_function_call_syntax_fallback(self):
        calls = parse_tool_calls('function_call({"name": "calculate", "arguments": {"expression": "5+7"}})')
        self.assertEqual(calls[0].name, "calculate")
        self.assertEqual(calls[0].arguments["expression"], "5+7")

    def test_plain_text_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_tool_calls("Hello, how are you today?")
        with self.assertRaises(ValueError):
            parse_tool_calls("")


class TestToolCallObject(unittest.TestCase):
    def test_to_from_dict_roundtrip(self):
        tc = ToolCall(name="w", arguments={"q": 1}, call_id="call_9")
        d = tc.to_dict()
        self.assertEqual(d["id"], "call_9")
        self.assertEqual(d["type"], "function")
        self.assertEqual(json.loads(d["function"]["arguments"]), {"q": 1})
        tc2 = ToolCall.from_dict(d)
        self.assertEqual((tc2.name, tc2.arguments, tc2.id), ("w", {"q": 1}, "call_9"))

    def test_from_dict_accepts_string_and_empty_args(self):
        tc = ToolCall.from_dict({"function": {"name": "w", "arguments": '{"a": 1}'}})
        self.assertEqual(tc.arguments, {"a": 1})
        tc = ToolCall.from_dict({"function": {"name": "w", "arguments": ""}})
        self.assertEqual(tc.arguments, {})

    def test_auto_id_is_stable_string(self):
        tc = ToolCall(name="w", arguments={"q": 1})
        self.assertTrue(isinstance(tc.id, str) and tc.id)

    def test_to_function_call_bridge(self):
        tc = ToolCall(name="w", arguments={"q": 1})
        fc = tc.to_function_call()
        self.assertIsInstance(fc, FunctionCall)
        self.assertEqual((fc.name, fc.arguments), ("w", {"q": 1}))


class TestToolRegistry(unittest.TestCase):
    def test_add_tool_with_decorated_function_and_execute(self):
        @tool(description="weather")
        def get_weather(location: str, unit: str = "celsius"):
            """Get weather."""
            return {"location": location, "unit": unit}

        reg = ToolRegistry()
        spec = reg.add_tool(get_weather)
        self.assertEqual(spec["function"]["name"], "get_weather")
        result = reg.execute("get_weather", {"location": "Paris"})
        self.assertEqual(result, {"location": "Paris", "unit": "celsius"})

    def test_add_tool_with_plain_callable_builds_spec(self):
        def search(query: str):
            """Search."""
            return "found:" + query

        reg = ToolRegistry()
        spec = reg.add_tool(search)
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "search")
        self.assertEqual(reg.execute("search", {"query": "x"}), "found:x")

    def test_get_tool_list_openai_shape(self):
        reg = ToolRegistry()

        def fn(x: int):
            """Doc."""
            return x

        reg.add_tool(fn)
        tools = reg.get_tool_list()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "fn")

    def test_registry_is_backwards_compatible_function_registry(self):
        reg = ToolRegistry()
        self.assertIsInstance(reg, FunctionRegistry)

        def impl(x, y=0):
            return x + y

        reg.add_function(name="adder",
                         schema={"name": "adder", "description": "",
                                 "parameters": {"type": "object"}},
                         implementation=impl)
        self.assertTrue(reg.has_function("adder"))
        self.assertEqual(reg.execute("adder", {"x": 2, "y": 3}), 5)

    def test_execute_unknown_raises_value_error(self):
        with self.assertRaises(ValueError):
            ToolRegistry().execute("nope", {})

    def test_implementation_errors_propagate(self):
        def boom(x: int):
            raise RuntimeError("kaput")

        reg = ToolRegistry()
        reg.add_tool(boom)
        with self.assertRaises(RuntimeError):
            reg.execute("boom", {"x": 1})


class TestShouldCallTools(unittest.TestCase):
    def test_matrix(self):
        self.assertFalse(should_call_tools("none", "auto"))
        self.assertFalse(should_call_tools("none", "none"))
        self.assertTrue(should_call_tools("auto", "auto"))
        self.assertTrue(should_call_tools({"name": "w"}, "auto"))
        self.assertFalse(should_call_tools(None, "none"))
        self.assertTrue(should_call_tools(None, "auto"))


if __name__ == "__main__":
    unittest.main()
