"""Offline, mock-based tests for the Apple Foundation Models backend.

No ``apple-fm-sdk`` install or macOS 26 hardware is required: the SDK is
replaced with a lightweight fake module injected via ``sys.modules`` so the
conversion, response-mapping, tool-building and streaming-contract logic can
be exercised on any platform.
"""

import asyncio
import json as _json
import sys
import types
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from local_llm_kit.backends.apple_foundation import AppleFoundationBackend


def _build_fake_sdk(available=(True, "ok")) -> types.ModuleType:
    """Construct a minimal stand-in for the ``apple_fm_sdk`` package."""

    fake = types.ModuleType("apple_fm_sdk")

    class _GenerationOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _GenerationSchema:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args

    class _GeneratedContent:
        def __init__(self, values: Optional[Dict[str, Any]] = None) -> None:
            self._values = values or {}

        def value(self, type_class: Any = None, for_property: Optional[str] = None) -> Any:
            if for_property is None:
                return self._values
            return self._values.get(for_property)

        def to_json(self) -> str:
            return _json.dumps(self._values, default=str)

    class _Tool:
        # Real SDK marks name/description/arguments_schema/call abstract; a
        # plain base is enough for subclass-style usage here.
        pass

    def _generable(arg: Any = None):
        def decorator(cls):
            if not hasattr(cls, "generation_schema"):
                cls.generation_schema = classmethod(lambda c: _GenerationSchema())
            return cls

        if isinstance(arg, type):
            return decorator(arg)
        return decorator

    class SystemLanguageModel:
        _available = available

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_kwargs = kwargs

        @classmethod
        def is_available(cls) -> tuple:
            return cls._available

    class LanguageModelSession:
        instances: List["LanguageModelSession"] = []
        next_response: Any = "ok response"

        def __init__(self, instructions: Any = None, model: Any = None, tools: Any = None) -> None:
            self.instructions = instructions
            self.model = model
            self.tools = list(tools or [])
            self.respond_calls: List[Dict[str, Any]] = []
            LanguageModelSession.instances.append(self)

        async def respond(self, prompt: str, options: Any = None, json_schema: Any = None) -> Any:
            self.respond_calls.append(
                {"prompt": prompt, "options": options, "json_schema": json_schema}
            )
            return LanguageModelSession.next_response

    fake.SystemLanguageModel = SystemLanguageModel
    fake.LanguageModelSession = LanguageModelSession
    fake.GenerationOptions = _GenerationOptions
    fake.GenerationSchema = _GenerationSchema
    fake.Tool = _Tool
    fake.generable = _generable
    fake.GeneratedContent = _GeneratedContent
    fake.ToolCallError = type("ToolCallError", (Exception,), {})
    return fake


class _AppleFMSdk:
    """Context manager that swaps in a fake ``apple_fm_sdk`` module."""

    def __init__(self, available=(True, "ok")) -> None:
        self.available = available
        self._patcher: Optional[mock._patch_dict] = None

    def __enter__(self) -> types.ModuleType:
        self.fake = _build_fake_sdk(self.available)
        self._patcher = mock.patch.dict(sys.modules, {"apple_fm_sdk": self.fake})
        self._patcher.start()
        return self.fake

    def __exit__(self, *exc: Any) -> None:
        self._patcher.stop()


def _run_async(coro) -> Any:
    """Run a coroutine on a fresh loop (works on 3.9-3.14, no running loop)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestAppleFoundationImportGuard(unittest.TestCase):
    """The optional SDK is never required to import the backend module."""

    def test_module_imports_without_sdk(self):
        # apple-fm-sdk is not installed in CI; the module must still import.
        from local_llm_kit.backends.apple_foundation import AppleFoundationBackend as Back

        self.assertTrue(Back.supports_native_chat)
        self.assertTrue(Back._accepts_tool_impls)

    def test_instantiation_without_sdk_gives_clear_hint(self):
        # Force `import apple_fm_sdk` to fail by pinning it to None.
        with mock.patch.dict(sys.modules, {"apple_fm_sdk": None}):
            with self.assertRaises(ImportError) as ctx:
                AppleFoundationBackend()
        self.assertIn("local-llm-kit[apple-fm]", str(ctx.exception))


class TestAppleFoundationBackendMocked(unittest.TestCase):
    """Core conversion/response/tool behaviour against a fake SDK (any platform)."""

    def setUp(self) -> None:
        self._ctx = _AppleFMSdk(available=(True, "ok"))
        self.fake = self._ctx.__enter__()
        self.fake.LanguageModelSession.instances = []
        self.fake.LanguageModelSession.next_response = "ok response"
        self.backend = AppleFoundationBackend(model_name="system", instructions="be helpful")

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def _last_session(self) -> Any:
        return self.fake.LanguageModelSession.instances[-1]

    # -- availability -----------------------------------------------------

    def test_unavailable_raises_runtime_error_with_reason(self):
        self.fake.SystemLanguageModel._available = (False, "not running macOS 26")
        with self.assertRaisesRegex(RuntimeError, "not running macOS 26"):
            AppleFoundationBackend()

    def test_init_sanity(self):
        self.assertEqual(self.backend.model, "system")
        self.assertEqual(self.backend.model_path, "system")

    # -- message conversion ----------------------------------------------

    def test_convert_messages_basic(self):
        instruction, prompt = AppleFoundationBackend._convert_messages(
            [{"role": "user", "content": "hi"}], None
        )
        self.assertEqual(instruction, "")
        self.assertEqual(prompt, "User: hi\nAssistant:")

    def test_convert_messages_system_instruction(self):
        instruction, prompt = AppleFoundationBackend._convert_messages(
            [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}],
            None,
        )
        self.assertEqual(instruction, "You are terse.")
        self.assertEqual(prompt, "User: hi\nAssistant:")

    def test_convert_messages_merges_instructions(self):
        instruction, _ = AppleFoundationBackend._convert_messages(
            [{"role": "system", "content": "sys msg"}], "ctor instructions"
        )
        self.assertIn("ctor instructions", instruction)
        self.assertIn("sys msg", instruction)

    def test_convert_messages_multiturn_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "tool_call_id": "1", "content": "22C"},
            {"role": "user", "content": "how about London?"},
        ]
        _, prompt = AppleFoundationBackend._convert_messages(messages, None)
        self.assertIn('Called: get_weather({"location":"Paris"})', prompt)
        self.assertIn("Tool result (get_weather): 22C", prompt)
        self.assertIn("User: how about London?", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_message_text_vision_parts(self):
        text = AppleFoundationBackend._message_text(
            [
                {"type": "text", "text": "look "},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]
        )
        self.assertEqual(text, "look [image]")

    # -- response extraction ---------------------------------------------

    def test_extract_response_str(self):
        content, usage, finish = AppleFoundationBackend._extract_response("hi there")
        self.assertEqual(content, "hi there")
        self.assertIsNone(usage)
        self.assertEqual(finish, "stop")

    def test_extract_response_object_with_usage(self):
        obj = types.SimpleNamespace(content="hello", usage={"input_tokens": 3, "output_tokens": 4})
        content, usage, finish = AppleFoundationBackend._extract_response(obj)
        self.assertEqual(content, "hello")
        self.assertEqual(usage, {"input_tokens": 3, "output_tokens": 4})

    def test_extract_response_dict(self):
        content, usage, _ = AppleFoundationBackend._extract_response(
            {"content": "from dict", "usage": {"input_tokens": 1}}
        )
        self.assertEqual(content, "from dict")
        self.assertEqual(usage, {"input_tokens": 1})

    def test_extract_response_generated_content_to_json(self):
        gc = self.fake.GeneratedContent({"name": "Paris", "temperature": 22})
        content, usage, _ = AppleFoundationBackend._extract_response(gc)
        self.assertEqual(content, '{"name": "Paris", "temperature": 22}')
        self.assertIsNone(usage)

    # -- generate / chat end to end -------------------------------------

    def test_generate_basic(self):
        self.fake.LanguageModelSession.next_response = "generated text"
        result = self.backend.generate("hello", temperature=0.2, max_new_tokens=10)
        self.assertEqual(result["text"], "generated text")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["model"], "system")
        session = self._last_session()
        self.assertEqual(session.respond_calls[0]["prompt"], "hello")
        opts = session.respond_calls[0]["options"]
        self.assertIsNotNone(opts)
        self.assertEqual(opts.kwargs["temperature"], 0.2)
        self.assertEqual(opts.kwargs["maximum_response_tokens"], 10)

    def test_chat_basic(self):
        self.fake.LanguageModelSession.next_response = "hi there"
        result = self.backend.chat([{"role": "user", "content": "say hi"}])
        self.assertEqual(result["text"], "hi there")
        session = self._last_session()
        self.assertIn("User: say hi", session.respond_calls[0]["prompt"])
        self.assertIn("Assistant:", session.respond_calls[0]["prompt"])

    def test_chat_uses_system_instructions(self):
        self.fake.LanguageModelSession.next_response = "ok"
        self.backend.chat(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        )
        session = self._last_session()
        # Constructor instructions are preserved and the system message is appended.
        self.assertIn("be helpful", session.instructions)
        self.assertIn("be brief", session.instructions)

    def test_chat_response_format_json_schema(self):
        self.fake.LanguageModelSession.next_response = '{"name": "Paris"}'
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = self.backend.chat(
            [{"role": "user", "content": "name a city"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "city", "schema": schema},
            },
        )
        session = self._last_session()
        self.assertEqual(session.respond_calls[0]["json_schema"], schema)
        self.assertEqual(result["text"], '{"name": "Paris"}')

    # -- json schema extraction ------------------------------------------

    def test_extract_json_schema_variants(self):
        extract = AppleFoundationBackend._extract_json_schema
        # Non-dict / None
        self.assertIsNone(extract(None))
        self.assertIsNone(extract("json_object"))
        self.assertIsNone(extract({"type": "text"}))

        # json_object -> generic object schema
        self.assertEqual(extract({"type": "json_object"}), {"type": "object"})

        # json_schema with nested .schema
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        self.assertEqual(
            extract({"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}),
            schema,
        )

        # json_schema without .schema but with .type -> return the nested dict
        self.assertEqual(
            extract({"type": "json_schema", "json_schema": schema}),
            schema,
        )

        # json_schema with only a name -> fallback to generic object
        self.assertEqual(
            extract({"type": "json_schema", "json_schema": {"name": "city"}}),
            {"type": "object"},
        )

        # Direct 'schema' key (not nested under json_schema)
        self.assertEqual(
            extract({"type": "json_schema", "schema": schema}),
            schema,
        )

        # Unknown type -> None
        self.assertIsNone(extract({"type": "weird"}))

    # -- tool spec normalization -----------------------------------------

    def test_normalize_tool_spec_openai_style(self):
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
        result = AppleFoundationBackend._normalize_tool_spec(tool)
        self.assertEqual(result["name"], "get_weather")

    def test_normalize_tool_spec_bare_with_name(self):
        tool = {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}
        result = AppleFoundationBackend._normalize_tool_spec(tool)
        self.assertEqual(result["name"], "get_weather")

    def test_normalize_tool_spec_none_and_non_dict(self):
        self.assertIsNone(AppleFoundationBackend._normalize_tool_spec(None))
        self.assertIsNone(AppleFoundationBackend._normalize_tool_spec("not a dict"))
        self.assertIsNone(AppleFoundationBackend._normalize_tool_spec(42))

    def test_normalize_tool_spec_with_spec_attribute(self):
        class ToolObj:
            spec = {"name": "from_spec", "description": "d", "parameters": {"type": "object"}}

        result = AppleFoundationBackend._normalize_tool_spec(ToolObj())
        self.assertEqual(result["name"], "from_spec")

    def test_normalize_tool_spec_missing_function_key(self):
        result = AppleFoundationBackend._normalize_tool_spec({"type": "function"})
        self.assertIsNone(result)

    # -- json type mapping -----------------------------------------------

    def test_json_type_to_python_all_types(self):
        cases = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List[str],
            "object": str,
        }
        for jtype, expected in cases.items():
            self.assertIs(
                AppleFoundationBackend._json_type_to_python({"type": jtype}), expected, jtype
            )

    def test_json_type_to_python_defaults(self):
        self.assertIs(AppleFoundationBackend._json_type_to_python(None), str)
        self.assertIs(AppleFoundationBackend._json_type_to_python("not a dict"), str)
        self.assertIs(AppleFoundationBackend._json_type_to_python({}), str)
        self.assertIs(AppleFoundationBackend._json_type_to_python({"type": "weird"}), str)

    # -- function calling -----------------------------------------------

    def test_tools_converted_to_apple_tool_objects(self):
        spec = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }

        def get_weather(location: str, unit: str = "celsius") -> dict:
            return {"temperature": 22, "unit": unit, "location": location}

        self.fake.LanguageModelSession.next_response = "It is 22C."
        self.backend.chat(
            [{"role": "user", "content": "weather in Paris?"}],
            tools=[spec],
            implementations={"get_weather": get_weather},
        )
        session = self._last_session()
        self.assertEqual(len(session.tools), 1)
        tool = session.tools[0]
        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(tool.description, "Weather for a city")
        self.assertIsInstance(tool.arguments_schema, self.fake.GenerationSchema)

    def test_tool_call_invokes_implementation(self):
        spec = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }

        calls = []

        def get_weather(location: str) -> dict:
            calls.append(location)
            return {"temperature": 22, "location": location}

        tool = next(
            iter(self.backend._build_apple_tools([spec], {"get_weather": get_weather}, True))
        )
        self.assertEqual(tool.name, "get_weather")
        args = self.fake.GeneratedContent({"location": "Paris"})
        result = _run_async(tool.call(args))
        self.assertEqual(calls, ["Paris"])
        self.assertIn("22", result)

    def test_tool_without_implementation_reports_error(self):
        spec = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
        tool = next(iter(self.backend._build_apple_tools([spec], None, True)))
        args = self.fake.GeneratedContent({"location": "Paris"})
        result = _run_async(tool.call(args))
        self.assertIn("no registered implementation", result)

    def test_execute_tools_false_omits_tool_registration(self):
        spec = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
        self.backend.chat(
            [{"role": "user", "content": "hi"}],
            tools=[spec],
            implementations={"get_weather": lambda location: "22C"},
            execute_tools=False,
        )
        session = self._last_session()
        self.assertEqual(session.tools, [])

    # -- streaming is unsupported ----------------------------------------

    def test_streaming_not_supported(self):
        for fn, kwargs in [
            (self.backend.generate_stream, {"prompt": "hi"}),
            (self.backend.chat_stream, {"messages": [{"role": "user", "content": "hi"}]}),
        ]:
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(NotImplementedError):
                    list(fn(**kwargs))
        with self.assertRaises(NotImplementedError):
            self.backend.chat([{"role": "user", "content": "hi"}], stream=True)

    # -- introspection ---------------------------------------------------

    def test_context_window_and_token_count(self):
        self.assertEqual(self.backend.get_context_window(), 4096)
        self.assertEqual(self.backend.count_tokens("one two three"), 3)
        self.assertEqual(self.backend.count_tokens(""), 0)

    def test_model_info_and_list_models(self):
        info = self.backend.model_info()
        self.assertTrue(info["apple_foundation"])
        self.assertFalse(info["streaming"])
        models = self.backend.list_models()
        self.assertEqual(models[0]["id"], "system")
        self.assertEqual(models[0]["owned_by"], "apple-foundation")


@unittest.skipUnless(sys.platform == "darwin", "macOS only: exercises the real availability check")
class TestAppleFoundationPlatform(unittest.TestCase):
    def test_apple_fm_available_is_bool_or_false(self):
        from local_llm_kit.backends.apple_foundation import apple_fm_available

        self.assertIsInstance(apple_fm_available(), bool)


if __name__ == "__main__":
    unittest.main()
