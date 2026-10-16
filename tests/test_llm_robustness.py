"""Offline tests for LLM request-path robustness (local_llm_kit.llm).

Covers: ``tool_timeout`` actually bounding a slow tool, non-JSON-serializable
tool results becoming tool output instead of a traceback, malformed
``messages`` being rejected without corrupting the instance's history, and
JSON-mode extraction keeping nested structures intact. A backend double is
injected, so no model, network or optional dependency is needed. Stdlib
unittest only; Py3.9 compatible.

Run: python -m pytest tests/test_llm_robustness.py -q
"""

import datetime
import json
import time
import unittest

from local_llm_kit import LLM, tool
from local_llm_kit.backends.base import BaseBackend


class RawTextBackend(BaseBackend):
    """Backend double that always returns a fixed string."""

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, **kwargs):
        return {"text": self.text, "finish_reason": "stop"}

    def generate_stream(self, prompt, **kwargs):
        yield {"text": self.text, "finish_reason": "stop"}

    def get_context_window(self):
        return 1024

    def count_tokens(self, text):
        return max(1, len(text) // 4)


class ToolCallBackend(RawTextBackend):
    """Backend double that emits a legacy-style function call."""

    def __init__(self, name, arguments="{}"):
        super().__init__('function_call({"name": "%s", "arguments": %s})' % (name, arguments))


def make_llm(backend):
    """LLM wired to *backend* with caching disabled (no shared state)."""
    return LLM(model_path="raw-model", backend_instance=backend, use_cache=False)


def tool_messages(llm):
    return [m for m in llm.memory.messages if m.get("role") == "tool"]


class TestToolTimeout(unittest.TestCase):
    def test_slow_tool_is_abandoned_at_the_timeout(self):
        @tool(description="Sleeps far longer than the timeout")
        def slow_tool() -> str:
            """Slow."""
            time.sleep(2.0)
            return "finished"

        llm = make_llm(ToolCallBackend("slow_tool"))
        llm.add_tool(slow_tool)

        started = time.monotonic()
        llm.chat(
            messages=[{"role": "user", "content": "go"}],
            tools=[slow_tool.spec],
            tool_timeout=0.25,
            max_iterations=1,
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.0, msg="tool_timeout did not bound the request")
        outputs = [m["content"] for m in tool_messages(llm)]
        self.assertTrue(outputs, msg="no tool result recorded")
        self.assertIn("timed out", outputs[0])


class TestToolResultSerialisation(unittest.TestCase):
    def test_non_json_serialisable_result_is_returned_as_tool_output(self):
        @tool(description="Returns a datetime")
        def when_tool() -> dict:
            """When."""
            return {"now": datetime.datetime(2026, 1, 1, 12, 0, 0)}

        llm = make_llm(ToolCallBackend("when_tool"))
        llm.add_tool(when_tool)

        resp = llm.chat(
            messages=[{"role": "user", "content": "go"}],
            tools=[when_tool.spec],
            max_iterations=1,
        )
        self.assertIn("choices", resp)
        outputs = [m["content"] for m in tool_messages(llm)]
        self.assertTrue(outputs, msg="no tool result recorded")
        self.assertIn("2026-01-01 12:00:00", outputs[0])


class TestMalformedMessages(unittest.TestCase):
    def test_non_dict_messages_are_rejected_without_corrupting_history(self):
        llm = make_llm(RawTextBackend("ok"))

        with self.assertRaises(TypeError):
            llm.chat(messages=["not-a-dict"])
        self.assertEqual(llm.memory.messages, [], msg="history was corrupted")

        # The same instance must still serve a well-formed request.
        resp = llm.chat(messages=[{"role": "user", "content": "hi"}])
        self.assertIn("choices", resp)

    def test_a_later_valid_call_is_unaffected_by_an_invalid_one(self):
        llm = make_llm(RawTextBackend("ok"))
        for bad in ("a-string", None, {"role": "user"}):
            with self.assertRaises(TypeError):
                llm.chat(messages=["ok", bad])
        self.assertEqual(llm.memory.messages, [])

    def test_memory_rejects_non_dict_entries_atomically(self):
        llm = make_llm(RawTextBackend("ok"))
        with self.assertRaises(TypeError):
            llm.memory.add_messages([{"role": "user", "content": "ok"}, 42])
        self.assertEqual(llm.memory.messages, [])
        with self.assertRaises(TypeError):
            llm.memory.add_message("nope")
        self.assertEqual(llm.memory.messages, [])


class TestJsonExtraction(unittest.TestCase):
    CASES = [
        'Sure, here it is: {"name": "Paris", "population": 2}',
        'Sure, here it is: {"name": "Paris", "geo": {"lat": 48.8, "lon": 2.3}}',
        'Answer: [{"a": 1}, {"b": {"c": 2}}]',
        'Nested deeply: {"a": {"b": {"c": [1, {"d": 2}]}}}',
    ]

    def test_nested_json_survives_json_mode(self):
        for raw in self.CASES:
            with self.subTest(text=raw):
                expected = raw.split(": ", 1)[1]
                llm = make_llm(RawTextBackend(raw))
                resp = llm.chat(
                    messages=[{"role": "user", "content": "x"}],
                    response_format={"type": "json_object"},
                )
                content = resp["choices"][0]["message"]["content"]
                self.assertEqual(content, expected)
                json.loads(content)  # must actually be valid JSON

    def test_plain_json_passes_through_unchanged(self):
        raw = '{"a": 1}'
        llm = make_llm(RawTextBackend(raw))
        resp = llm.chat(
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )
        self.assertEqual(resp["choices"][0]["message"]["content"], raw)


if __name__ == "__main__":
    unittest.main()
