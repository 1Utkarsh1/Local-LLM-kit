"""Regression tests for audit fixes: strict validation at public entry points.

Covers: ToolCall/FunctionCall null-field rejection (the email question),
LLM.chat message/response_format validation, LLM model_path validation,
MessageHistory type checks, SimpleVectorStore corrupt-file handling, and
model-download destination guards. All offline; no downloads, no heavy deps.

Run: python -m pytest tests/test_regression_audit.py -q
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from local_llm_kit import LLM
from local_llm_kit.function_calling import FunctionCall
from local_llm_kit.memory import MessageHistory
from local_llm_kit.rag import SimpleVectorStore
from local_llm_kit.tools import ToolCall


def make_echo():
    return LLM(model_path="echo", backend="echo")


class TestToolCallStrictValidation(unittest.TestCase):
    def test_valid_records_still_parse(self):
        tc = ToolCall.from_dict({"function": {"name": "w", "arguments": '{"a": 1}'}})
        self.assertEqual((tc.name, tc.arguments), ("w", {"a": 1}))
        tc = ToolCall.from_dict({"function": {"name": "w"}})
        self.assertEqual(tc.arguments, {})
        tc = ToolCall.from_dict({"name": "w", "arguments": {"a": 1}})
        self.assertEqual(tc.name, "w")

    def test_non_dict_records_raise_type_error(self):
        for bad in (None, "str", [], [{"function": {"name": "w"}}]):
            with self.assertRaises(TypeError, msg="ToolCall %r" % (bad,)):
                ToolCall.from_dict(bad)

    def test_null_function_raises_type_error(self):
        with self.assertRaises(TypeError):
            ToolCall.from_dict({"function": None})
        with self.assertRaises(TypeError):
            ToolCall.from_dict({"function": "str"})

    def test_null_or_missing_name_raises_value_error(self):
        for bad in (
            {"function": {"name": None, "arguments": {}}},
            {"function": {"arguments": {}}},
            {"function": {"name": "", "arguments": {}}},
            {"function": {"name": 123, "arguments": {}}},
            {},
        ):
            with self.assertRaises(ValueError, msg="ToolCall %r" % (bad,)):
                ToolCall.from_dict(bad)

    def test_null_arguments_raises_value_error(self):
        with self.assertRaises(ValueError):
            ToolCall.from_dict({"function": {"name": "w", "arguments": None}})
        with self.assertRaises(ValueError):
            ToolCall.from_dict({"function": {"name": "w", "arguments": [1, 2]}})


class TestFunctionCallStrictValidation(unittest.TestCase):
    def test_valid_records_still_parse(self):
        fc = FunctionCall.from_dict({"name": "w", "arguments": '{"a": 1}'})
        self.assertEqual((fc.name, fc.arguments), ("w", {"a": 1}))
        fc = FunctionCall.from_dict({"name": "w", "arguments": {"a": 1}})
        self.assertEqual(fc.arguments, {"a": 1})
        fc = FunctionCall.from_dict({"name": "w", "arguments": ""})
        self.assertEqual(fc.arguments, {})

    def test_non_dict_raises_type_error(self):
        for bad in (None, "str", []):
            with self.assertRaises(TypeError, msg="FunctionCall %r" % (bad,)):
                FunctionCall.from_dict(bad)

    def test_null_name_or_arguments_raise_value_error(self):
        for bad in (
            {},
            {"name": "x"},
            {"name": None, "arguments": {}},
            {"name": "", "arguments": {}},
            {"name": "x", "arguments": None},
        ):
            with self.assertRaises(ValueError, msg="FunctionCall %r" % (bad,)):
                FunctionCall.from_dict(bad)


class TestLLMChatValidation(unittest.TestCase):
    def test_non_list_messages_raise_type_error(self):
        llm = make_echo()
        for bad in (None, "str", {"role": "user"}):
            with self.assertRaises(TypeError, msg="messages %r" % (bad,)):
                llm.chat(messages=bad)

    def test_non_dict_or_roleless_messages_raise(self):
        llm = make_echo()
        with self.assertRaises(TypeError):
            llm.chat(messages=[None])
        for bad in (
            [{"content": "hi"}],
            [{"role": 123, "content": "x"}],
            [{"role": "", "content": "x"}],
        ):
            with self.assertRaises(ValueError, msg="messages %r" % (bad,)):
                llm.chat(messages=bad)

    def test_optional_content_still_accepted(self):
        # content=None / missing content / empty history are legitimate.
        for good in ([{"role": "user"}], [{"role": "user", "content": None}], []):
            resp = make_echo().chat(messages=good)
            self.assertIn("choices", resp)

    def test_bad_response_format_raises_not_attribute_error(self):
        llm = make_echo()
        with self.assertRaises(TypeError):
            llm.chat(messages=[{"role": "user", "content": "hi"}], response_format=123)
        with self.assertRaises(TypeError):
            llm.chat(messages=[{"role": "user", "content": "hi"}], response_format=["json_object"])
        with self.assertRaises(ValueError):
            llm.chat(
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "bogus"},
            )

    def test_bad_model_path_rejected(self):
        for bad in (None, "", 123):
            with self.assertRaises(ValueError, msg="model_path %r" % (bad,)):
                LLM(model_path=bad, backend="echo")


class TestMemoryValidation(unittest.TestCase):
    def test_add_messages_rejects_non_list(self):
        with self.assertRaises(TypeError):
            MessageHistory().add_messages(None)

    def test_add_messages_rejects_non_dict_entries(self):
        with self.assertRaises(TypeError):
            MessageHistory().add_messages([None])

    def test_add_message_rejects_non_dict(self):
        with self.assertRaises(TypeError):
            MessageHistory().add_message(None)


class TestVectorStoreLoadValidation(unittest.TestCase):
    def _write(self, payload):
        path = os.path.join(tempfile.mkdtemp(), "store.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
        return path

    def test_null_fields_raise_value_error_not_type_error(self):
        path = self._write({"ids": None, "texts": None, "metadatas": None, "vectors": None})
        with self.assertRaises(ValueError):
            SimpleVectorStore.load(path)

    def test_round_trip_still_works(self):
        store = SimpleVectorStore(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
        store.add(["hello"], metadatas=[{"source": "a"}])
        path = os.path.join(tempfile.mkdtemp(), "store.json")
        store.save(path)
        loaded = SimpleVectorStore.load(path, embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
        self.assertEqual(loaded.texts, ["hello"])


class TestDownloadDestinationGuards(unittest.TestCase):
    def test_directory_destination_refused(self):
        from local_llm_kit import models

        cache = tempfile.mkdtemp()
        os.makedirs(os.path.join(cache, "urls", "f.gguf"))
        with self.assertRaises(ValueError):
            models._download_url(
                "https://huggingface.co/x/f.gguf", os.path.join(cache, "urls", "f.gguf")
            )

    def test_unusable_url_filename_rejected(self):
        from local_llm_kit import models

        with self.assertRaises(ValueError):
            models.download_model(
                "https://huggingface.co/x/f.gguf",
                filename="..",
                cache_dir=tempfile.mkdtemp(),
            )

    def test_valid_url_filename_still_accepted(self):
        from local_llm_kit import models

        cache = tempfile.mkdtemp()
        with mock.patch("local_llm_kit.models._download_url", return_value="MOCK") as m:
            out = models.download_model("https://huggingface.co/x/f.gguf", cache_dir=cache)
        self.assertEqual(out, "MOCK")
        dest = m.call_args[0][1]
        self.assertTrue(dest.endswith(os.path.join("urls", "f.gguf")))


if __name__ == "__main__":
    unittest.main()
