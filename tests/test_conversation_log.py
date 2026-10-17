"""Offline tests for the SQLite conversation logging module.

Uses a temporary database file (never the real XDG default path) so tests
are hermetic.  Stdlib unittest + sqlite3 only; no model downloads.  Python 3.9
compatible.

Run: python -m pytest tests/test_conversation_log.py -q
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from local_llm_kit.conversation_log import (
    DEFAULT_LOG_PATH,
    ConversationLog,
    get_conversation_log,
)


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


class TestConversationLogCRUD(unittest.TestCase):
    def setUp(self):
        self.db_path = _tmp_db()
        self.addCleanup(os.unlink, self.db_path)
        self.log = ConversationLog(self.db_path)
        self.addCleanup(self.log.close)

    def test_create_and_get_conversation(self):
        conv_id = self.log.create(model="echo", system_prompt="be helpful")
        self.assertEqual(len(conv_id), 32)
        conv = self.log.get_conversation(conv_id)
        self.assertEqual(conv["model"], "echo")
        self.assertEqual(conv["system_prompt"], "be helpful")
        self.assertIsNone(conv["metadata"])

    def test_get_conversation_missing_raises(self):
        with self.assertRaises(KeyError):
            self.log.get_conversation("nope")

    def test_create_with_metadata(self):
        conv_id = self.log.create(model="gpt-4o", metadata={"user": "abc"})
        conv = self.log.get_conversation(conv_id)
        self.assertEqual(conv["metadata"], {"user": "abc"})

    def test_delete_conversation_cascade(self):
        conv_id = self.log.create(model="echo")
        mid = self.log.add_message(conv_id, "user", "hello")
        _tid = self.log.add_tool_call(mid, "weather", {"city": "Paris"})
        self.assertEqual(self.log.delete(conv_id), 1)
        self.assertEqual(self.log.get_conversations(), [])
        self.assertEqual(self.log.get_messages(conv_id), [])

    def test_clear_removes_all(self):
        a = self.log.create(model="echo")
        self.log.create(model="echo")
        self.log.add_message(a, "user", "hi")
        self.log.clear()
        self.assertEqual(self.log.get_conversations(), [])


class TestMessagesAndTools(unittest.TestCase):
    def setUp(self):
        self.db_path = _tmp_db()
        self.addCleanup(os.unlink, self.db_path)
        self.log = ConversationLog(self.db_path)
        self.addCleanup(self.log.close)
        self.conv_id = self.log.create(model="echo")

    def test_add_and_retrieve_messages(self):
        self.log.add_message(self.conv_id, "user", "hello")
        self.log.add_message(
            self.conv_id, "assistant", "hi there", model="echo", finish_reason="stop"
        )
        msgs = self.log.get_messages(self.conv_id)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "hello")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["finish_reason"], "stop")
        self.assertEqual(msgs[1]["model"], "echo")
        self.assertEqual(msgs[1]["prompt_tokens"], None)

    def test_add_tool_call_arguments_dict_and_str(self):
        mid = self.log.add_message(self.conv_id, "assistant", None)
        self.log.add_tool_call(mid, "weather", {"city": "Paris"}, result="sunny")
        self.log.add_tool_call(mid, "search", '{"q": "cats"}', result=None)
        calls = self.log.get_tool_calls(mid)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "weather")
        self.assertEqual(calls[0]["arguments"], {"city": "Paris"})
        self.assertEqual(calls[0]["result"], "sunny")
        self.assertEqual(calls[1]["name"], "search")
        self.assertEqual(calls[1]["arguments"], {"q": "cats"})

    def test_get_conversation_history_enriched(self):
        mid = self.log.add_message(self.conv_id, "assistant", "call")
        self.log.add_tool_call(mid, "weather", {"city": "Paris"})
        history = self.log.get_conversation_history(self.conv_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["tool_calls"][0]["name"], "weather")

    def test_get_conversation_history_no_tool_details(self):
        self.log.add_message(self.conv_id, "user", "hi")
        history = self.log.get_conversation_history(self.conv_id, include_tool_details=False)
        self.assertNotIn("tool_calls", history[0])

    def test_search_like(self):
        self.log.add_message(self.conv_id, "user", "tell me about cats")
        self.log.add_message(self.conv_id, "user", "something else")
        results = self.log.search("cats")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "tell me about cats")


class TestLogInteraction(unittest.TestCase):
    def setUp(self):
        self.db_path = _tmp_db()
        self.addCleanup(os.unlink, self.db_path)
        self.log = ConversationLog(self.db_path)
        self.addCleanup(self.log.close)

    def test_log_interaction_creates_conversation(self):
        conv_id = self.log.log_interaction(
            model="echo",
            messages=[{"role": "user", "content": "hello"}],
            response={"choices": [{"message": {"content": "hi", "role": "assistant"}}]},
        )
        history = self.log.get_conversation_history(conv_id)
        self.assertEqual(len(history), 2)  # user + assistant

    def test_log_interaction_with_tool_calls(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    }
                }
            ]
        }
        conv_id = self.log.log_interaction(
            model="echo",
            messages=[{"role": "user", "content": "weather?"}],
            response=resp,
        )
        history = self.log.get_conversation_history(conv_id)
        assistant_msg = history[1]
        self.assertEqual(assistant_msg["tool_calls"][0]["name"], "weather")
        self.assertEqual(assistant_msg["tool_calls"][0]["arguments"], {"city": "Paris"})

    def test_log_interaction_reuses_conversation_id(self):
        conv_id = self.log.log_interaction(
            model="echo",
            messages=[{"role": "user", "content": "first"}],
            response={"choices": [{"message": {"content": "one"}}]},
        )
        self.log.log_interaction(
            model="echo",
            messages=[{"role": "user", "content": "second"}],
            response={"choices": [{"message": {"content": "two"}}]},
            conversation_id=conv_id,
        )
        msgs = self.log.get_messages(conv_id)
        self.assertEqual(len(msgs), 4)

    def test_log_interaction_plain_string_response(self):
        conv_id = self.log.log_interaction(
            model="echo",
            messages=[{"role": "user", "content": "hi"}],
            response="hello back",
        )
        msgs = self.log.get_messages(conv_id)
        self.assertEqual(msgs[-1]["content"], "hello back")


class TestExportConversation(unittest.TestCase):
    def setUp(self):
        self.db_path = _tmp_db()
        self.addCleanup(os.unlink, self.db_path)
        self.log = ConversationLog(self.db_path)
        self.addCleanup(self.log.close)
        self.conv_id = self.log.log_interaction(
            model="echo",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "be nice"},
            ],
            response={"choices": [{"message": {"content": "hi"}}]},
        )

    def test_export_json(self):
        out = self.log.export_conversation(self.conv_id, "json")
        data = json.loads(out)
        self.assertIn("conversation", data)
        self.assertIn("history", data)

    def test_export_text(self):
        out = self.log.export_conversation(self.conv_id, "text")
        self.assertIn("Conversation", out)
        self.assertIn("Assistant: hi", out)

    def test_export_openai_format(self):
        out = self.log.export_conversation(self.conv_id, "openai")
        data = json.loads(out)
        self.assertIn("messages", data)
        self.assertEqual(data["model"], "echo")

    def test_export_invalid_format(self):
        with self.assertRaises(ValueError):
            self.log.export_conversation(self.conv_id, "xml")


class TestDefaultLogPath(unittest.TestCase):
    def test_default_log_path_constant(self):
        self.assertTrue(DEFAULT_LOG_PATH.endswith("conversation_logs.db"))

    def test_get_conversation_log_explicit_not_cached(self):
        path = _tmp_db()
        self.addCleanup(os.unlink, path)
        log = get_conversation_log(path)
        self.addCleanup(log.close)
        self.assertIsInstance(log, ConversationLog)
        self.assertEqual(log._path, path)

    def test_get_conversation_log_none_uses_default_and_caches(self):
        # When path is None, the module-level cached instance is returned.
        # We patch DEFAULT_LOG_PATH to a temp file so we don't touch ~/.local.
        fd, tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, tmp)
        with mock.patch("local_llm_kit.conversation_log.DEFAULT_LOG_PATH", tmp):
            first = get_conversation_log(None)
            self.addCleanup(first.close)
            second = get_conversation_log(None)
            self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
