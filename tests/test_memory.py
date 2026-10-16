"""Tests for MessageHistory memory management and token counting."""

import unittest

from local_llm_kit.memory import MessageHistory


class TestMessageHistory(unittest.TestCase):
    def test_basic_message_history(self):
        history = MessageHistory(context_window=1000)
        history.add_message({"role": "user", "content": "Hello"})
        history.add_message({"role": "assistant", "content": "Hi there"})
        messages = history.get_messages()
        self.assertEqual(len(messages), 2)
        history.clear()
        self.assertEqual(len(history.get_messages()), 0)

    def test_count_message_tokens_with_tool_calls(self):
        history = MessageHistory(context_window=1000)
        # Assistant message with tool_calls
        msg_tool_calls = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city": "San Francisco", "state": "CA"}',
                    },
                }
            ],
        }
        tokens_tool_call = history._count_message_tokens(msg_tool_calls)
        # Just role would be ~1 token; with tool calls and args it should be substantially higher
        self.assertGreater(tokens_tool_call, 10)

    def test_count_message_tokens_with_function_call(self):
        history = MessageHistory(context_window=1000)
        msg_func = {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "lookup_weather",
                "arguments": '{"city": "Paris"}',
            },
        }
        tokens = history._count_message_tokens(msg_func)
        self.assertGreater(tokens, 5)

    def test_count_message_tokens_with_image_parts(self):
        history = MessageHistory(context_window=1000)
        msg_parts = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
        tokens = history._count_message_tokens(msg_parts)
        self.assertGreater(tokens, 5)


if __name__ == "__main__":
    unittest.main()
