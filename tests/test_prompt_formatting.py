"""
Tests for the prompt formatting module.
"""

import json
import unittest

from local_llm_kit.prompt_formatting import (
    ChatMLPromptFormatter,
    Llama2ChatPromptFormatter,
    MistralInstructPromptFormatter,
    PlainInstructPromptFormatter,
    VicunaPromptFormatter,
    get_prompt_formatter,
)


class TestPromptFormatting(unittest.TestCase):
    """Tests for prompt formatting functionality."""

    def test_formatter_selection(self):
        """Test that the correct formatter is selected for different models."""
        # Test Llama formatter
        formatter = get_prompt_formatter("meta-llama/Llama-2-7b-chat-hf")
        self.assertIsInstance(formatter, Llama2ChatPromptFormatter)

        # Test Mistral formatter
        formatter = get_prompt_formatter("mistralai/Mistral-7B-Instruct-v0.1")
        self.assertIsInstance(formatter, MistralInstructPromptFormatter)

        # Test Vicuna formatter
        formatter = get_prompt_formatter("lmsys/vicuna-7b-v1.5")
        self.assertIsInstance(formatter, VicunaPromptFormatter)

        # Test ChatML formatter
        formatter = get_prompt_formatter("HuggingFaceH4/zephyr-7b-beta")
        self.assertIsInstance(formatter, ChatMLPromptFormatter)

        # Test default formatter
        formatter = get_prompt_formatter("some-unknown-model")
        self.assertIsInstance(formatter, PlainInstructPromptFormatter)

    def test_llama2_chat_formatting(self):
        """Test Llama2 chat formatter."""
        formatter = Llama2ChatPromptFormatter()

        # Test basic formatting
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        formatted = formatter.format_messages(messages)
        self.assertIn("[INST]", formatted)
        self.assertIn("You are a helpful assistant", formatted)
        self.assertIn("Hello, how are you?", formatted)

        # Test multi-turn conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        formatted = formatter.format_messages(messages)
        self.assertIn("Hi there!", formatted)
        self.assertIn("How are you?", formatted)

        # Test with function definitions
        functions = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            }
        ]

        formatted = formatter.format_messages(messages, functions=functions)
        self.assertIn("Functions available to call", formatted)
        self.assertIn("get_weather", formatted)

    def test_function_call_formatting(self):
        """Test function call formatting."""
        formatter = ChatMLPromptFormatter()

        # Create a message with function call
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "Paris"}),
                },
            },
        ]

        formatted = formatter.format_messages(messages)
        self.assertIn("function_call", formatted)
        self.assertIn("get_weather", formatted)
        self.assertIn("Paris", formatted)


if __name__ == "__main__":
    unittest.main()
