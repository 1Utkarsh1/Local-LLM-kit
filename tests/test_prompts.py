"""Offline tests for prompt formatting (local_llm_kit.prompt_formatting).

Covers: get_prompt_formatter routing (incl. Llama3/Qwen/Gemma/Phi/DeepSeek),
message_text with vision parts, has_images, and every formatter's
format_messages. Stdlib unittest only; no model downloads. Py3.9 compatible.

Run: python -m pytest tests/test_prompts.py -q
"""
import unittest

from local_llm_kit.prompt_formatting import (
    BasePromptFormatter,
    ChatMLPromptFormatter,
    DeepSeekChatPromptFormatter,
    GemmaChatPromptFormatter,
    Llama2ChatPromptFormatter,
    Llama3ChatPromptFormatter,
    MistralInstructPromptFormatter,
    PhiChatPromptFormatter,
    PlainInstructPromptFormatter,
    QwenChatPromptFormatter,
    VicunaPromptFormatter,
    get_prompt_formatter,
    has_images,
    message_text,
)

BASIC_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]

VISION_MESSAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
    ],
}

ALL_FORMATTERS = [
    ChatMLPromptFormatter,
    Llama2ChatPromptFormatter,
    MistralInstructPromptFormatter,
    VicunaPromptFormatter,
    PlainInstructPromptFormatter,
    Llama3ChatPromptFormatter,
    QwenChatPromptFormatter,
    GemmaChatPromptFormatter,
    PhiChatPromptFormatter,
    DeepSeekChatPromptFormatter,
]


class TestRouting(unittest.TestCase):
    def test_modern_families(self):
        cases = [
            ("meta-llama/Llama-3-8B-Instruct", Llama3ChatPromptFormatter),
            ("meta-llama/Llama-3.1-70B-Instruct", Llama3ChatPromptFormatter),
            ("llama3-8b", Llama3ChatPromptFormatter),
            ("Qwen/Qwen2.5-7B-Instruct", QwenChatPromptFormatter),
            ("Qwen/Qwen3-8B", QwenChatPromptFormatter),
            ("QwQ-32B", QwenChatPromptFormatter),
            ("google/gemma-2-9b-it", GemmaChatPromptFormatter),
            ("google/gemma-3-4b-it", GemmaChatPromptFormatter),
            ("microsoft/Phi-3-mini-4k-instruct", PhiChatPromptFormatter),
            ("microsoft/phi-4", PhiChatPromptFormatter),
            ("deepseek-ai/DeepSeek-V3", DeepSeekChatPromptFormatter),
            ("deepseek-r1-distill-llama-8b", DeepSeekChatPromptFormatter),
        ]
        for model_name, cls in cases:
            with self.subTest(model=model_name):
                self.assertIsInstance(get_prompt_formatter(model_name), cls)

    def test_legacy_families(self):
        cases = [
            ("meta-llama/Llama-2-7b-chat-hf", Llama2ChatPromptFormatter),
            ("mistralai/Mistral-7B-Instruct-v0.1", MistralInstructPromptFormatter),
            ("Mixtral-8x7B-Instruct-v0.1", MistralInstructPromptFormatter),
            ("lmsys/vicuna-7b-v1.5", VicunaPromptFormatter),
            ("HuggingFaceH4/zephyr-7b-beta", ChatMLPromptFormatter),
            ("tiiuae/falcon-7b", ChatMLPromptFormatter),
            ("some-unknown-model", PlainInstructPromptFormatter),
            ("my-custom-model", PlainInstructPromptFormatter),
            ("meta-llama/Llama-2-7b-hf", PlainInstructPromptFormatter),  # non-chat Llama2
        ]
        for model_name, cls in cases:
            with self.subTest(model=model_name):
                self.assertIsInstance(get_prompt_formatter(model_name), cls)

    def test_routing_is_case_insensitive(self):
        self.assertIsInstance(get_prompt_formatter("META-LLAMA/LLAMA-3-8B-INSTRUCT"), Llama3ChatPromptFormatter)
        self.assertIsInstance(get_prompt_formatter("QWEN2.5-7B"), QwenChatPromptFormatter)
        self.assertIsInstance(get_prompt_formatter("DEEPSEEK-V3"), DeepSeekChatPromptFormatter)


class TestMessageText(unittest.TestCase):
    def test_plain_string_passes_through(self):
        self.assertEqual(message_text({"role": "user", "content": "hi"}), "hi")

    def test_missing_content_gives_empty_string(self):
        self.assertEqual(message_text({"role": "user"}), "")

    def test_text_parts_are_joined(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]}
        self.assertEqual(message_text(msg), "hello\nworld")

    def test_image_url_part_becomes_placeholder(self):
        self.assertEqual(message_text(VISION_MESSAGE), "What is in this image?\n[image]")

    def test_image_type_part_becomes_placeholder(self):
        msg = {"role": "user", "content": [{"type": "image", "data": "AAA"}]}
        self.assertIn("[image]", message_text(msg))

    def test_non_dict_parts_are_stringified(self):
        msg = {"role": "user", "content": ["hello"]}
        self.assertEqual(message_text(msg), "hello")


class TestHasImages(unittest.TestCase):
    def test_true_for_image_url_parts(self):
        self.assertTrue(has_images([VISION_MESSAGE]))

    def test_false_for_text_only(self):
        self.assertFalse(has_images([{"role": "user", "content": "hi"}]))
        self.assertFalse(has_images([]))


class TestAllFormattersSmoke(unittest.TestCase):
    def test_every_formatter_handles_basic_messages(self):
        for cls in ALL_FORMATTERS:
            with self.subTest(formatter=cls.__name__):
                out = cls().format_messages(list(BASIC_MESSAGES))
                self.assertIsInstance(out, str)
                self.assertIn("Hello, how are you?", out)

    def test_every_formatter_accepts_functions_and_json_mode(self):
        functions = [{"name": "get_weather", "description": "Get weather",
                      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}]
        for cls in ALL_FORMATTERS:
            with self.subTest(formatter=cls.__name__):
                out = cls().format_messages(list(BASIC_MESSAGES), functions=functions)
                self.assertIn("get_weather", out)
                out_json = cls().format_messages(list(BASIC_MESSAGES), json_mode=True)
                self.assertIn("json", out_json.lower())

    def test_every_formatter_handles_vision_content_without_crashing(self):
        for cls in ALL_FORMATTERS:
            with self.subTest(formatter=cls.__name__):
                out = cls().format_messages([VISION_MESSAGE])
                self.assertIn("What is in this image?", out)

    def test_every_formatter_handles_function_role(self):
        messages = [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None,
             "function_call": {"name": "get_weather", "arguments": '{"location": "Paris"}'}},
            {"role": "function", "name": "get_weather", "content": '{"temp": 22}'},
        ]
        for cls in ALL_FORMATTERS:
            with self.subTest(formatter=cls.__name__):
                out = cls().format_messages(messages)
                self.assertIsInstance(out, str)


class TestIndividualFormatters(unittest.TestCase):
    def test_llama3_tokens_and_assistant_cue(self):
        out = Llama3ChatPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("<|begin_of_text|>", out)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", out)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", out)
        self.assertTrue(out.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>"))

    def test_llama3_tools_preamble_uses_tool_call_style(self):
        tools = [{"type": "function", "function": {"name": "w", "description": "d",
                 "parameters": {"type": "object", "properties": {}}}}]
        out = Llama3ChatPromptFormatter().format_messages(list(BASIC_MESSAGES), tools=tools)
        self.assertIn("<tool_call>", out)
        self.assertIn('"w"', out)

    def test_llama3_tool_role_rendered(self):
        messages = [{"role": "user", "content": "hi"},
                    {"role": "tool", "content": "result!"}]
        out = Llama3ChatPromptFormatter().format_messages(messages)
        self.assertIn("result!", out)

    def test_qwen_is_chatml_with_tool_rename(self):
        tools = [{"type": "function", "function": {"name": "w", "description": "d",
                 "parameters": {"type": "object", "properties": {}}}}]
        out = QwenChatPromptFormatter().format_messages(list(BASIC_MESSAGES), tools=tools)
        self.assertIn("<|im_start|>user", out)
        self.assertIn("Tools available to call", out)
        self.assertTrue(out.rstrip().endswith("<|im_start|>assistant"))

    def test_gemma_turn_tokens_and_model_cue(self):
        out = GemmaChatPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("<start_of_turn>user", out)
        self.assertIn("<end_of_turn>", out)
        self.assertTrue(out.endswith("<start_of_turn>model\n"))

    def test_gemma_maps_assistant_to_model(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        out = GemmaChatPromptFormatter().format_messages(messages)
        self.assertIn("<start_of_turn>model\nhello<end_of_turn>", out)

    def test_phi_pipes_and_trailing_assistant(self):
        out = PhiChatPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("<|user|>", out)
        self.assertIn("<|system|>", out)
        self.assertTrue(out.endswith("<|assistant|>\n"))

    def test_deepseek_headers(self):
        out = DeepSeekChatPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("### User:", out)
        self.assertTrue(out.endswith("### Assistant:\n"))

    def test_llama2_inst_markers(self):
        out = Llama2ChatPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("[INST]", out)
        self.assertIn("You are a helpful assistant", out)

    def test_mistral_inst_markers(self):
        out = MistralInstructPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("[INST]", out)
        self.assertIn("Hello, how are you?", out)

    def test_vicuna_roles(self):
        out = VicunaPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("HUMAN:", out)
        self.assertIn("ASSISTANT:", out)

    def test_chatml_function_call_rendering(self):
        import json as _json
        messages = [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None,
             "function_call": {"name": "get_weather",
                               "arguments": _json.dumps({"location": "Paris"})}},
        ]
        out = ChatMLPromptFormatter().format_messages(messages)
        self.assertIn("<|im_start|>assistant", out)
        self.assertIn("get_weather", out)
        self.assertIn("Paris", out)

    def test_plain_labels(self):
        out = PlainInstructPromptFormatter().format_messages(list(BASIC_MESSAGES))
        self.assertIn("User:", out)
        self.assertIn("Assistant:", out)

    def test_base_formatter_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            BasePromptFormatter().format_messages(list(BASIC_MESSAGES))


if __name__ == "__main__":
    unittest.main()
