"""Tests for OpenAICompatBackend enhancements:
- apfel fallback from /completions (HTTP 501) to /chat/completions.
- apfel embeddings support check.
- reasoning_content preservation in chat and chat_stream.
- SSE keep-alive ping parsing robustness.
"""

import unittest
from unittest.mock import patch

import pytest

from local_llm_kit import clear_backend_cache
from local_llm_kit.backends.openai_compat import OpenAICompatBackend


class TestOpenAICompatAppleAndReasoning(unittest.TestCase):
    def setUp(self):
        clear_backend_cache()

    def test_apfel_supports_embeddings_is_false(self):
        backend = OpenAICompatBackend(model="apple-foundationmodel", provider="apfel")
        assert backend.supports_embeddings is False
        with pytest.raises(NotImplementedError) as exc_info:
            backend.embed(["test"])
        assert "does not expose /v1/embeddings" in str(exc_info.value)

    def test_apfel_generate_routes_to_chat(self):
        backend = OpenAICompatBackend(model="apple-foundationmodel", provider="apfel")
        with patch.object(backend, "chat") as mock_chat:
            mock_chat.return_value = {
                "text": "Fallback chat answer",
                "finish_reason": "stop",
                "model": "apple-foundationmodel",
            }
            res = backend.generate("Hello there")
            assert res["text"] == "Fallback chat answer"
            mock_chat.assert_called_once()
            args, kwargs = mock_chat.call_args
            assert args[0] == [{"role": "user", "content": "Hello there"}]

    def test_generic_server_501_falls_back_to_chat(self):
        backend = OpenAICompatBackend(model="custom", provider="generic")
        with patch.object(
            backend,
            "_post_json",
            side_effect=[
                RuntimeError(
                    "OpenAI-compatible request POST /completions failed with HTTP 501: Not Implemented"
                ),
                {
                    "choices": [{"message": {"content": "Answer from chat"}}],
                    "finish_reason": "stop",
                },
            ],
        ):
            res = backend.generate("Hello there")
            assert res["text"] == "Answer from chat"

    def test_reasoning_content_in_chat(self):
        backend = OpenAICompatBackend(model="deepseek-r1", provider="vllm")
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Final 42",
                        "reasoning_content": "Thinking: 6 * 7 = 42",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        with patch.object(backend, "_post_json", return_value=mock_response):
            res = backend.chat([{"role": "user", "content": "What is 6*7?"}])
            assert res["text"] == "Final 42"
            assert res["reasoning_content"] == "Thinking: 6 * 7 = 42"

    def test_reasoning_content_in_chat_stream(self):
        backend = OpenAICompatBackend(model="deepseek-r1", provider="vllm")
        events = [
            {"choices": [{"delta": {"reasoning_content": "Step 1: calculate. "}}]},
            {"choices": [{"delta": {"reasoning_content": "Step 2: verify. "}}]},
            {"choices": [{"delta": {"content": "The answer is 42."}}]},
            {"choices": [{"finish_reason": "stop"}]},
        ]
        with patch.object(backend, "_post_sse", return_value=iter(events)):
            chunks = list(backend.chat_stream([{"role": "user", "content": "Calculate"}]))
            reasoning = "".join(c.get("reasoning_content", "") for c in chunks)
            content = "".join(c.get("text", "") for c in chunks)
            assert reasoning == "Step 1: calculate. Step 2: verify. "
            assert content == "The answer is 42."
            assert chunks[-1]["finish_reason"] == "stop"

    def test_sse_parser_handles_keep_alive_and_pings(self):
        backend = OpenAICompatBackend(model="test", provider="openai-compat")
        raw_stream = (
            b": keep-alive\n"
            b": ping\n"
            b"\n"
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n'
            b": comment\n"
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n'
            b"data: [DONE]\n"
        )

        class FakeHTTPResponse:
            def __init__(self, data):
                self.lines = data.splitlines(keepends=True)

            def __enter__(self):
                return iter(self.lines)

            def __exit__(self, *args):
                pass

        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(raw_stream)):
            events = list(backend._post_sse("/chat/completions", {}))
            assert len(events) == 2
            assert events[0]["choices"][0]["delta"]["content"] == "Hello"
            assert events[1]["choices"][0]["delta"]["content"] == " world"
