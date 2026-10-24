"""Transformers context and streaming contracts, without model downloads or torch."""

import contextlib
import importlib
import queue
import subprocess
import sys
import types
import unittest
from unittest import mock

from local_llm_kit import LLM


class InputIds(list):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, length=2):
        self.length = length

    def __call__(self, prompt, **kwargs):
        return {"input_ids": InputIds([list(range(self.length))])}

    def encode(self, text):
        return list(range(self.length))

    def decode(self, tokens, **kwargs):
        return "answer"


class TextStreamer:
    """The HF contract: token tensors enter put(); iteration yields text deltas."""

    def __init__(self, tokenizer, skip_prompt=False, **kwargs):
        self.skip_prompt = skip_prompt
        self.items = queue.Queue()

    def put(self, value):
        if not isinstance(value, InputIds):
            raise TypeError("put() requires token tensors")
        if not self.skip_prompt:
            self.on_finalized_text("prompt ")

    def on_finalized_text(self, text, stream_end=False):
        self.items.put(text)
        if stream_end:
            self.items.put(None)

    def end(self):
        self.on_finalized_text("", stream_end=True)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.items.get(timeout=2)
        if value is None:
            raise StopIteration
        return value


class TestTransformersContextAndStreaming(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("local_llm_kit.backends.transformers")
        patcher = mock.patch.dict(
            self.module.__dict__,
            {
                "torch": types.SimpleNamespace(no_grad=contextlib.nullcontext),
                "transformers": types.SimpleNamespace(TextIteratorStreamer=TextStreamer),
                "TextIteratorStreamer": TextStreamer,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.backend = object.__new__(self.module.TransformersBackend)
        self.backend.device = "cpu"
        self.backend.model_path = "test-model"
        self.backend.stop_token_ids = []
        self.backend.tokenizer = Tokenizer()
        self.backend.model = types.SimpleNamespace(
            config=types.SimpleNamespace(max_position_embeddings=8),
            generate=mock.Mock(side_effect=self.generate),
        )

    def generate(self, input_ids, **kwargs):
        if len(input_ids[0]) + kwargs["max_new_tokens"] > 8:
            raise IndexError("index out of range in self")
        streamer = kwargs.get("streamer")
        if streamer is not None:
            streamer.put(input_ids)
            streamer.on_finalized_text("Hello ")
            streamer.on_finalized_text("world!")
            streamer.end()
        return types.SimpleNamespace(sequences=[input_ids[0] + [9]])

    def test_oversized_prompt_is_rejected_before_model_or_stream_worker(self):
        self.backend.tokenizer.length = 9
        for stream in (False, True):
            with (
                self.subTest(stream=stream),
                mock.patch.object(self.module.threading, "Thread") as worker,
            ):
                with self.assertRaisesRegex(ValueError, "context window.*8"):
                    result = self.backend.generate("prompt", max_new_tokens=1, stream=stream)
                    if stream:
                        list(result)
                self.backend.model.generate.assert_not_called()
                worker.assert_not_called()

    def test_output_budget_is_included_in_context_validation(self):
        self.backend.tokenizer.length = 7
        for generate in (self.backend.generate, self.backend.generate_stream):
            with self.subTest(method=generate.__name__):
                with self.assertRaisesRegex(ValueError, "context window"):
                    result = generate("prompt", max_new_tokens=2)
                    if generate.__name__ == "generate_stream":
                        list(result)
        self.backend.model.generate.assert_not_called()

    def test_exact_context_boundary_still_generates(self):
        self.backend.tokenizer.length = 6
        self.assertEqual(self.backend.generate("prompt", max_new_tokens=2)["text"], "answer")
        chunks = list(self.backend.generate_stream("prompt", max_new_tokens=2))
        self.assertEqual("".join(chunk["text"] for chunk in chunks), "Hello world!")
        self.assertEqual(self.backend.model.generate.call_count, 2)

    def test_stream_preserves_deltas_and_excludes_the_prompt(self):
        chunks = list(self.backend.generate_stream("prompt", max_new_tokens=2))
        self.assertEqual([chunk["text"] for chunk in chunks if chunk["text"]], ["Hello ", "world!"])

    def test_worker_error_is_raised_in_the_consumer(self):
        failure = RuntimeError("generation failed")
        self.backend.model.generate.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            list(self.backend.generate_stream("prompt", max_new_tokens=2))
        self.assertIs(caught.exception, failure)

    def test_error_after_stream_end_is_not_silently_accepted(self):
        failure = RuntimeError("finalization failed")

        def generate(input_ids, **kwargs):
            kwargs["streamer"].end()
            raise failure

        self.backend.model.generate.side_effect = generate
        with self.assertRaises(RuntimeError) as caught:
            list(self.backend.generate_stream("prompt", max_new_tokens=2))
        self.assertIs(caught.exception, failure)

    def test_public_chat_and_completion_use_the_same_guard(self):
        self.backend.tokenizer.length = 9
        client = LLM("test-model", backend_instance=self.backend, use_cache=False)
        for stream in (False, True):
            for method, prompt in (
                (client.complete, "prompt"),
                (client.chat, [{"role": "user", "content": "prompt"}]),
            ):
                with self.subTest(method=method.__name__, stream=stream):
                    with self.assertRaisesRegex(ValueError, "context window"):
                        result = method(prompt, stream=stream, max_tokens=1)
                        if stream:
                            list(result)
        self.backend.model.generate.assert_not_called()

    def test_missing_optional_dependencies_raise_the_install_hint(self):
        code = """
import sys
sys.modules['torch'] = None
from local_llm_kit.backends.transformers import TransformersBackend
try:
    TransformersBackend('unused')
except ImportError as error:
    assert 'pip install transformers' in str(error)
else:
    raise AssertionError('Missing torch must not load a model')
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)
