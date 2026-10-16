"""Regression tests for the AirLLM ``generate_stream`` streaming contract.

AirLLM's underlying APIs do not expose a stable per-token callback, so the
adapter streams a single buffered chunk. These tests lock that contract:
``generate_stream`` must be a generator yielding exactly one chunk with the
standard ``text``/``finish_reason`` shape that streaming consumers expect.

They reuse the lightweight fakes (``FakeAirModel`` / ``Tokenizer`` / ``Tensor``)
defined in ``test_local_backends`` so decoded output stays consistent.
"""

import inspect
import sys
import types
import unittest
from unittest import mock

from local_llm_kit.backends.airllm import AirLLMBackend
from tests.test_local_backends import FakeAirModel


class TestAirLLMLStreaming(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeAirModel()
        self.auto = mock.Mock()
        self.auto.from_pretrained.return_value = self.model
        patcher = mock.patch.dict(
            sys.modules, {"airllm": types.SimpleNamespace(AutoModel=self.auto)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _backend(self, **kwargs):
        return AirLLMBackend("/trusted/model", device="cpu", **kwargs)

    def test_generate_stream_is_a_generator(self) -> None:
        backend = self._backend()
        result = backend.generate_stream("question", max_new_tokens=5)
        self.assertTrue(inspect.isgenerator(result))
        self.assertNotIsInstance(result, dict)

    def test_generate_stream_yields_single_standard_chunk(self) -> None:
        backend = self._backend()
        sync_result = backend.generate("question", max_new_tokens=5)
        chunks = list(backend.generate_stream("question", max_new_tokens=5))
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(set(chunk.keys()), {"text", "finish_reason"})
        self.assertEqual(chunk["text"], sync_result["text"])
        self.assertEqual(chunk["finish_reason"], sync_result["finish_reason"])

    def test_streamed_chunk_matches_buffered_generate(self) -> None:
        backend = self._backend()
        sync_result = backend.generate("question", max_new_tokens=5)
        streamed = list(backend.generate_stream("question", max_new_tokens=5))
        self.assertEqual(streamed[0]["text"], sync_result["text"])
        self.assertEqual(streamed[0]["finish_reason"], sync_result["finish_reason"])

    def test_generate_stream_zero_tokens_yields_empty_chunk(self) -> None:
        backend = self._backend()
        # generate() short-circuits max_new_tokens == 0 to empty text / "length".
        chunks = list(backend.generate_stream("question", max_new_tokens=0))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "")
        self.assertEqual(chunks[0]["finish_reason"], "length")

    def test_generate_stream_does_not_leak_extra_keys(self) -> None:
        backend = self._backend()
        chunks = list(backend.generate_stream("question", max_new_tokens=5))
        # Streaming consumers must not have to defend against model/raw/usage.
        self.assertNotIn("model", chunks[0])
        self.assertNotIn("raw", chunks[0])


if __name__ == "__main__":
    unittest.main()
