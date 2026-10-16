"""Tests for the Apple Foundation Models backend (apple-fm).

Covers:
  - Platform / OS / architecture prerequisite checks.
  - Python version requirements.
  - Missing optional dependency hints.
  - Availability checking (disabled, not ready).
  - Contract testing for generate, generate_stream, chat, chat_stream.
  - Cumulative snapshot to incremental delta conversion.
  - Context window floor (cold-start safeguard).
  - Token counting and unsupported embeddings handling.
  - Resource cleanup on close.
  - High-level LLM auto-detection and apfel bridge resolution.
  - Opt-in real-hardware integration test.
"""

import os
import sys
import unittest
from unittest.mock import patch

import pytest

from local_llm_kit import LLM, clear_backend_cache
from local_llm_kit.backends import BACKEND_ALIASES, BACKENDS, OPENAI_SERVER_URLS
from local_llm_kit.backends.apple_fm import _DEFAULT_CONTEXT_FLOOR, AppleFMBackend


class FakeModel:
    def __init__(self, is_available=True, reason="Model ready", context_size=8192):
        self._is_available = is_available
        self._reason = reason
        self.context_size = context_size

    def is_available(self):
        return self._is_available, self._reason

    async def token_count(self, text):
        return len(text.split())


class FakeSession:
    def __init__(self, instructions=None, model=None):
        self.instructions = instructions
        self.model = model

    async def respond(self, prompt, options=None):
        return "Apple FM response to: " + prompt

    async def stream_response(self, prompt, options=None):
        # apple_fm_sdk yields cumulative snapshots
        parts = ["Apple ", "Apple FM ", "Apple FM response ", "Apple FM response to: " + prompt]
        for part in parts:
            yield part


class FakeAppleFMSDK:
    def __init__(self, is_available=True, reason="Model ready", context_size=8192):
        self._is_available = is_available
        self._reason = reason
        self._context_size = context_size

    def SystemLanguageModel(self):
        return FakeModel(
            is_available=self._is_available,
            reason=self._reason,
            context_size=self._context_size,
        )

    def LanguageModelSession(self, instructions=None, model=None):
        return FakeSession(instructions=instructions, model=model)

    class GenerationOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class SamplingMode:
        @staticmethod
        def random(top_p=0.9):
            return {"top_p": top_p}


class TestAppleFMBackendPrerequisites(unittest.TestCase):
    def test_non_darwin_raises_runtime_error(self):
        with patch("sys.platform", "linux"):
            with pytest.raises(RuntimeError) as exc_info:
                AppleFMBackend._check_platform_prerequisites()
            assert "only supported on macOS" in str(exc_info.value)

    def test_non_arm64_raises_runtime_error(self):
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="x86_64"):
            with pytest.raises(RuntimeError) as exc_info:
                AppleFMBackend._check_platform_prerequisites()
            assert "requires Apple Silicon" in str(exc_info.value)

    def test_python_under_310_raises_runtime_error(self):
        with (
            patch("sys.platform", "darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("sys.version_info", (3, 9, 6)),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                AppleFMBackend._check_platform_prerequisites()
            assert "requires Python 3.10+" in str(exc_info.value)
            assert "Apfel" in str(exc_info.value)

    def test_missing_dependency_raises_import_error(self):
        with patch.object(AppleFMBackend, "_check_platform_prerequisites", return_value=None):
            backend = AppleFMBackend()
            with patch.dict(sys.modules, {"apple_fm_sdk": None}):
                with pytest.raises(ImportError) as exc_info:
                    backend._get_sdk()
                assert "local-llm-kit[apple-fm]" in str(exc_info.value)

    def test_model_unavailable_raises_runtime_error(self):
        fake_sdk = FakeAppleFMSDK(is_available=False, reason="Assets downloading (42%)")
        with patch.object(AppleFMBackend, "_check_platform_prerequisites", return_value=None):
            backend = AppleFMBackend()
            backend._sdk = fake_sdk
            with pytest.raises(RuntimeError) as exc_info:
                backend._get_model()
            assert "Apple Foundation Model is currently unavailable" in str(exc_info.value)
            assert "Assets downloading" in str(exc_info.value)


class TestAppleFMBackendContract(unittest.TestCase):
    def setUp(self):
        clear_backend_cache()
        self.fake_sdk = FakeAppleFMSDK(context_size=8192)
        self.patcher = patch.object(
            AppleFMBackend, "_check_platform_prerequisites", return_value=None
        )
        self.patcher.start()
        self.backend = AppleFMBackend()
        self.backend._sdk = self.fake_sdk

    def tearDown(self):
        self.backend.close()
        self.patcher.stop()

    def test_generate_sync(self):
        res = self.backend.generate("Hello world")
        assert isinstance(res, dict)
        assert res["text"] == "Apple FM response to: Hello world"
        assert res["finish_reason"] == "stop"
        assert res["model"] == "apple-foundationmodel"
        assert "usage" in res
        assert res["usage"]["prompt_tokens"] > 0

    def test_generate_stream_incremental_deltas(self):
        chunks = list(self.backend.generate_stream("Tell a tale"))
        deltas = [c["text"] for c in chunks if c.get("text")]
        # Cumulative snapshots in FakeSession were:
        # ["Apple ", "Apple FM ", "Apple FM response ", "Apple FM response to: Tell a tale"]
        # Incremental deltas should reconstruct the full text without repeats:
        full = "".join(deltas)
        assert full == "Apple FM response to: Tell a tale"
        assert any(c.get("finish_reason") == "stop" for c in chunks)

    def test_chat_and_chat_stream(self):
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        res = self.backend.chat(messages)
        assert res["text"] == "Apple FM response to: What is 2+2?"
        assert res["finish_reason"] == "stop"

        stream_chunks = list(self.backend.chat_stream(messages))
        streamed_text = "".join(c.get("text", "") for c in stream_chunks)
        assert streamed_text == "Apple FM response to: What is 2+2?"
        assert stream_chunks[-1].get("finish_reason") == "stop"

    def test_context_window_and_floor(self):
        assert self.backend.get_context_window() == 8192

        # Cold start returns 0 on macOS 27:
        zero_sdk = FakeAppleFMSDK(context_size=0)
        backend2 = AppleFMBackend()
        backend2._sdk = zero_sdk
        assert backend2.get_context_window() == _DEFAULT_CONTEXT_FLOOR

    def test_count_tokens(self):
        count = self.backend.count_tokens("one two three four five")
        assert count == 5

    def test_embed_unsupported(self):
        with pytest.raises(NotImplementedError) as exc_info:
            self.backend.embed(["hello"])
        assert "does not expose embedding vectors" in str(exc_info.value)
        assert "sentence-transformers" in str(exc_info.value)

    def test_model_info(self):
        info = self.backend.model_info()
        assert info["backend"] == "apple-fm"
        assert info["on_device"] is True
        assert info["zero_download"] is True
        assert info["context_window"] == 8192


class TestAppleIntegrationRegistry(unittest.TestCase):
    def setUp(self):
        clear_backend_cache()

    def test_backend_registry_and_aliases(self):
        assert "apple-fm" in BACKENDS
        assert BACKEND_ALIASES["apfel"] == "openai-compat"
        assert BACKEND_ALIASES["apple-server"] == "openai-compat"
        assert BACKEND_ALIASES["apple"] == "apple-fm"
        assert BACKEND_ALIASES["apple-foundation"] == "apple-fm"
        assert BACKEND_ALIASES["apple-intelligence"] == "apple-fm"

        assert OPENAI_SERVER_URLS["apfel"] == "http://127.0.0.1:11434/v1"
        assert OPENAI_SERVER_URLS["apple-server"] == "http://127.0.0.1:11434/v1"

    def test_llm_auto_backend_for_apple(self):
        fake_sdk = FakeAppleFMSDK()
        with patch.object(AppleFMBackend, "_check_platform_prerequisites", return_value=None):
            with patch(
                "local_llm_kit.backends.apple_fm.AppleFMBackend._get_sdk", return_value=fake_sdk
            ):
                llm = LLM(model_path="apple", backend="auto", use_cache=False)
                assert llm.backend.backend_name == "apple-fm"

    def test_llm_apfel_preset_url(self):
        llm = LLM(model_path="apple-foundationmodel", backend="apfel", use_cache=False)
        assert llm.backend.base_url == "http://127.0.0.1:11434/v1"
        assert llm.backend.provider == "apfel"
        assert llm.backend.supports_embeddings is False


class TestAppleRealHardwareOptIn(unittest.TestCase):
    @pytest.mark.skipif(
        os.environ.get("LOCAL_LLM_TEST_REAL_APPLE") != "1",
        reason="Opt-in test: set LOCAL_LLM_TEST_REAL_APPLE=1 on eligible Apple Silicon Mac to run real inference.",
    )
    def test_real_apple_intelligence_inference(self):
        # Only runs when explicitly requested
        backend = AppleFMBackend()
        res = backend.generate("Hello from Local LLM Kit real Apple test.")
        assert len(res["text"]) > 0
        assert res["finish_reason"] == "stop"
